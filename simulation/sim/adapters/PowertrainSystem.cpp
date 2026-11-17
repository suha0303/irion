#include "aerodynamics.hpp"
#include "battery_aging.hpp"
#include "command.hpp"
#include "control.hpp"
#include "fault_schedule.hpp"
#include "joint_phase.hpp"
#include "powertrain.hpp"
#include "propulsion_health.hpp"
#include "sensor.hpp"
#include "wind.hpp"
#include "bounded_wind.hpp"
#include <cstdlib>
#include <fstream>
#include <gz/msgs/Utility.hh>
#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <iomanip>
#include <mutex>
#include <optional>
#include <random>

namespace x500 {
class PowertrainSystem : public gz::sim::System,
                         public gz::sim::ISystemConfigure,
                         public gz::sim::ISystemPreUpdate {
  gz::sim::Link link;
  gz::transport::Node node;
  gz::transport::Node::Publisher battery;
  Powertrain core;
  CommandGate command_gate;
  BatterySensor sensor;
  bool write_traces = true;
  bool command_was_fresh = false;
  bool envelope_ok = true;
  unsigned rejected_commands = 0;
  int bench_axis = -1;
  std::mutex mutex;
  std::ofstream truth, control_events, command_trace, sensor_trace, wind_trace;
  uint64_t wind_update_index = 0;
  Changes changes;
  MissionFaultSchedule mission_fault;
  BatteryTransition battery_transition;
  PropulsionTransition propulsion_transition;
  bool battery_profile_locked = false;
  bool propulsion_profile_locked = false;
  gz::math::Vector3d wind{}, gust{};
  Wind turbulence{17, 0};
  JointWeatherClock joint_clock;
  bool joint_phase_enabled = false;
  std::ofstream joint_wind_trace;
  double weather_time = 0;
  std::optional<BoundedWind> bounded;
  uint64_t bounded_samples = 0, wind_force_evaluations = 0;
  double bounded_max_norm = 0, bounded_min_norm = 1e100, bounded_max_step_delta = 0;
  gz::math::Vector3d previous_bounded_wind{};
  double time = 0, last_log = -1, last_wind = -1;
  double ambient = 298.15, fault_at = -1, fault_value = 1;
  std::string fault;
  bool applied = false;
  long long applied_at = -1, boundary_at = -1;
  double ramp_s = 0, base_friction = 0;
  static double env(const char *name, double fallback) {
    const char *v = std::getenv(name);
    return v ? std::stod(v) : fallback;
  }

public:
  void Configure(const gz::sim::Entity &entity,
                 const std::shared_ptr<const sdf::Element> &settings,
                 gz::sim::EntityComponentManager &ecm,
                 gz::sim::EventManager &) override {
    if (!settings->HasElement("profile_sha256") ||
        settings->Get<std::string>("profile_sha256") != profile::sha256)
      throw std::runtime_error("profile and compiled model mismatch");
    bench_axis = static_cast<int>(env("X500_ROTATION_TEST", -1));
    sensor = BatterySensor(static_cast<unsigned>(env("X500_SEED", 17)) + 1000);
    sensor.cfg.period_s = settings->Get<double>("period_s");
    sensor.cfg.delay_s = settings->Get<double>("delay_s");
    sensor.cfg.voltage_bias_v = settings->Get<double>("voltage_bias_v");
    sensor.cfg.current_bias_a = settings->Get<double>("current_bias_a");
    sensor.cfg.voltage_noise_std_v =
        settings->Get<double>("voltage_noise_std_v");
    sensor.cfg.current_noise_std_a =
        settings->Get<double>("current_noise_std_a");
    sensor.cfg.drift_std_v = settings->Get<double>("drift_std_v");
    sensor.cfg.drift_tau_s = settings->Get<double>("drift_tau_s");
    sensor.cfg.quantization_v = settings->Get<double>("quantization_v");
    sensor.cfg.voltage_min = settings->Get<double>("voltage_min");
    sensor.cfg.voltage_max = settings->Get<double>("voltage_max");
    sensor.cfg.current_min = settings->Get<double>("current_min");
    sensor.cfg.current_max = settings->Get<double>("current_max");
    sensor.cfg.drop_begin_s = settings->Get<double>("drop_begin_s");
    sensor.cfg.drop_end_s = settings->Get<double>("drop_end_s");
    gz::sim::Model model(entity);
    link = gz::sim::Link(model.LinkByName(ecm, "base_link"));
    link.EnableVelocityChecks(ecm);
    const auto name = model.Name(ecm);
    const char *battery_file = std::getenv("X500_BATTERY_FILE");
    if (!battery_file)
      throw std::runtime_error("explicit measured battery profile required");
    BatterySnapshot::read(battery_file).install(core.p);
    const char *prop_file = std::getenv("X500_PROPULSION_FILE");
    if (prop_file)
      PropulsionSnapshot::read(prop_file).install(core.p);
    propulsion_profile_locked = env("X500_PROPULSION_PROFILE_LOCKED", 0) == 1;
    if (propulsion_profile_locked) {
      if (!prop_file)
        throw std::invalid_argument("locked propulsion requires a snapshot");
      for (const char *key : {"X500_MOTOR_R_MULTIPLIER", "X500_KT_MULTIPLIER",
                              "X500_KQ_MULTIPLIER", "X500_FRICTION"})
        if (std::getenv(key))
          throw std::invalid_argument("scalar propulsion overrides retired; select a joint profile");
    }
    core.state.soc = env("X500_SOC", .9);
    ambient = env("X500_TEMPERATURE", 298.15);
    core.state.temperature = ambient;
    core.state.motor_temperature.fill(ambient);
    core.p.resistance *= env("X500_MOTOR_R_MULTIPLIER", 1);
    core.p.kt_prop *= env("X500_KT_MULTIPLIER", 1);
    core.p.kq_prop *= env("X500_KQ_MULTIPLIER", 1);
#ifdef DRONE_PROFILE_HEADER
    core.p.battery_heat_capacity = profile::battery_heat_capacity;
    core.p.battery_cooling = profile::battery_cooling;
#endif
    core.p.friction = env("X500_FRICTION", core.p.friction);
    core.p.validate();
    base_friction = core.p.friction;
    ramp_s = env("X500_FAULT_RAMP_S", 0);
    turbulence = Wind(static_cast<unsigned>(env("X500_SEED", 17)),
                      env("X500_GUST_SIGMA", 0));
    wind.Set(env("X500_WIND_E", 0), env("X500_WIND_N", 0), 0);
    if (const char *environment = std::getenv("X500_WIND_ENVIRONMENT")) {
      if (wind.Length() != 0 || turbulence.sigma != 0)
        throw std::invalid_argument("bounded and legacy wind cannot be mixed");
      bounded.emplace(static_cast<unsigned>(env("X500_SEED", 17)), environment,
                      env("X500_WIND_DIRECTION_DEG", 0), env("X500_WIND_VARIATION", 1));
      auto initial = bounded->sample(0);
      wind.Set(initial[0], initial[1], initial[2]);
      previous_bounded_wind = wind;
    }
    joint_phase_enabled = std::getenv("X500_JOINT_ORIGIN") != nullptr;
    if (joint_phase_enabled && !bounded) throw std::invalid_argument("joint phase requires bounded wind");
    fault_at = env("X500_FAULT_AT", -1);
    fault_value = env("X500_FAULT_VALUE", 1.4);
    const char *f = std::getenv("X500_FAULT");
    if (f)
      fault = f;
    if (propulsion_profile_locked && fault == "friction")
      throw std::invalid_argument("friction-only fault retired; select joint propulsion profile");
    battery_profile_locked = env("X500_BATTERY_PROFILE_LOCKED", 0) == 1;
    if (battery_profile_locked) {
      if (!battery_file)
        throw std::invalid_argument("locked battery requires a snapshot file");
      auto snapshot = BatterySnapshot::read(battery_file);
      // Reject conflicting environment overrides of the selected initial state.
      if (core.p.r0 != snapshot.r0 || core.p.r1 != snapshot.r1 ||
          core.p.c1 != snapshot.c1 || core.p.capacity_ah != snapshot.capacity)
        throw std::invalid_argument("battery environment overrides the selected snapshot");
    }
    if (fault == "battery_stage") {
      const char *target = std::getenv("X500_BATTERY_TARGET_FILE");
      if (!battery_profile_locked || !target)
        throw std::invalid_argument("battery transition requires locked initial/target snapshots");
      battery_transition.configure(core, BatterySnapshot::read(target), ramp_s);
    }
    if (fault == "propulsion_profile") {
      const char *target = std::getenv("X500_PROPULSION_TARGET_FILE");
      if (!propulsion_profile_locked || !target)
        throw std::invalid_argument("propulsion transition requires locked endpoints");
      propulsion_transition.configure(core, PropulsionSnapshot::read(target), ramp_s);
    }
    if (const char *offset = std::getenv("X500_FAULT_AFTER_MISSION_S"))
      mission_fault.configure(std::stod(offset), fault_at, fault);
    const char *path = std::getenv("X500_TRUTH_PATH");
    if (!path)
      throw std::runtime_error("X500_TRUTH_PATH required");
    // Raw MAVLink collection does not need simulator-only traces.
    write_traces = std::string(path) != "/dev/null";
    if (write_traces) {
      const auto trace_path = [path](const char *suffix) {
        return std::string(path) + suffix;
      };
      truth.open(path);
      if (joint_phase_enabled) {
        joint_wind_trace.open(trace_path(".joint-wind.csv"));
        joint_wind_trace << "source_us,weather_seconds,east,north,up\n" << std::setprecision(17);
        if (!joint_wind_trace) throw std::runtime_error("joint wind trace open");
      }
      wind_trace.open(trace_path(".wind-trace.jsonl"));
      if (bounded) {
        double center = bounded->environment == "calm" ? 0 : bounded->environment == "low" ? 1.25 : 3.75;
        double amplitude = bounded->environment == "calm" ? 0 : 1;
        wind_trace << std::setprecision(17)
                   << "{\"schema\":\"drone.bounded-wind-trace/" << (joint_phase_enabled ? "2" : "1") << "\",\"kind\":\"configuration\",\"seed_u32\":"
                   << static_cast<unsigned>(env("X500_SEED", 17))
                   << ",\"profile\":{\"revision\":\"bounded-latent-ou-c2-enu-v1\",\"environment\":\""
                   << bounded->environment << "\",\"direction_deg\":" << bounded->direction_deg
                   << ",\"variation\":" << bounded->variation << "},\"latent_std_dimensionless\":1,\"latent_tau_s\":2,\"latent_knot_period_s\":1"
                   << ",\"speed_midpoint_mps\":" << center << ",\"speed_amplitude_mps\":" << amplitude
                   << ",\"direction_amplitude_rad\":" << BoundedWind::pi / 3
                   << ",\"elevation_amplitude_rad\":" << BoundedWind::pi / 12;
        if (joint_phase_enabled) wind_trace << ",\"phase_revision\":\"joint-native-consumption-c2-v2\"";
        wind_trace << "}\n";
      } else {
        wind_trace << std::setprecision(17)
                   << "{\"schema\":\"drone.ou-wind-trace/1\",\"kind\":\"configuration\",\"seed_u32\":"
                   << static_cast<unsigned>(env("X500_SEED", 17))
                   << ",\"sigma_per_axis_mps\":" << turbulence.sigma
                   << ",\"tau_s\":" << turbulence.tau
                   << ",\"mean_enu_mps\":[" << wind.X() << ',' << wind.Y() << ",0]}\n";
      }
      if (!wind_trace)
        throw std::runtime_error("cannot open OU wind trace");
      control_events.open(trace_path(".control-events.jsonl"));
      command_trace.open(trace_path(".command-trace.jsonl"));
      sensor_trace.open(trace_path(".sensor-trace.jsonl"));
      sensor_trace << std::setprecision(12);
      truth << std::setprecision(12);
      if (!truth)
        throw std::runtime_error("cannot open truth");
    }
    node.Subscribe("/" + name + "/command/motor_speed",
                   &PowertrainSystem::command, this);
    node.Advertise("/" + name + "/apply_parameter_change",
                   &PowertrainSystem::change, this);
    node.Advertise("/" + name + "/change_status",
                   &PowertrainSystem::change_status, this);
    node.Advertise("/" + name + "/schedule_fault",
                   &PowertrainSystem::schedule_fault, this);
    battery =
        node.Advertise<gz::msgs::Double_V>("/" + name + "/battery_sensor");
  }
  bool schedule_fault(const gz::msgs::Double_V &req, gz::msgs::Double_V &reply) {
    std::lock_guard<std::mutex> lock(mutex);
    if (req.data_size() != 2 || req.data(0) != 1) {
      reply.add_data(-1);
      return true;
    }
    try {
      const auto onset = mission_fault.schedule(
          req.data(1), static_cast<long long>(std::llround(time * 1e6)));
      fault_at = static_cast<double>(onset) / 1e6;
      reply.add_data(0);
      reply.add_data(static_cast<double>(onset));
    } catch (const std::invalid_argument &) {
      reply.add_data(-1);
    }
    return true;
  }
  bool change(const gz::msgs::Double_V &req, gz::msgs::Double_V &reply) {
    std::lock_guard<std::mutex> lock(mutex);
    if (propulsion_profile_locked) {
      reply.add_data(-1);
      return true;
    }
    if (req.data_size() != 5 || req.data(0) != 1) {
      reply.add_data(-1);
      return true;
    }
    for (int i = 0; i < 5; ++i)
      if (!std::isfinite(req.data(i))) {
        reply.add_data(-1);
        return true;
      }
    if (req.data(1) < 1 || req.data(1) > 1e9 || req.data(2) < 0 ||
        req.data(2) > 2 || req.data(4) < 0 || req.data(4) > 1e15 ||
        std::floor(req.data(1)) != req.data(1) ||
        std::floor(req.data(2)) != req.data(2)) {
      reply.add_data(-1);
      return true;
    }
    try {
      bool queued = changes.enqueue({static_cast<unsigned>(req.data(1)),
                                     static_cast<int>(req.data(2)), req.data(3),
                                     static_cast<long long>(req.data(4))});
      reply.add_data(queued ? 0 : 1);
    } catch (const std::exception &) {
      reply.add_data(-1);
    }
    return true;
  }
  bool change_status(const gz::msgs::Double_V &req, gz::msgs::Double_V &reply) {
    std::lock_guard<std::mutex> lock(mutex);
    if (req.data_size() != 1 || !std::isfinite(req.data(0)) ||
        req.data(0) < 1 || req.data(0) > 1e9) {
      reply.add_data(-1);
      return true;
    }
    try {
      const auto &r = changes.status(static_cast<unsigned>(req.data(0)));
      reply.add_data(r.applied_us);
      reply.add_data(r.revision);
    } catch (const std::out_of_range &) {
      reply.add_data(-1);
    }
    return true;
  }
  void command(const gz::msgs::Actuators &msg) {
    std::lock_guard<std::mutex> lock(mutex);
    std::string result = "malformed_protocol";
    try {
      if (msg.velocity_size() != static_cast<int>(rotor_count) || msg.header().data_size() != 4)
        throw std::invalid_argument("header");
      std::map<std::string, std::string> headers;
      for (const auto &entry : msg.header().data()) {
        if (entry.value_size() != 1 ||
            !headers.emplace(entry.key(), entry.value(0)).second)
          throw std::invalid_argument("duplicate header");
      }
      const auto &seq_text = headers.at("sequence");
      if (seq_text.empty() ||
          seq_text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("sequence");
      uint64_t seq = std::stoull(seq_text);
      const auto &stamp = msg.header().stamp();
      if (stamp.sec() < 0 || stamp.nsec() < 0 || stamp.nsec() >= 1000000000 ||
          stamp.sec() > 1000000000)
        throw std::invalid_argument("stamp");
      int64_t sample = stamp.sec() * 1000000LL + stamp.nsec() / 1000;
      Four counts{};
      for (std::size_t i = 0; i < rotor_count; ++i)
        counts[i] = msg.velocity(i);
      result = command_gate.accept(headers.at("schema"), headers.at("order"),
                                   headers.at("semantic"), seq, sample,
                                   static_cast<int64_t>(time * 1e6), counts);
    } catch (const std::exception &) {
    }
    if (result != "accepted") {
      ++rejected_commands;
      if (write_traces) {
        command_trace << "{\"kind\":\"command_rejected\",\"reason\":\"" << result
                      << "\",\"sim_us\":" << static_cast<int64_t>(time * 1e6)
                      << "}\n";
        command_trace.flush();
      }
    }
  }
  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override {
    if (info.paused || info.dt.count() <= 0)
      return;
    double t = std::chrono::duration<double>(info.simTime).count();
    double dt = std::chrono::duration<double>(info.dt).count();
    Four command;
    double scheduled_fault_at;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (t < time)
        throw std::runtime_error("clock reset requires a new run identity");
      time = t;
      scheduled_fault_at = fault_at;
      for (const auto &r :
           changes.apply(core, static_cast<long long>(t * 1e6))) {
        if (write_traces) {
          control_events << "{\"request_id\":" << r.change.id
                         << ",\"field\":" << r.change.field
                         << ",\"value\":" << r.change.value
                         << ",\"requested_time_us\":" << r.change.requested_us
                         << ",\"applied_time_us\":" << r.applied_us
                         << ",\"revision\":" << r.revision << "}\n";
          control_events.flush();
        }
      }
      bool fresh = command_gate.fresh(static_cast<int64_t>(t * 1e6));
      command = command_gate.duty;
      if (!fresh)
        command.fill(0);
      if (write_traces && fresh != command_was_fresh) {
        command_trace << "{\"kind\":\"command_state\",\"state\":\""
                      << (fresh ? "active" : "timeout_coast")
                      << "\",\"sim_us\":" << static_cast<int64_t>(t * 1e6)
                      << "}\n";
        command_trace.flush();
        command_was_fresh = fresh;
      }
      if (bench_axis >= 0)
        command = bench_axis == 2 ? Four{.4, .4, 0, 0} : Four{0, 0, 0, 0};
    }
    if (joint_phase_enabled && joint_clock.origin_us < 0) {
      std::ifstream marker(std::getenv("X500_JOINT_ORIGIN"));
      std::string record;
      if (std::getline(marker, record)) {
        const auto field = record.find("\"actual_start_us\":");
        if (field == std::string::npos) throw std::runtime_error("invalid native joint origin receipt");
        const auto origin = std::stoll(record.substr(field + 18));
        joint_clock.notice(origin, static_cast<int64_t>(std::llround(t*1e6)));
        if (mission_fault.enabled) {
          std::lock_guard<std::mutex> lock(mutex);
          const auto onset = mission_fault.schedule(static_cast<double>(origin), static_cast<int64_t>(std::llround(t*1e6)));
          fault_at = static_cast<double>(onset)/1e6; scheduled_fault_at = fault_at;
        }
        const std::string destination = std::string(std::getenv("X500_JOINT_ORIGIN"));
        const auto slash = destination.find_last_of('/');
        const std::string receipt_path = destination.substr(0, slash+1) + "joint-physics-origin.json";
        std::ofstream receipt(receipt_path + ".tmp");
        receipt << "{\"schema\":\"drone.joint-physics-origin/1\",\"origin_us\":" << origin
                << ",\"notice_us\":" << joint_clock.notice_us << ",\"fault_onset_us\":"
                << (mission_fault.enabled ? mission_fault.onset_us : -1) << "}\n";
        receipt.close();
        if (!receipt || std::rename((receipt_path+".tmp").c_str(), receipt_path.c_str())) throw std::runtime_error("joint physics receipt commit");
      }
    }
    if (scheduled_fault_at >= 0 && t >= scheduled_fault_at && (!applied || ramp_s > 0)) {
      double fraction =
          ramp_s > 0 ? std::clamp((t - scheduled_fault_at) / ramp_s, 0., 1.) : 1.;
      if (fault == "battery_stage")
        battery_transition.apply(core, t - scheduled_fault_at);
      else if (fault == "propulsion_profile")
        propulsion_transition.apply(core, t - scheduled_fault_at);
      else if (fault == "friction")
        core.p.friction =
            base_friction + fraction * (fault_value - base_friction);
      else
        throw std::runtime_error("unsupported physical fault");
      if (!applied && fraction > 0) {
        applied = true;
        applied_at = static_cast<long long>(std::llround(t * 1e6));
      }
      if (boundary_at < 0 && fraction >= .5)
        boundary_at = static_cast<long long>(std::llround(t * 1e6));
    }
    if (bounded) {
      weather_time = joint_phase_enabled ? joint_clock.phase_seconds(static_cast<int64_t>(std::llround(t*1e6))) : t;
      auto current = bounded->sample(weather_time);
      wind.Set(current[0], current[1], current[2]);
      gust.Set(0, 0, 0);
      double magnitude = wind.Length();
      bool inside = bounded->environment == "calm" ? magnitude == 0 :
                    bounded->environment == "low" ? magnitude > 0 && magnitude <= 2.5 :
                    magnitude > 2.5 && magnitude <= 5;
      if (!std::isfinite(magnitude) || !inside)
        throw std::runtime_error("bounded generator left declared wind environment");
      ++wind_force_evaluations;
      bounded_min_norm = std::min(bounded_min_norm, magnitude);
      bounded_max_norm = std::max(bounded_max_norm, magnitude);
      bounded_max_step_delta = std::max(bounded_max_step_delta, (wind - previous_bounded_wind).Length());
      previous_bounded_wind = wind;
      if (write_traces && joint_phase_enabled) {
        joint_wind_trace << static_cast<int64_t>(std::llround(t*1e6)) << ',' << weather_time << ',' << wind.X() << ',' << wind.Y() << ',' << wind.Z() << '\n';
        if (!joint_wind_trace) throw std::runtime_error("joint wind write");
      }
      last_wind = t;
    } else if (t - last_wind >= .01) {
      double h = last_wind < 0 ? .01 : t - last_wind;
      turbulence.step(h);
      for (int k = 0; k < 3; ++k)
        gust[k] = turbulence.gust[k];
      last_wind = t;
      if (write_traces) {
        ++wind_update_index;
        wind_trace << "{\"kind\":\"update\",\"index\":" << wind_update_index
                   << ",\"time_us\":" << static_cast<int64_t>(t * 1e6)
                   << ",\"dt_s\":" << h << ",\"gust_enu_mps\":["
                   << gust.X() << ',' << gust.Y() << ',' << gust.Z() << "]}\n";
        wind_trace.flush();
        if (!wind_trace)
          throw std::runtime_error("OU wind trace write failed");
      }
    }
    auto o = core.step(dt, command, true, ambient);
    auto pose = link.WorldPose(ecm);
    auto vel = link.WorldLinearVelocity(ecm);
    auto rates = link.WorldAngularVelocity(ecm);
    if (!pose || !vel || !rates)
      return;
    auto q = pose->Rot();
    auto body_rate = q.RotateVectorReverse(*rates);
    auto air = q.RotateVectorReverse(*vel - wind - gust);
    auto forces = wrench(core.p, core.state, o, {air.X(), air.Y(), air.Z()},
                         {body_rate.X(), body_rate.Y(), body_rate.Z()});
    gz::math::Vector3d force(forces.force[0], forces.force[1], forces.force[2]);
    gz::math::Vector3d moment(forces.moment[0], forces.moment[1],
                              forces.moment[2]);
    if (bench_axis >= 0) {
      force.Set(0, 0, 0);
      for (int k = 0; k < 3; ++k)
        if (k != bench_axis)
          moment[k] = 0;
      if (bench_axis < 2)
        moment[bench_axis] = .01;
    }
    link.AddWorldWrench(ecm, q.RotateVector(force), q.RotateVector(moment));
    bool valid = air.Length() <= profile::max_air_speed_mps &&
                 body_rate.Length() <= profile::max_body_rate_radps;
    const auto advance = inplane_flow(core.state, {air.X(),air.Y(),air.Z()},
                                     {body_rate.X(),body_rate.Y(),body_rate.Z()});
    if (profile::max_inplane_advance_ratio > 0)
      valid = valid && advance.defined &&
              advance.maximum_ratio <= profile::max_inplane_advance_ratio;
#ifdef DRONE_PROFILE_HEADER
    valid = valid && core.state.soc >= profile::min_soc &&
            o.voltage >= profile::min_pack_voltage_v;
#endif
    for (auto omega : core.state.omega)
      valid = valid && omega <= profile::max_rotor_speed_radps;
    envelope_ok = envelope_ok && valid;
    for (const auto &sample : sensor.step(t, o.voltage, o.current)) {
      gz::msgs::Double_V m;
      m.add_data(sample.voltage);
      m.add_data(sample.current);
      m.add_data(sample.sample_s);
      battery.Publish(m);
      if (!write_traces)
        continue;
      sensor_trace << "{\"sample_s\":" << sample.sample_s
                   << ",\"publish_s\":" << t
                   << ",\"voltage_v\":" << sample.voltage
                   << ",\"current_a\":" << sample.current
                   << ",\"physical_voltage_v\":" << sample.physical_voltage
                   << ",\"physical_current_a\":" << sample.physical_current
                   << ",\"clipped\":" << (sample.clipped ? "true" : "false")
                   << "}\n";
      sensor_trace.flush();
    }
    if (write_traces && t - last_log >= .01) {
      if (bounded) {
        ++bounded_samples;
        wind_trace << "{\"kind\":\"sample\",\"index\":" << bounded_samples
                   << ",\"time_us\":" << static_cast<int64_t>(t * 1e6)
                   << ",\"sample_time_s\":" << t
                   << (joint_phase_enabled ? ",\"weather_time_s\":" + std::to_string(weather_time) : "") << ",\"wind_enu_mps\":["
                   << wind.X() << ',' << wind.Y() << ',' << wind.Z()
                   << "],\"latent_state\":[" << bounded->state[0] << ',' << bounded->state[1] << ',' << bounded->state[2]
                   << "],\"latent_knot_index\":" << bounded->knot << "}\n";
        wind_trace.flush();
        if (!wind_trace)
          throw std::runtime_error("bounded wind trace write failed");
      }
      truth << "{\"time_us\":" << static_cast<long long>(t * 1e6)
            << ",\"voltage_v\":" << o.voltage << ",\"current_a\":" << o.current
            << ",\"model_valid\":" << (valid ? "true" : "false")
            << ",\"relative_airspeed_mps\":" << air.Length()
            << ",\"max_inplane_advance_ratio\":" << advance.maximum_ratio
            << ",\"inplane_advance_ratio_defined\":" << (advance.defined ? "true" : "false")
            << ",\"ground_velocity_enu_mps\":[" << vel->X() << ',' << vel->Y() << ',' << vel->Z() << ']'
            << ",\"air_velocity_flu_mps\":[" << air.X() << ',' << air.Y() << ',' << air.Z() << ']';
      if (bounded) {
        truth << ",\"wind_sample_index\":" << bounded_samples
              << ",\"wind_force_evaluations\":" << wind_force_evaluations
              << ",\"wind_observed_min_norm_mps\":" << bounded_min_norm
              << ",\"wind_observed_max_norm_mps\":" << bounded_max_norm
              << ",\"wind_max_physics_step_delta_mps\":" << bounded_max_step_delta;
      } else {
        truth
            << ",\"wind_update_index\":" << wind_update_index
            << ",\"wind_update_time_us\":" << static_cast<int64_t>(last_wind * 1e6)
            << ",\"gust_enu_mps\":[" << gust.X() << ',' << gust.Y() << ',' << gust.Z() << ']';
      }
      truth
            << ",\"sensor_dropped_total\":" << sensor.dropped
            << ",\"command_sequence\":" << command_gate.sequence
            << ",\"command_sample_us\":" << command_gate.sample_us
            << ",\"force_state_time_us\":"
            << static_cast<int64_t>((t - dt) * 1e6)
            << ",\"command_rejected_total\":" << rejected_commands
            << ",\"body_rate_flu\":[" << body_rate.X() << ',' << body_rate.Y()
            << ',' << body_rate.Z() << ']' << ",\"force_flu_n\":[" << force.X()
            << ',' << force.Y() << ',' << force.Z() << ']'
            << ",\"moment_flu_nm\":[" << moment.X() << ',' << moment.Y() << ','
            << moment.Z() << ']' << ",\"soc\":" << core.state.soc
            << ",\"temperature_k\":" << core.state.temperature
            << ",\"r0_ohm\":" << core.p.r0 << ",\"r1_ohm\":" << core.p.r1
            << ",\"c1_f\":" << core.p.c1
            << ",\"battery_stage_fraction\":" << battery_transition.fraction
            << ",\"battery_stage_onset_us\":"
            << static_cast<long long>(std::llround(scheduled_fault_at * 1e6))
            << ",\"battery_parameter_charge_c\":" << core.state.battery_parameter_charge_c
            << ",\"battery_parameter_energy_j\":" << core.state.battery_parameter_energy_j
            << ",\"battery_rc_parameter_work_j\":" << core.state.battery_rc_parameter_work_j
            << ",\"motor_resistance_ohm\":" << core.p.resistance
            << ",\"motor_coulomb_nm\":" << core.p.coulomb
            << ",\"esc_efficiency\":" << core.p.esc_efficiency
            << ",\"propulsion_profile_fraction\":" << propulsion_transition.fraction
            << ",\"propulsion_profile_onset_us\":"
            << (fault == "propulsion_profile" ? static_cast<long long>(std::llround(scheduled_fault_at * 1e6)) : -1)
            << ",\"kt_prop\":" << core.p.kt_prop
            << ",\"kq_prop\":" << core.p.kq_prop
            << ",\"friction_nms\":" << core.p.friction
            << ",\"capacity_ah\":" << core.p.capacity_ah
            << ",\"fault_applied\":" << (applied ? "true" : "false")
            << ",\"fault_applied_time_us\":" << applied_at
            << ",\"fault_boundary_time_us\":" << boundary_at
            << ",\"energy_residual_j\":" << core.energy_residual()
            << ",\"battery_energy_residual_j\":"
            << core.battery_energy_residual()
            << ",\"bus_energy_j\":" << core.state.bus_energy
            << ",\"solver_residual_v\":" << o.residual
            << ",\"altitude_m\":" << pose->Pos().Z() << ",\"battery_ocv_cell_v\":[";
      for (std::size_t i = 0; i < core.p.ocv_cell.size(); ++i)
        truth << (i ? "," : "") << core.p.ocv_cell[i];
      truth << "],\"command\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << command[i];
      truth << "],\"omega_rad_s\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << core.state.omega[i];
      truth << "],\"thrust_n\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << o.thrust[i];
      truth << "],\"shaft_torque_nm\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << o.torque[i];
      truth << "],\"motor_current_a\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << o.motor_current[i];
      truth << "],\"motor_temperature_k\":[";
      for (std::size_t i = 0; i < rotor_count; ++i)
        truth << (i ? "," : "") << core.state.motor_temperature[i];
      truth << "],\"wind_enu_mps\":[" << wind.X() + gust.X() << ','
            << wind.Y() + gust.Y() << ',' << wind.Z() + gust.Z() << "]}\n";
      truth.flush();
      if (!truth)
        throw std::runtime_error("truth write failed");
      last_log = t;
    }
  }
};
} // namespace x500
GZ_ADD_PLUGIN(x500::PowertrainSystem, gz::sim::System,
              x500::PowertrainSystem::ISystemConfigure,
              x500::PowertrainSystem::ISystemPreUpdate)

// Generic public loader name; the original C++ registration remains loadable.
GZ_ADD_PLUGIN_ALIAS(x500::PowertrainSystem, "drone::PowertrainSystem")
