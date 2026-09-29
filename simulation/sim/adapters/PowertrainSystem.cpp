#include "aerodynamics.hpp"
#include "command.hpp"
#include "control.hpp"
#include "powertrain.hpp"
#include "sensor.hpp"
#include "wind.hpp"
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
  bool command_was_fresh = false;
  bool envelope_ok = true;
  unsigned rejected_commands = 0;
  int bench_axis = -1;
  std::mutex mutex;
  std::ofstream truth, control_events, command_trace, sensor_trace;
  Changes changes;
  gz::math::Vector3d wind{}, gust{};
  Wind turbulence{17, 0};
  double time = 0, last_log = -1, last_wind = -1;
  double ambient = 298.15, fault_at = -1, fault_value = 1, base_r0 = .04;
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
    if (battery_file) {
      std::ifstream f(battery_file);
      f >> core.p.r0 >> core.p.r1 >> core.p.c1 >> core.p.capacity_ah;
      for (auto &v : core.p.ocv_cell)
        f >> v;
      if (!f)
        throw std::runtime_error("invalid calibrated battery file");
      core.p.calibrated = true;
    }
    const char *prop_file = std::getenv("X500_PROPULSION_FILE");
    if (prop_file) {
      std::ifstream f(prop_file);
      f >> core.p.kt_prop >> core.p.kq_prop >> core.p.friction >>
          core.p.coulomb >> core.p.esc_efficiency >> core.p.resistance;
      if (!f)
        throw std::runtime_error("invalid propulsion profile");
    }
    core.state.soc = env("X500_SOC", .9);
    ambient = env("X500_TEMPERATURE", 298.15);
    core.state.temperature = ambient;
    core.state.motor_temperature.fill(ambient);
    core.p.r0 = env("X500_R0", core.p.r0);
    core.p.r1 *= env("X500_R1_MULTIPLIER", 1);
    core.p.resistance *= env("X500_MOTOR_R_MULTIPLIER", 1);
    core.p.kt_prop *= env("X500_KT_MULTIPLIER", 1);
    core.p.kq_prop *= env("X500_KQ_MULTIPLIER", 1);
    base_r0 = core.p.r0;
    core.p.capacity_ah = env("X500_CAPACITY", core.p.capacity_ah);
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
    fault_at = env("X500_FAULT_AT", -1);
    fault_value = env("X500_FAULT_VALUE", 1.4);
    const char *f = std::getenv("X500_FAULT");
    if (f)
      fault = f;
    const char *path = std::getenv("X500_TRUTH_PATH");
    if (!path)
      throw std::runtime_error("X500_TRUTH_PATH required");
    truth.open(path);
    control_events.open(std::string(path) + ".control-events.jsonl");
    command_trace.open(std::string(path) + ".command-trace.jsonl");
    sensor_trace.open(std::string(path) + ".sensor-trace.jsonl");
    sensor_trace << std::setprecision(12);
    truth << std::setprecision(12);
    if (!truth)
      throw std::runtime_error("cannot open truth");
    node.Subscribe("/" + name + "/command/motor_speed",
                   &PowertrainSystem::command, this);
    node.Advertise("/" + name + "/apply_parameter_change",
                   &PowertrainSystem::change, this);
    node.Advertise("/" + name + "/change_status",
                   &PowertrainSystem::change_status, this);
    battery =
        node.Advertise<gz::msgs::Double_V>("/" + name + "/battery_sensor");
  }
  bool change(const gz::msgs::Double_V &req, gz::msgs::Double_V &reply) {
    std::lock_guard<std::mutex> lock(mutex);
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
      command_trace << "{\"kind\":\"command_rejected\",\"reason\":\"" << result
                    << "\",\"sim_us\":" << static_cast<int64_t>(time * 1e6)
                    << "}\n";
      command_trace.flush();
    }
  }
  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override {
    if (info.paused || info.dt.count() <= 0)
      return;
    double t = std::chrono::duration<double>(info.simTime).count();
    double dt = std::chrono::duration<double>(info.dt).count();
    Four command;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (t < time)
        throw std::runtime_error("clock reset requires a new run identity");
      time = t;
      for (const auto &r :
           changes.apply(core, static_cast<long long>(t * 1e6))) {
        control_events << "{\"request_id\":" << r.change.id
                       << ",\"field\":" << r.change.field
                       << ",\"value\":" << r.change.value
                       << ",\"requested_time_us\":" << r.change.requested_us
                       << ",\"applied_time_us\":" << r.applied_us
                       << ",\"revision\":" << r.revision << "}\n";
        control_events.flush();
      }
      bool fresh = command_gate.fresh(static_cast<int64_t>(t * 1e6));
      command = command_gate.duty;
      if (!fresh)
        command.fill(0);
      if (fresh != command_was_fresh) {
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
    if (fault_at >= 0 && t >= fault_at && (!applied || ramp_s > 0)) {
      double fraction =
          ramp_s > 0 ? std::clamp((t - fault_at) / ramp_s, 0., 1.) : 1.;
      if (fault == "r0") {
#ifdef DRONE_PROFILE_HEADER
        core.p.r0 = profile::interconnect_resistance +
                    (base_r0 - profile::interconnect_resistance) *
                        (1 + fraction * (fault_value - 1));
#else
        core.p.r0 = base_r0 * (1 + fraction * (fault_value - 1));
#endif
      } else if (fault == "friction")
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
    if (t - last_wind >= .01) {
      double h = last_wind < 0 ? .01 : t - last_wind;
      turbulence.step(h);
      for (int k = 0; k < 3; ++k)
        gust[k] = turbulence.gust[k];
      last_wind = t;
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
    if (t - last_log >= .01) {
      truth << "{\"time_us\":" << static_cast<long long>(t * 1e6)
            << ",\"voltage_v\":" << o.voltage << ",\"current_a\":" << o.current
            << ",\"model_valid\":" << (valid ? "true" : "false")
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
            << ",\"motor_resistance_ohm\":" << core.p.resistance
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
            << ",\"altitude_m\":" << pose->Pos().Z() << ",\"command\":[";
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
            << wind.Y() + gust.Y() << ',' << gust.Z() << "]}\n";
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
