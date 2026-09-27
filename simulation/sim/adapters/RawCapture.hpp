#pragma once
#include "CaptureTiming.hpp"
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <uORB/Subscription.hpp>
#include <uORB/topics/actuator_outputs.h>
#include <uORB/topics/battery_status.h>
#include <uORB/topics/trajectory_setpoint.h>
#include <uORB/topics/vehicle_acceleration.h>
#include <uORB/topics/vehicle_angular_velocity.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_land_detected.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_local_position_setpoint.h>
#include <uORB/topics/vehicle_rates_setpoint.h>
#include <uORB/topics/vehicle_status.h>
class RawCapture {
  CaptureTiming timing;
  FILE *f = nullptr, *diag = nullptr;
  uint64_t last_diag = 0;
  int rotor_count = 4;
  std::map<std::string, unsigned> generations, skipped;
  uORB::Subscription rate{ORB_ID(vehicle_angular_velocity)},
      target{ORB_ID(vehicle_rates_setpoint)},
      attitude{ORB_ID(vehicle_attitude)}, accel{ORB_ID(vehicle_acceleration)},
      pos{ORB_ID(vehicle_local_position)},
      position_target{ORB_ID(vehicle_local_position_setpoint)},
      trajectory{ORB_ID(trajectory_setpoint)}, battery{ORB_ID(battery_status)},
      status{ORB_ID(vehicle_status)}, land{ORB_ID(vehicle_land_detected)},
      outputs{ORB_ID(actuator_outputs)};
  void begin(const char *topic, uint64_t t, uint64_t sample,
             unsigned generation) {
    auto previous = generations.find(topic);
    if (previous != generations.end() && generation > previous->second + 1)
      skipped[topic] += generation - previous->second - 1;
    generations[topic] = generation;
    t = hrt_absolute_time();
    timing.begin_event(t);
    fprintf(f,
            "{\"topic\":\"%s\",\"sample_time_us\":%llu,\"available_time_us\":%"
            "llu,\"fields\":{",
            topic, (unsigned long long)sample, (unsigned long long)t);
  }
  void number(const char *key, double value, bool comma = true) {
    fprintf(f, "\"%s\":", key);
    if (std::isfinite(value))
      fprintf(f, "%.9g", value);
    else
      fputs("null", f);
    if (comma)
      fputc(',', f);
  }
  void array(const char *key, const float *values, int n) {
    fprintf(f, "\"%s\":[", key);
    for (int i = 0; i < n; ++i) {
      if (i)
        fputc(',', f);
      if (std::isfinite(values[i]))
        fprintf(f, "%.9g", (double)values[i]);
      else
        fputs("null", f);
    }
    fputc(']', f);
  }
  void end() {
    fputs("}}\n", f);
    timing.end_event();
  }

public:
  ~RawCapture() {
    if (f)
      fclose(f);
    if (diag)
      fclose(diag);
  }
  void update(uint64_t now) {
    if (!f) {
      const char *path = getenv("X500_CAPTURE_PATH");
      if (!path)
        return;
      if (const char *count = getenv("CAPSTONE_ROTOR_COUNT")) {
        char *end = nullptr;
        const long value = std::strtol(count, &end, 10);
        if (end == count || *end != '\0' || value < 4 || value > 8) {
          PX4_ERR("invalid profile rotor count");
          std::_Exit(74);
        }
        rotor_count = static_cast<int>(value);
      }
      std::signal(SIGXFSZ, SIG_IGN);
      f = fopen(path, "w");
      diag = fopen((std::string(path) + ".diagnostics.jsonl").c_str(), "w");
      if (!f || !diag) {
        PX4_ERR("raw capture open failure");
        std::_Exit(74);
      }
      timing.open(getenv("X500_CAPTURE_TIMING_PATH"), getenv("X500_RUN_ID"));
    }
    timing.start(f, hrt_absolute_time);
    vehicle_angular_velocity_s r{};
    if (rate.update(&r)) {
      begin("vehicle_angular_velocity", now, r.timestamp_sample,
            rate.get_last_generation());
      array("xyz", r.xyz, 3);
      end();
    }
    vehicle_rates_setpoint_s rs{};
    if (target.update(&rs)) {
      begin("vehicle_rates_setpoint", now, rs.timestamp,
            target.get_last_generation());
      number("roll", rs.roll);
      number("pitch", rs.pitch);
      number("yaw", rs.yaw, false);
      end();
    }
    vehicle_attitude_s a{};
    if (attitude.update(&a)) {
      begin("vehicle_attitude", now, a.timestamp_sample,
            attitude.get_last_generation());
      array("q", a.q, 4);
      end();
    }
    vehicle_acceleration_s ac{};
    if (accel.update(&ac)) {
      begin("vehicle_acceleration", now, ac.timestamp_sample,
            accel.get_last_generation());
      array("xyz", ac.xyz, 3);
      end();
    }
    vehicle_local_position_s p{};
    if (pos.update(&p)) {
      begin("vehicle_local_position", now, p.timestamp_sample,
            pos.get_last_generation());
      number("x", p.x);
      number("y", p.y);
      number("z", p.z);
      number("vz", p.vz);
      number("z_valid", p.z_valid, false);
      end();
    }
    vehicle_local_position_setpoint_s ps{};
    if (position_target.update(&ps)) {
      begin("vehicle_local_position_setpoint", now, ps.timestamp,
            position_target.get_last_generation());
      number("vx", ps.vx);
      number("vy", ps.vy);
      number("vz", ps.vz, false);
      end();
    }
    trajectory_setpoint_s ts{};
    if (trajectory.update(&ts)) {
      begin("trajectory_setpoint", now, ts.timestamp,
            trajectory.get_last_generation());
      array("velocity", ts.velocity, 3);
      end();
    }
    battery_status_s b{};
    if (battery.update(&b)) {
      begin("battery_status", now, b.timestamp, battery.get_last_generation());
      number("voltage_v", b.voltage_v);
      number("current_a", b.current_a, false);
      end();
    }
    vehicle_status_s st{};
    if (status.update(&st)) {
      begin("vehicle_status", now, st.timestamp, status.get_last_generation());
      number("arming_state", st.arming_state);
      number("nav_state", st.nav_state);
      number("hil_state", st.hil_state, false);
      end();
    }
    vehicle_land_detected_s l{};
    if (land.update(&l)) {
      begin("vehicle_land_detected", now, l.timestamp,
            land.get_last_generation());
      number("landed", l.landed, false);
      end();
    }
    actuator_outputs_s out{};
    if (outputs.update(&out)) {
      begin("actuator_outputs", now, out.timestamp,
            outputs.get_last_generation());
      if (getenv("CAPSTONE_ROTOR_COUNT")) {
        // uORB reports driver capacity (16), unlike the active N-value Gazebo
        // request. The first N channels are the explicitly mapped motor slots.
        if (out.noutputs < static_cast<unsigned>(rotor_count) ||
            out.noutputs > sizeof(out.output) / sizeof(out.output[0])) {
          PX4_ERR("profile actuator output capacity mismatch");
          std::_Exit(74);
        }
        for (int i = 0; i < rotor_count; ++i) {
          if (!std::isfinite(out.output[i]) || out.output[i] < 0 ||
              out.output[i] > 1000) {
            PX4_ERR("invalid profile actuator output value");
            std::_Exit(74);
          }
        }
      }
      array("output", out.output, rotor_count);
      end();
    }
    timing.capture_done();
    if (diag && now - last_diag >= 1000000) {
      for (const auto &item : generations)
        fprintf(diag,
                "{\"kind\":\"uorb_latest_sample_stats\",\"topic\":\"%s\","
                "\"through_time_us\":%llu,\"source_generation\":%u,\"skipped_"
                "total\":%u,\"policy\":\"declared_latest_sample_capture\"}\n",
                item.first.c_str(), (unsigned long long)now, item.second,
                skipped[item.first]);
      if (fflush(diag) != 0 || ferror(diag)) {
        PX4_ERR("raw diagnostics write failure");
        std::_Exit(74);
      }
      last_diag = now;
    }
    timing.flush_begin();
    if (fflush(f) != 0 || ferror(f)) {
      PX4_ERR("raw capture write failure");
      std::_Exit(74);
    }
    timing.finish(f, hrt_absolute_time);
  }
};
