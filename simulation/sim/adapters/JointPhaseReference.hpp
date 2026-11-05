#pragma once
// Linux SITL only. Both the clock writer and actual reference consumer use this mutex.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace joint_phase {
inline std::mutex &clock_mutex() { static std::mutex value; return value; }
inline bool enabled() { return std::getenv("X500_JOINT_REFERENCE") != nullptr; }
class Reference {
  struct Row { std::array<float, 3> p, v, a; float yaw; };
  std::vector<Row> rows;
  FILE *trace = nullptr;
  bool loaded = false, ready = false, pending = false, landed = false, requested_land = false;
  int64_t ready_since = -1, origin = -1, duration_us = 480000000;
  size_t selected = 0;
  uint64_t sequence = 0;
  [[noreturn]] static void fail(const char *message) {
    std::fprintf(stderr, "joint_phase_failure: %s\n", message);
    std::fflush(stderr); std::_Exit(74);
  }
  void load() {
    if (loaded || !enabled()) return;
    FILE *input = std::fopen(std::getenv("X500_JOINT_REFERENCE"), "r");
    if (!input) fail("reference open");
    if (const char *duration = std::getenv("X500_JOINT_DURATION_US")) {
      const std::string value(duration);
      if (value != "180000000" && value != "480000000") fail("unsupported mission duration");
      duration_us = std::stoll(value);
    }
    for (int64_t i = 0; i <= duration_us / 100000; ++i) {
      long long phase; Row row{};
      if (std::fscanf(input, "%lld,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", &phase,
          &row.p[0], &row.p[1], &row.p[2], &row.v[0], &row.v[1], &row.v[2],
          &row.a[0], &row.a[1], &row.a[2], &row.yaw) != 11 || phase != i * 100000LL)
        fail("reference shape/cadence");
      for (const auto &vector : {row.p, row.v, row.a})
        for (float value : vector) if (!std::isfinite(value)) fail("reference finite");
      if (!std::isfinite(row.yaw)) fail("reference yaw");
      rows.push_back(row);
    }
    if (std::fgetc(input) != EOF) fail("reference trailing rows");
    std::fclose(input);
    const char *path = std::getenv("X500_JOINT_RECEIPTS");
    if (!path || !(trace = std::fopen(path, "w"))) fail("consumption trace open");
    std::fprintf(trace, "sequence,source_us,phase_us,index,px,py,pz,vx,vy,vz,ax,ay,az,yaw\n");
    loaded = true;
  }
  void receipt(const char *environment, int64_t now, const char *kind) {
    const char *path = std::getenv(environment);
    const char *hash = std::getenv("X500_JOINT_POLICY_SHA256");
    if (!path || !hash || std::string(hash).size() != 64) fail("receipt path/policy");
    std::string temporary = std::string(path) + ".tmp";
    FILE *file = std::fopen(temporary.c_str(), "w");
    if (!file) fail("receipt open");
    std::fprintf(file, "{\"schema\":\"drone.joint-phase-native-receipt/1\",\"revision\":\"joint-native-consumption-c2-v2\",\"kind\":\"%s\",\"policy_sha256\":\"%s\",\"requested_start_us\":60000000,\"actual_start_us\":%lld,\"actual_time_us\":%lld,\"duration_us\":%lld,\"clock\":\"px4-boot\",\"consumer\":\"PositionControl::setInputSetpoint\"}\n", kind, hash, static_cast<long long>(origin), static_cast<long long>(now), static_cast<long long>(duration_us));
    if (std::fflush(file) || std::ferror(file)) fail("receipt write");
    if (std::fclose(file) || std::rename(temporary.c_str(), path)) fail("receipt commit");
  }
public:
  ~Reference() { if (trace) std::fclose(trace); }
  bool active() const { return enabled(); }
  void fallback() const { if (enabled() && origin >= 0 && !landed) fail("controller fallback during joint mission"); }
  template<class Setpoint>
  void prepare(int64_t now, bool eligible, double x, double y, double z, Setpoint &point) {
    if (!enabled()) return;
    load();
    const bool hover = eligible && std::sqrt(x*x + y*y + (z+3)*(z+3)) <= .5;
    if (origin < 0) {
      if (!hover) ready_since = -1;
      else if (ready_since < 0) ready_since = now;
      if (hover && now < 59000000 && now - ready_since >= 1000000) ready = true;
      if (now >= 59000000 && !ready) fail("late native hover readiness");
      if (now < 60000000) return;
      if (!hover) fail("not ready at actual start");
      pending = true;
    } else if (!eligible) return;
    int64_t phase = origin < 0 ? 0 : now - origin;
    if (phase < 0) fail("reference clock reversal");
    selected = static_cast<size_t>(phase / 100000);
    if (selected >= rows.size()) selected = rows.size()-1;
    const auto &row = rows[selected];
    point.timestamp = static_cast<uint64_t>(now);
    for (int axis = 0; axis < 3; ++axis) {
      point.position[axis] = row.p[axis]; point.velocity[axis] = row.v[axis]; point.acceleration[axis] = row.a[axis];
    }
    point.yaw = row.yaw; point.yawspeed = NAN;
  }
  template<class Setpoint>
  void consumed(int64_t now, bool offboard, bool auto_land, const Setpoint &point) {
    if (!enabled()) return;
    if (pending) {
      origin = now; pending = false;
      receipt("X500_JOINT_ORIGIN", now, "actual_reference_consumption_start");
    }
    if (origin < 0) return;
    if (auto_land && !landed) {
      if (now < origin + duration_us || now > origin + duration_us + 100000) fail("actual landing outside declared end budget");
      receipt("X500_JOINT_LANDING", now, "actual_auto_land_reference_consumption");
      landed = true;
    }
    if (!offboard || landed) return;
    if (std::fprintf(trace, "%llu,%lld,%lld,%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        static_cast<unsigned long long>(++sequence), static_cast<long long>(now), static_cast<long long>(now-origin), selected,
        double(point.position[0]), double(point.position[1]), double(point.position[2]),
        double(point.velocity[0]), double(point.velocity[1]), double(point.velocity[2]),
        double(point.acceleration[0]), double(point.acceleration[1]), double(point.acceleration[2]), double(point.yaw)) < 0 || std::fflush(trace) || std::ferror(trace)) fail("consumption trace write");
  }
  bool land_due(int64_t now) const { return enabled() && origin >= 0 && now >= origin + duration_us && !requested_land; }
  void land_published(int64_t now) { requested_land = true; receipt("X500_JOINT_LAND_REQUEST", now, "native_land_publication_not_activation"); }
};
} // namespace joint_phase
