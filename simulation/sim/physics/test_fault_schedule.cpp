#include "fault_schedule.hpp"
#include <cassert>
#include <limits>

template <class F> void rejects(F action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  using x500::MissionFaultSchedule;
  MissionFaultSchedule disabled;
  rejects([&] { disabled.schedule(17000000, 18000000); });
  assert(disabled.onset_us == -1);
  for (const auto kind : {"r0", "friction"}) {
    MissionFaultSchedule schedule;
    schedule.configure(100, -1, kind);
    assert(schedule.onset_us == -1);
    rejects([&] { schedule.schedule(19000000, 18000000); });
    rejects([&] { schedule.schedule(17000000, 117000000); });
    for (double bad : {-1.0, 1.5, 1e15 + 1,
                       std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()})
      rejects([&] { schedule.schedule(bad, 18000000); });
    assert(schedule.origin_us == -1);
    assert(schedule.schedule(17000000, 18000000) == 117000000);
    assert(schedule.schedule(17000000, 118000000) == 117000000);
    rejects([&] { schedule.schedule(17000001, 18000000); });
    assert(schedule.onset_us == 117000000);
  }
  for (double bad : {-1.0, 0.0, 1801.0, 0.0000001,
                     std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity()}) {
    MissionFaultSchedule schedule;
    rejects([&] { schedule.configure(bad, -1, "r0"); });
  }
  MissionFaultSchedule mixed, unsupported;
  rejects([&] { mixed.configure(100, 180, "r0"); });
  rejects([&] { unsupported.configure(100, -1, "battery_soc"); });
}
