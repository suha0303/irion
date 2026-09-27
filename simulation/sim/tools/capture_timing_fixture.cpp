// Synthetic writer/clock fixture; does not execute PX4 or a flight.
#include "../adapters/CaptureTiming.hpp"

static uint64_t sim_time = 0;
static uint64_t fake_hrt() { return sim_time; }
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  CaptureTiming timing;
  timing.open(std::string(argv[2]) == "disabled" ? nullptr : argv[2], "fixture");
  auto *raw = std::fopen(argv[1], "w");
  if (!raw)
    return 3;
  for (const auto t : {1000ULL, 1000ULL, 900000000000ULL}) {
    sim_time = t;
    timing.start(raw, fake_hrt);
    timing.begin_event(t);
    std::fprintf(raw, "{\"available_time_us\":%llu}\n", t);
    timing.end_event();
    timing.capture_done();
    timing.flush_begin();
    if (std::fflush(raw))
      return 4;
    timing.finish(raw, fake_hrt);
  }
  return std::fclose(raw) ? 4 : 0;
}
