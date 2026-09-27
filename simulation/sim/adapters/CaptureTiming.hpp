#pragma once
// Linux SITL diagnostics only: syscall bypasses PX4's simulated clock wrappers.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

class CaptureTiming {
public:
  using SimClock = uint64_t (*)();
  static uint64_t host_ns() {
    struct timespec ts {};
    if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &ts) != 0)
      fail("capture timing host clock");
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
  }
  void open(const char *path, const char *id) {
    if (!path)
      return;
    if (!id || !*id)
      fail("capture timing missing run ID");
    run_id = id;
    if (run_id.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_") != std::string::npos)
      fail("capture timing invalid run ID");
    trace = std::fopen(path, "w");
    if (!trace)
      fail("capture timing open");
  }
  ~CaptureTiming() {
    if (trace)
      std::fclose(trace);
  }
  void start(FILE *native, SimClock clock) {
    if (!trace)
      return;
    begin_host = host_ns();
    begin_sim = clock();
    begin_clock_end_host = host_ns();
    byte_start = offset(native);
    first_event = event_count + 1;
    serialize_sum = 0;
    serialize_start = serialize_end = 0;
    first_available = last_available = 0;
  }
  void begin_event(uint64_t available_us) {
    if (!trace)
      return;
    event_start = host_ns();
    if (!serialize_start) {
      serialize_start = event_start;
      first_available = available_us;
    }
    last_available = available_us;
  }
  void end_event() {
    if (!trace)
      return;
    serialize_end = host_ns();
    serialize_sum += serialize_end - event_start;
    ++event_count;
  }
  void capture_done() {
    if (trace)
      capture_end = host_ns();
  }
  void flush_begin() {
    if (trace)
      flush_start = host_ns();
  }
  void finish(FILE *native, SimClock clock) {
    if (!trace)
      return;
    const auto flush_end = host_ns();
    const auto end_sim = clock();
    const auto end_clock_host = host_ns();
    const auto byte_end = offset(native);
    // Emission/flush of this sidecar occurs AFTER all measured native stages.
    // Gaps before the next batch include its overhead; no fsync is added here.
    std::ostringstream row;
    row << "{\"schema\":\"x500.capture-timing/1\",\"run_id\":\"" << run_id
        << "\",\"clock_id\":\"px4-boot\",\"host_clock\":\"CLOCK_MONOTONIC\","
           "\"host_clock_unit\":\"ns\",\"batch_index\":" << batch_index++
        << ",\"first_event_index\":" << first_event
        << ",\"last_event_index\":" << event_count
        << ",\"event_count\":" << event_count - first_event + 1
        << ",\"native_byte_start\":" << byte_start
        << ",\"native_byte_end\":" << byte_end
        << ",\"first_available_time_us\":" << first_available
        << ",\"last_available_time_us\":" << last_available
        << ",\"capture_start_host_monotonic_ns\":" << begin_host
        << ",\"capture_start_px4_hrt_us\":" << begin_sim
        << ",\"capture_start_clock_read_end_host_monotonic_ns\":" << begin_clock_end_host
        << ",\"serialization_start_host_monotonic_ns\":" << serialize_start
        << ",\"serialization_end_host_monotonic_ns\":" << serialize_end
        << ",\"serialization_write_sum_ns\":" << serialize_sum
        << ",\"capture_end_host_monotonic_ns\":" << capture_end
        << ",\"native_flush_start_host_monotonic_ns\":" << flush_start
        << ",\"native_flush_end_host_monotonic_ns\":" << flush_end
        << ",\"native_flush_end_px4_hrt_us\":" << end_sim
        << ",\"native_flush_end_clock_read_end_host_monotonic_ns\":" << end_clock_host
        << "}\n";
    const auto text = row.str();
    if (std::fwrite(text.data(), 1, text.size(), trace) != text.size() ||
        std::fflush(trace) != 0 || std::ferror(trace))
      fail("capture timing write");
  }

private:
  [[noreturn]] static void fail(const char *message) {
    std::perror(message);
    std::_Exit(74);
  }
  static uint64_t offset(FILE *file) {
    const auto pos = ftello(file);
    if (pos < 0)
      fail("capture timing native offset");
    return static_cast<uint64_t>(pos);
  }
  FILE *trace = nullptr;
  std::string run_id;
  uint64_t batch_index = 0, event_count = 0, first_event = 0;
  uint64_t begin_host = 0, begin_sim = 0, begin_clock_end_host = 0;
  uint64_t byte_start = 0, event_start = 0, serialize_sum = 0;
  uint64_t serialize_start = 0, serialize_end = 0;
  uint64_t capture_end = 0, flush_start = 0;
  uint64_t first_available = 0, last_available = 0;
};
