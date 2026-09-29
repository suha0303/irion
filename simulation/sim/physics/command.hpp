#pragma once
#include "powertrain.hpp"
#include <cstdint>
#include <string>
namespace x500 {
struct CommandGate {
  bool seen = false;
  uint64_t sequence = 0;
  int64_t sample_us = -1;
  RotorValues duty{};
  std::string accept(const std::string &schema, const std::string &order,
                     const std::string &semantic, uint64_t seq, int64_t sample,
                     int64_t now, const RotorValues &counts) {
#ifdef CAPSTONE_PROFILE_GENERIC
    if (schema != profile::esc_schema || order != profile::esc_order ||
#else
    if (schema != "x500.esc.request/1" || order != "FR,RL,FL,RR" ||
#endif
        semantic != "post_mapping_count_0_1000")
      return "protocol_mismatch";
    if (sample < 0 || sample > now + 20000 || now - sample > 500000)
      return "stale_or_future_sample";
    if (seen && (seq <= sequence || sample < sample_us))
      return "sequence_or_time_regression";
    for (auto v : counts)
      if (!std::isfinite(v) || v < 0 || v > 1000)
        return "invalid_count";
    sequence = seq;
    sample_us = sample;
    seen = true;
    for (std::size_t i = 0; i < rotor_count; ++i)
      duty[i] = counts[i] / 1000.;
    return "accepted";
  }
  bool fresh(int64_t now) const { return seen && now - sample_us <= 500000; }
};
} // namespace x500
