#pragma once
#include "powertrain.hpp"
#include <deque>
#include <map>
namespace x500 {
struct Change {
  unsigned id;
  int field;
  double value;
  long long requested_us;
};
struct Receipt {
  Change change;
  long long applied_us = -1;
  unsigned revision = 0;
};
class Changes {
  std::deque<Change> pending;
  std::map<unsigned, Receipt> receipts;
  unsigned revision = 0;

public:
  bool enqueue(const Change &c) {
    if (c.field != 1)
      throw std::invalid_argument("scalar battery changes retired; select a measured snapshot");
    if (c.id == 0 || !std::isfinite(c.value) ||
        c.value < 0 || c.value > .001 || c.requested_us < 0)
      throw std::invalid_argument("invalid physical change");
    auto i = receipts.find(c.id);
    if (i != receipts.end()) {
      const auto &old = i->second.change;
      if (old.field != c.field || old.value != c.value ||
          old.requested_us != c.requested_us)
        throw std::invalid_argument("request ID reused with different payload");
      return false;
    }
    if (pending.size() >= 32)
      throw std::runtime_error("control queue full");
    pending.push_back(c);
    receipts.emplace(c.id, Receipt{c});
    return true;
  }
  std::deque<Receipt> apply(Powertrain &core, long long now) {
    std::deque<Receipt> applied;
    while (!pending.empty()) {
      auto c = pending.front();
      if (c.requested_us > now)
        break;
      core.p.friction = c.value;
      auto &r = receipts.at(c.id);
      r.applied_us = now;
      r.revision = ++revision;
      applied.push_back(r);
      pending.pop_front();
    }
    return applied;
  }
  const Receipt &status(unsigned id) const { return receipts.at(id); }
};
} // namespace x500
