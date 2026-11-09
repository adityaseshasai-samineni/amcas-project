#include "amcas_common.h"

namespace Ramulator {

// Timeout row policy: keep a row open, precharge it after `timeout` controller cycles without a
// column access to that bank (the common adaptive baseline of simple controllers).
// Caveat: "idle" is measured from the last column command, including cycles where a queued
// request to the row waits for timing; this matches the usual hardware counter.
class TimeoutRowPolicy : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, TimeoutRowPolicy, "Timeout")

  AmcasRP m_;
  int m_timeout = 100;
  std::vector<char> m_open, m_inj;
  std::vector<Clk_t> m_last;

  void init() override {
    m_.bind(cast_parent<ControllerBase>());
    RAMULATOR_PARSE_PARAM(m_timeout, int, "timeout").default_val(100);
    m_open.assign(m_.nbanks, 0);
    m_inj.assign(m_.nbanks, 0);
    m_last.assign(m_.nbanks, 0);
  }

  void on_issue(const Request& req) override {
    const auto& meta = m_.spec->command_meta[req.command];
    if (meta.is_closing) {
      m_.dev->for_each_target_bank(req.command, req.addr_vec, [&](int id) { m_open[id] = 0; m_inj[id] = 0; });
    }
    if (meta.is_opening && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b >= 0) { m_open[b] = 1; m_last[b] = m_.ctrl->m_clk; }
    }
    if (meta.is_accessing && !meta.is_closing) {
      int b = m_.bank_of(req);
      if (b >= 0) { m_open[b] = 1; m_last[b] = m_.ctrl->m_clk; }
    }
  }

  void pre_schedule() override {
    Clk_t now = m_.ctrl->m_clk;
    for (int i = 0; i < m_.nbanks; i++) {
      if (!m_open[i] || m_inj[i] || now - m_last[i] < (Clk_t)m_timeout) continue;
      if (m_.inject_prepb(i)) m_inj[i] = 1;
    }
  }
};

}  // namespace Ramulator
