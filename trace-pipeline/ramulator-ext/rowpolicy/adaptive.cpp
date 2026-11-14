#include <algorithm>

#include "amcas_common.h"

namespace Ramulator {

// Adaptive row policy (grid-owner): a small saturating counter per bank tracks recent row hits versus
// conflicts and switches the bank between "keep the row open" and "close after every access".
//   counter in [0, max]; keep open while counter >= thresh, else auto-precharge after the column access.
//   events:  column access to an already-used open row (2nd+ access of an activation)  -> +1  (hit)
//            row closed by the controller for a different row (conflict, PRE)            -> -1
//            ACT after WE closed the bank: same row as the one we closed (premature)    -> +1
//                                          different row (the close was right)          -> -1
// Parameters: max (7), thresh (4), init (4). No address hashing, no queue lookahead; 3 bits per bank.
class AdaptiveRowPolicy : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, AdaptiveRowPolicy, "Adaptive")

  struct Bank {
    int row = -1, acc = 0, ctr = 4, last_row = -1;
    bool closed_by_us = false;
  };
  AmcasRP m_;
  int m_max = 7, m_thresh = 4, m_init = 4;
  std::vector<Bank> m_b;

  void init() override {
    m_.bind(cast_parent<ControllerBase>());
    RAMULATOR_PARSE_PARAM(m_max, int, "max").default_val(7);
    RAMULATOR_PARSE_PARAM(m_thresh, int, "thresh").default_val(4);
    RAMULATOR_PARSE_PARAM(m_init, int, "init").default_val(4);
    m_b.assign(m_.nbanks, Bank());
    for (auto& s : m_b) s.ctr = m_init;
  }
  void up(Bank& s) { if (s.ctr < m_max) s.ctr++; }
  void down(Bank& s) { if (s.ctr > 0) s.ctr--; }

  void try_upgrade_command(Request& req) override {
    if (!m_.is_col(req)) return;
    int b = m_.bank_of(req);
    if (b < 0 || m_b[b].row != m_.row_of(req)) return;
    if (m_b[b].ctr < m_thresh) m_.auto_precharge(req);
  }

  void on_issue(const Request& req) override {
    const auto& meta = m_.spec->command_meta[req.command];
    if (meta.is_opening && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b < 0) return;
      Bank& s = m_b[b];
      int row = m_.row_of(req);
      if (s.closed_by_us) {
        if (row == s.last_row) up(s); else down(s);
        s.closed_by_us = false;
      }
      s.row = row; s.acc = 0;
      return;
    }
    if (meta.is_accessing && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b < 0) return;
      Bank& s = m_b[b];
      if (s.acc >= 1) up(s);  // row already served an access in this activation: a hit
      s.acc++;
      if (meta.is_closing) { s.closed_by_us = true; s.last_row = s.row; s.row = -1; }
      return;
    }
    if (meta.is_closing) {  // PRE / PREab by the controller (conflict or refresh)
      m_.dev->for_each_target_bank(req.command, req.addr_vec, [&](int id) {
        Bank& s = m_b[id];
        if (s.row >= 0) down(s);
        s.row = -1;
      });
    }
  }
};

}  // namespace Ramulator
