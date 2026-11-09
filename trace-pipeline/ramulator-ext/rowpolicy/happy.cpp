#include <cstdint>

#include "amcas_common.h"

namespace Ramulator {

// HAPPY-style adaptive page closure (after Ghasempour et al., "HAPPY: Hybrid Address-based Page
// Policy in DRAMs", MEMSYS 2016). Predicts, at every column access, whether the open row will
// be hit again; if not it auto-precharges (RDA/WRA), saving the tRP of the later conflict.
//
// What is implemented (a simplification, from our reading of the paper's idea; the paper was
// not re-read in detail and its exact structures may differ):
//  * Predictor indexed by an ADDRESS HASH of (bank, row): the address is folded to
//    log2(entries) bits, so hardware cost is independent of memory size (the point of HAPPY).
//    Aliasing is allowed.
//  * mode=binary (default): 2-bit saturating counter per entry, "this page gets reused".
//    Close after an access when counter < 2. Trained when a row is closed: a row closed by the
//    controller (conflict) after >=2 column accesses -> increment, after 1 -> decrement; a row
//    that WE closed is judged at the next ACT of the bank: same row again -> premature close ->
//    increment; different row -> correct close -> decrement.
//  * mode=count: the entry stores the number of column accesses the page received in its last
//    activation; close once that many have been served (0/untrained -> stay open).
//    A premature close doubles the stored count.
//  NOT implemented: the paper's per-thread/global history combination and its tournament between
//  predictors, write/read distinction, and any queue lookahead (the controller's own check that
//  an upgrade must not strand another active request still applies).
class HappyRowPolicy : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, HappyRowPolicy, "HAPPY")

  struct Bank {
    int row = -1, idx = 0, acc = 0;
    bool closed_by_us = false;
    int last_row = -1, last_idx = 0, last_acc = 0;
  };
  AmcasRP m_;
  int m_entries = 1024;
  std::string m_mode;
  bool m_count = false;
  std::vector<uint8_t> m_t;
  std::vector<Bank> m_b;

  void init() override {
    m_.bind(cast_parent<ControllerBase>());
    RAMULATOR_PARSE_PARAM(m_entries, int, "entries").default_val(1024);
    RAMULATOR_PARSE_PARAM(m_mode, std::string, "mode").default_val("binary");
    m_count = (m_mode == "count");
    m_t.assign(m_entries, m_count ? 0 : 2);
    m_b.assign(m_.nbanks, Bank());
  }

  int hash(int bank, int row) const {
    uint64_t x = ((uint64_t)bank << 20) ^ (uint64_t)row;
    x ^= x >> 11; x *= 0x9E3779B97F4A7C15ull; x ^= x >> 29;
    return (int)(x % (uint64_t)m_entries);
  }
  void inc(int i) { if (m_t[i] < 3) m_t[i]++; }
  void dec(int i) { if (m_t[i] > 0) m_t[i]--; }

  bool predict_close(const Bank& s) const {
    if (m_count) return m_t[s.idx] > 0 && s.acc + 1 >= m_t[s.idx];  // this access is the predicted last
    return m_t[s.idx] < 2;
  }

  void try_upgrade_command(Request& req) override {
    if (!m_.is_col(req)) return;
    int b = m_.bank_of(req);
    if (b < 0 || m_b[b].row != m_.row_of(req)) return;
    if (predict_close(m_b[b])) m_.auto_precharge(req);
  }

  void on_issue(const Request& req) override {
    const auto& meta = m_.spec->command_meta[req.command];
    if (meta.is_opening && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b < 0) return;
      Bank& s = m_b[b];
      int row = m_.row_of(req);
      if (s.closed_by_us) {  // judge our earlier close
        if (row == s.last_row) {
          if (m_count) m_t[s.last_idx] = (uint8_t)std::min(255, std::max(2, 2 * (int)m_t[s.last_idx]));
          else inc(s.last_idx);
        } else if (!m_count) dec(s.last_idx);
        s.closed_by_us = false;
      }
      s.row = row; s.idx = hash(b, row); s.acc = 0;
      return;
    }
    if (meta.is_accessing && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b < 0) return;
      Bank& s = m_b[b];
      s.acc++;
      if (meta.is_closing) {  // RDA/WRA: closed by our prediction (or by an upgrade of a plain RD)
        s.closed_by_us = true; s.last_row = s.row; s.last_idx = s.idx; s.last_acc = s.acc;
        s.row = -1;
      }
      return;
    }
    if (meta.is_closing) {  // PRE / PREab: closed by the controller (conflict or refresh)
      m_.dev->for_each_target_bank(req.command, req.addr_vec, [&](int id) {
        Bank& s = m_b[id];
        if (s.row >= 0) {
          if (m_count) m_t[s.idx] = (uint8_t)std::min(255, std::max(1, s.acc));
          else if (s.acc >= 2) inc(s.idx);
          else dec(s.idx);
        }
        s.row = -1;
      });
    }
  }
};

}  // namespace Ramulator
