#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

#include "amcas_common.h"

namespace Ramulator {

// Oracle row policy (grid-owner): a future-aware bound on any row-buffer policy, two passes.
//   mode=record : behaves exactly as Open page; logs (request address, flat bank, row) of every column command
//                 to <path>.<channel> (it only serves to learn address -> (bank, row) for this channel).
//   mode=oracle : reads that log and the original trace (`trace`, lines "LD 0xADDR", the ARRIVAL order of requests
//                 in a saturated replay) and builds, per bank of this channel, the sequence of rows in arrival
//                 order. At a column command to row r it finds the request's slot (first not-yet-served entry with
//                 row r in the bank's sequence), then looks at the next `horizon` unserved entries of the bank: if
//                 any has row r the row is kept open (a hit is coming; FR-FCFS serves it first), otherwise the
//                 column command is upgraded to auto-precharge (explicit PREpb if RDA is not ready). Because the
//                 decision depends on arrival order, not on the order the scheduler happens to serve, it does not
//                 degrade when the scheduler reorders. horizon=1 means "the bank's very next access decides".
class OracleRowPolicy : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, OracleRowPolicy, "Oracle")

  AmcasRP m_;
  std::string m_mode, m_path, m_trace;
  int m_horizon = 16, m_window = 256;
  bool m_oracle = false, m_loaded = false;
  FILE* m_f = nullptr;
  long m_nlog = 0;
  std::vector<std::vector<int32_t>> m_seq;
  std::vector<std::vector<char>> m_done;
  std::vector<size_t> m_head;
  std::vector<char> m_pend, m_inj;

  void init() override {
    m_.bind(cast_parent<ControllerBase>());
    RAMULATOR_PARSE_PARAM(m_mode, std::string, "mode").default_val("record");
    RAMULATOR_PARSE_PARAM(m_path, std::string, "path").default_val("");
    RAMULATOR_PARSE_PARAM(m_trace, std::string, "trace").default_val("");
    RAMULATOR_PARSE_PARAM(m_horizon, int, "horizon").default_val(16);
    m_oracle = (m_mode == "oracle");
    m_pend.assign(m_.nbanks, 0);
    m_inj.assign(m_.nbanks, 0);
  }
  ~OracleRowPolicy() { if (m_f) fclose(m_f); }

  void load() {
    m_loaded = true;
    std::string p = m_path + "." + std::to_string(m_.ctrl->m_channel_id);
    m_seq.assign(m_.nbanks, {});
    m_done.assign(m_.nbanks, {});
    m_head.assign(m_.nbanks, 0);
    if (m_oracle) {
      FILE* f = fopen(p.c_str(), "rb");
      if (!f) throw std::runtime_error("Oracle: cannot open " + p);
      std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> mp;
      int64_t rec[2];
      while (fread(rec, sizeof(int64_t), 2, f) == 2)
        mp[(uint64_t)rec[0]] = {(int32_t)(rec[1] >> 32), (int32_t)(rec[1] & 0xffffffff)};
      fclose(f);
      std::ifstream t(m_trace);
      if (!t) throw std::runtime_error("Oracle: cannot open trace " + m_trace);
      std::string line;
      while (std::getline(t, line)) {
        size_t k = line.find("0x");
        if (k == std::string::npos) continue;
        auto it = mp.find(std::strtoull(line.c_str() + k, nullptr, 16));
        if (it == mp.end()) continue;
        int b = it->second.first;
        if (b >= 0 && b < m_.nbanks) m_seq[b].push_back(it->second.second);
      }
      for (int i = 0; i < m_.nbanks; i++) m_done[i].assign(m_seq[i].size(), 0);
    } else {
      m_f = fopen(p.c_str(), "wb");
      if (!m_f) throw std::runtime_error("Oracle: cannot create " + p);
      setvbuf(m_f, nullptr, _IOFBF, 1 << 20);
    }
  }

  // slot of this request: first unserved entry with row r within m_window live entries of the head; -1 if none
  long find(int b, int r) {
    auto& q = m_seq[b]; auto& d = m_done[b];
    int seen = 0;
    for (size_t i = m_head[b]; i < q.size() && seen < m_window; i++) {
      if (d[i]) continue;
      if (q[i] == r) return (long)i;
      seen++;
    }
    return -1;
  }
  // true if one of the next `horizon` unserved entries after slot idx has row r
  bool reuse_soon(int b, long idx, int r) {
    auto& q = m_seq[b]; auto& d = m_done[b];
    int seen = 0;
    for (size_t i = idx + 1; i < q.size() && seen < m_horizon; i++) {
      if (d[i]) continue;
      if (q[i] == r) return true;
      seen++;
    }
    return false;
  }

  void try_upgrade_command(Request& req) override {
    if (!m_oracle || !m_.is_col(req)) return;
    if (!m_loaded) load();
    int b = m_.bank_of(req);
    if (b < 0) return;
    int r = m_.row_of(req);
    long i = find(b, r);
    if (i < 0) return;
    if (!reuse_soon(b, i, r)) m_.auto_precharge(req);
  }

  void on_issue(const Request& req) override {
    if (!m_loaded) load();
    const auto& meta = m_.spec->command_meta[req.command];
    if (meta.is_closing && !meta.is_accessing) {
      m_.dev->for_each_target_bank(req.command, req.addr_vec, [&](int id) { m_pend[id] = 0; m_inj[id] = 0; });
    }
    if (meta.is_accessing && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b < 0) return;
      int r = m_.row_of(req);
      if (!m_oracle) {
        int64_t rec[2] = {(int64_t)req.addr, ((int64_t)b << 32) | (uint32_t)r};
        fwrite(rec, sizeof(int64_t), 2, m_f);
        if (++m_nlog % 8192 == 0) fflush(m_f);
        return;
      }
      long i = find(b, r);
      if (i >= 0) {
        bool close = !reuse_soon(b, i, r);
        m_done[b][i] = 1;
        while (m_head[b] < m_seq[b].size() && m_done[b][m_head[b]]) m_head[b]++;
        if (close && !meta.is_closing) m_pend[b] = 1;  // RDA not available/ready: explicit PRE later
      }
    }
  }

  void pre_schedule() override {
    if (!m_oracle) return;
    for (int i = 0; i < m_.nbanks; i++) {
      if (!m_pend[i] || m_inj[i]) continue;
      if (m_.inject_prepb(i)) m_inj[i] = 1;
    }
  }

  void post_schedule() override {
    if (m_f && m_.ctrl->m_clk % 100000 == 0) fflush(m_f);
  }
};

}  // namespace Ramulator
