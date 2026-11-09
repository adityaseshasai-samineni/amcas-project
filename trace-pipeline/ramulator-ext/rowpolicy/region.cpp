#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "amcas_common.h"

namespace Ramulator {

// RegionPolicy (AMCAS proposal): the page policy is chosen per physical address range.
// Ranges come from a text file, one per line: "START END CLASS" (START inclusive, END exclusive,
// decimal or 0x hex, CLASS a single letter, e.g. W weights, K KV cache, O activations).
// Addresses are the physical addresses the controller sees (post placement, post mapping; the
// region mapper only changes bits below the 2 MiB granule, so granule ranges are mapping-independent).
// Classes listed in `close_classes` use a ClosedCAP-style policy (auto-precharge once `cap` column
// accesses have hit the bank, same convention as ClosedCAP); every other class, and addresses
// outside all ranges, are open-page.
class RegionRowPolicy : public IRowPolicy, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRowPolicy, RegionRowPolicy, "Region")

  AmcasRP m_;
  std::string m_ranges, m_close;
  int m_cap = 4;
  std::vector<uint64_t> m_s, m_e;
  std::vector<char> m_c;
  std::vector<int> m_acc;
  std::vector<char> m_wantclose, m_inj;

  void init() override {
    m_.bind(cast_parent<ControllerBase>());
    RAMULATOR_PARSE_PARAM(m_ranges, std::string, "ranges").default_val("");
    RAMULATOR_PARSE_PARAM(m_close, std::string, "close_classes").default_val("KO");
    RAMULATOR_PARSE_PARAM(m_cap, int, "cap").default_val(4);
    m_acc.assign(m_.nbanks, 0);
    m_wantclose.assign(m_.nbanks, 0);
    m_inj.assign(m_.nbanks, 0);
    std::ifstream f(m_ranges);
    if (!f) throw std::runtime_error("Region: cannot open ranges file '" + m_ranges + "'");
    std::string line;
    std::vector<std::tuple<uint64_t, uint64_t, char>> v;
    while (std::getline(f, line)) {
      std::istringstream is(line);
      std::string a, b, c;
      if (!(is >> a >> b >> c)) continue;
      v.emplace_back(std::stoull(a, nullptr, 0), std::stoull(b, nullptr, 0), c[0]);
    }
    std::sort(v.begin(), v.end());
    for (auto& t : v) { m_s.push_back(std::get<0>(t)); m_e.push_back(std::get<1>(t)); m_c.push_back(std::get<2>(t)); }
  }

  bool closes(Addr_t a) const {
    uint64_t x = (uint64_t)a;
    auto it = std::upper_bound(m_s.begin(), m_s.end(), x);
    if (it == m_s.begin()) return false;
    size_t k = (it - m_s.begin()) - 1;
    if (x >= m_e[k]) return false;
    return m_close.find(m_c[k]) != std::string::npos;
  }

  void try_upgrade_command(Request& req) override {
    if (!m_.is_col(req)) return;
    int b = m_.bank_of(req);
    if (b < 0 || m_acc[b] < m_cap || !closes(req.addr)) return;
    m_.auto_precharge(req);
  }

  void on_issue(const Request& req) override {
    const auto& meta = m_.spec->command_meta[req.command];
    if (meta.is_closing) {
      m_.dev->for_each_target_bank(req.command, req.addr_vec, [&](int id) { m_acc[id] = 0; m_wantclose[id] = 0; m_inj[id] = 0; });
    }
    if (meta.is_accessing && !meta.is_closing && !req.addr_vec.empty()) {
      int b = m_.bank_of(req);
      if (b >= 0) {
        m_acc[b]++;
        if (m_acc[b] >= m_cap && closes(req.addr)) m_wantclose[b] = 1;
      }
    }
  }

  void pre_schedule() override {
    // Fallback when auto-precharge was not available/ready: explicit PREpb.
    for (int i = 0; i < m_.nbanks; i++) {
      if (!m_wantclose[i] || m_inj[i]) continue;
      if (m_.inject_prepb(i)) m_inj[i] = 1;
    }
  }
};

}  // namespace Ramulator
