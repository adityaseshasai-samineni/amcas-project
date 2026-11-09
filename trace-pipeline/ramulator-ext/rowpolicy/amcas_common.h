// Shared helpers for the AMCAS row policies (Region, Timeout, HAPPY). Not a plugin.
#ifndef AMCAS_ROWPOLICY_COMMON_H
#define AMCAS_ROWPOLICY_COMMON_H

#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "ramulator/base/param.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/rowpolicy/i_row_policy.h"

namespace Ramulator {

// Command ids, flat-bank helpers and "close this bank" primitives.
struct AmcasRP {
  ControllerBase* ctrl = nullptr;
  DRAMDevice* dev = nullptr;
  const DRAMSpec* spec = nullptr;
  int prepb = -1, rd = -1, wr = -1, rda = -1, wra = -1, row_level = -1;
  bool has_ap = false;
  int nbanks = 0;

  void bind(ControllerBase* c) {
    ctrl = c;
    dev = &c->m_device;
    spec = dev->m_spec;
    auto find = [&](std::initializer_list<const char*> names, bool req) {
      for (const char* n : names)
        if (spec->has_command(n)) return spec->get_command_id(n);
      if (req) throw std::runtime_error("AMCAS row policy: DRAM standard lacks RD/WR");
      return -1;
    };
    prepb = spec->get_command_id("PREpb");
    rd = find({"RD", "RD_S"}, true);
    wr = find({"WR", "WR_S"}, true);
    rda = find({"RDA", "RDA_S"}, false);
    wra = find({"WRA", "WRA_S"}, false);
    has_ap = rda != -1 && wra != -1;
    row_level = spec->get_level_id("Row");
    nbanks = static_cast<int>(dev->m_bank_nodes.size());
  }
  int bank_of(const Request& r) const {
    if (r.addr_vec.empty()) return -1;
    int b = dev->get_flat_bank_id(r.addr_vec);
    return (b >= 0 && b < nbanks) ? b : -1;
  }
  int row_of(const Request& r) const { return r.addr_vec[row_level]; }
  bool is_col(const Request& r) const { return r.command == rd || r.command == wr; }
  // Rewrite RD->RDA / WR->WRA if AP exists and its timing is ready. Returns true if rewritten.
  bool auto_precharge(Request& req) {
    if (!has_ap) return false;
    if (req.command == rd && ctrl->check_timing(rda, req.addr_vec)) {
      req.command = rda; req.final_command = rda; return true;
    }
    if (req.command == wr && ctrl->check_timing(wra, req.addr_vec)) {
      req.command = wra; req.final_command = wra; return true;
    }
    return false;
  }
  AddrVec_t bank_addr(int i) const {
    AddrVec_t a(spec->level_count, -1);
    for (auto* n = dev->m_bank_nodes[i]; n; n = n->m_parent_node) a[n->m_level] = n->m_node_id;
    return a;
  }
  // Inject an explicit PREpb for flat bank i; true if accepted.
  bool inject_prepb(int i) {
    AddrVec_t a = bank_addr(i);
    if (!ctrl->check_timing(prepb, a)) return false;
    Request r(std::move(a), Request::Cmd, prepb);
    return ctrl->priority_send(r);
  }
};

}  // namespace Ramulator
#endif
