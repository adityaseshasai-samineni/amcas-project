#!/usr/bin/env python3
import pathlib,sys
p=pathlib.Path(sys.argv[1]);h=p/'src/controller.h';s=h.read_text()
if 'HasPendingTransactions' not in s:
 s=s.replace('    int QueueUsage() const;', '''    int QueueUsage() const;
    // Demand state only; write acknowledgments alone occur before DRAM WR.
    bool HasPendingTransactions() const {
        return !pending_rd_q_.empty() || !pending_wr_q_.empty() ||
               !return_queue_.empty() || !unified_queue_.empty() ||
               !read_queue_.empty() || !write_buffer_.empty();
    }''');h.write_text(s)
h=p/'src/dram_system.h';s=h.read_text()
if 'HasPendingTransactions' not in s:
 s=s.replace('    int GetChannel(uint64_t hex_addr) const;', '''    int GetChannel(uint64_t hex_addr) const;
    bool HasPendingTransactions() const {
        for (const auto *c : ctrls_) if (c->HasPendingTransactions()) return true;
        return false;
    }''');h.write_text(s)
h=p/'src/memory_system.h';s=h.read_text()
if 'HasPendingTransactions' not in s:
 s=s.replace('    void PrintStats() const;', '''    void PrintStats() const;
    bool HasPendingTransactions() const { return dram_system_->HasPendingTransactions(); }''');h.write_text(s)
