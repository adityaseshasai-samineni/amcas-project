from pathlib import Path
p=Path.home()/'amcas-pro/tracing/ramulator2-gpu-drain/src/ramulator/frontend/impl/memory_trace/loadstore_trace.cpp'
s=p.read_text();s=s.replace('size_t m_trace_count = 0;', 'size_t m_trace_count = 0;\n  size_t m_completed = 0;\n  size_t m_drain_guard = 0;')
s=s.replace('const Trace& t = m_trace[m_curr_trace_idx];','if (m_trace_count >= m_trace_length) {\n      if (m_completed == m_trace_length) ++m_drain_guard;\n      return;\n    }\n    const Trace& t = m_trace[m_curr_trace_idx];')
s=s.replace('req.size_bytes = m_memory_system->get_tx_bytes();','req.size_bytes = m_memory_system->get_tx_bytes();\n    req.callback = [this](Request&) { ++m_completed; };')
s=s.replace('return m_trace_count >= m_trace_length;', 'return m_trace_count >= m_trace_length && m_completed == m_trace_length && m_drain_guard >= 4096;')
p.write_text(s)
