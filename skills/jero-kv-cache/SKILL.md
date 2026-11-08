---
name: jero-kv-cache
description: "Manage shared-memory KV cache arenas, memfd SCM_RIGHTS descriptor passing, and bounded RLIMIT_MEMLOCK ceilings with jero-kv-cache."
---

# `jero-kv-cache` Shared-Memory KV Arena & `RLIMIT_MEMLOCK` Invariants

## 1. Bounded `RLIMIT_MEMLOCK` Ceiling (Never `unlimited`)

- Never configure or assume `RLIMIT_MEMLOCK = unlimited`; locking 100% of physical RAM starves Linux kernel slab allocation, page table creation, and SSH daemon responsiveness.
- Always enforce a bounded lockable-memory ceiling that reserves safe OS headroom (e.g., 28 GiB maximum on a 32 GiB headless target node), and fail admission cleanly (`InsufficientLockableMemory`, exit code `11`) when requested arenas exceed available lockable memory.
- In unit tests, integration tests, and diagnostic probes, default to conservative context bounds (e.g., `n_ctx: 64`, requiring ~64 KiB KV cache) so tests pass in unprivileged environments with standard 64 MiB `ulimit -l`.
