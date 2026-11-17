# jero-kv-cache

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-kv-cache/card.svg" alt="jero-kv-cache Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kv-cache/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kv-cache/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kv-cache/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kv-cache/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Context memory holder daemon and shared memory arena management for the SpacemiT K3 AI platform.

## Overview

`jero-kv-cache` is a minimal, policy-free daemon (`jero-kv-cache`) and shared-memory arena library holding per-model context memory arenas:

- **Policy-Free Memory Holder**: Allocates and locks (`mlock`) shared-memory arenas within operator `RLIMIT_MEMLOCK` bounds.
- **FD Passing via `SCM_RIGHTS`**: Passes arena file descriptors to worker and controller processes over Unix Domain Sockets (`KvServer`).
- **Vector-Aligned KV Planes & Batched Views**: Stores `F16` and `Q8_0` (plus `F32`) KV slabs aligned to 128 bytes (`VECTOR_PLANE_ALIGNMENT = 128`) with zero-copy batched `ArenaKvWriter` / `ArenaKvReader` access and `DoryVirtualKvMapper` (`per_loop` / `last_loop`).
- **O(1) Recurrent State Arena**: Maintains dual ping-pong recurrent/SSM state banks (`recurrent_banks_in_arena_mut`, `commit_recurrent_step_in_arena`) for zero-tearing state transitions.
- **Fault Survival**: Survives worker abnormal termination and supervisor reconnects, retaining active KV context blocks.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependency**: [`jero-core`](https://github.com/LMA1980/jero-core)
- **Companion Crates**: [`jero-engine`](https://github.com/LMA1980/jero-engine), [`jero-manager`](https://github.com/LMA1980/jero-manager), and [`jero-arch-dory`](https://github.com/LMA1980/jero-arch-dory)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
