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

---

Context memory holder daemon and shared memory arena management for the SpacemiT K3 AI platform.

## Overview
`jero-kv-cache` is a minimal, policy-free daemon process holding per-model context memory arenas:
- **Policy-Free Memory Holder**: Allocates and locks (`mlock`) shared-memory arenas within operator `RLIMIT_MEMLOCK` bounds.
- **FD Passing via SCM_RIGHTS**: Passes arena file descriptors to worker and controller processes over Unix Domain Sockets.
- **Fault Survival**: Survives worker abnormal termination and supervisor reconnects, retaining active KV context blocks.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [jero-runner](https://github.com/LMA1980/jero-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/jero-runner/blob/main/PLAN.md)
- Specs: [specs/memory/kv-holder.md](https://github.com/LMA1980/jero-runner/blob/main/specs/memory/kv-holder.md)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
