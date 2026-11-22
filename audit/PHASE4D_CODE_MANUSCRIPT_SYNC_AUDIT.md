# Phase 4D — Code / Manuscript Synchronization Audit

## Core authoritative Phase-2 configuration

- PASS — USE_OPENSKY_FILE=False
- PASS — data seed 7
- PASS — primary MAX_EVALS=5_000
- PASS — benchmark BENCH_MAX_EVALS=10_000
- PASS — recovery cutoff=260

## Canonical notebook narrative scan

| Phase | 23:00 stale value | Misleading real/OpenSky claim | 'night curfew' mentions | Semi-synthetic mentions |
|---|---:|---:|---:|---:|
| Phase2 | NO | NO | 0 | 3 |
| Phase2B | NO | NO | 0 | 2 |
| Phase2C | NO | NO | 2 | 2 |
| Phase3A | NO | NO | 0 | 2 |
| Phase3B | NO | NO | 0 | 3 |
| Phase3C | NO | NO | 0 | 3 |
| Phase3D | NO | NO | 0 | 2 |
| Phase3E | NO | NO | 0 | 3 |
| Phase3F | NO | NO | 0 | 3 |
| Phase3G | NO | NO | 0 | 2 |

### Interpretation

- No canonical notebook contains the stale **23:00** recovery-horizon value.
- No canonical notebook contains the misleading claim that the reported experiment is an observed OpenSky disruption-day extract.
- Phase 2C uses **night curfew** only to explain that the term is unsupported and is replaced by **experimental recovery cutoff**.
- Phase 3D mentions real CGK/JATSC procedures only to state that model-feasibility is **not** regulatory certification.

## Release decision

The canonical Phase 2–3G notebook set is synchronized with the revised manuscript on the key provenance, horizon, and evaluation-budget issues. Legacy pre-revision notebooks should not be included in the public release. The Phase-4D staging package contains only the canonical revision notebooks.