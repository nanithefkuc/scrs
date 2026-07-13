# Branch disposition

This document records the Phase 0 consolidation decisions for the `v1` publication baseline.

| Branch | Disposition | Rationale |
| --- | --- | --- |
| `master` | Merged into `v1` | Its scalar-finalization work was reviewed during the merge. Where it conflicted with `simd`, the newer shared scale-bank and grouped-kernel implementation was retained. |
| `simd` | Merged into `v1` | Selected as the integration base because it contains the accepted SIMD and encoder optimizations. |
| `opt/gfni-source-major` | Merged (already reachable) | Its commit is an ancestor of the `simd` integration base. |
| `opt/source-major-reconstruction` | Merged (already reachable) | Its commit is an ancestor of the `simd` integration base. |
| `exp/phase7-followups` | Merged (already reachable) | Its commit is an ancestor of the `simd` integration base. |
| `exp/xor-slp-phase6` | Merged (already reachable) | Its commit is an ancestor of the `simd` integration base. |
| `xor-scheduling` | Intentionally rejected from `v1`; retained for reference | It is a separate speculative decoder scheduling experiment and has not been validated as part of the accepted baseline. |
