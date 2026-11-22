# AMQSA / SDS-FDR — reproducibility package

Code, data, and result archives for the manuscript

> **An Adaptive Multi-Neighborhood Simulated Annealing Method for Flight Delay Recovery and Operational Analytics**
> Decision Analytics Journal, manuscript DAJOUR-D-26-00518 (under review).

The paper defines the **Sequence-Dependent Separation Flight Delay Recovery (SDS-FDR)** problem and proposes the **Adaptive Multi-Neighborhood Quenched Simulated Annealing (AMQSA)** method. In SDS-FDR, runway release times follow sequence-dependent separation rules, a recovery cutoff, and optional cancellations. This formulation removes the fixed-slot degeneracy of the total-delay objective.

## Scientific scope

- **Primary case.** A fixed **semi-synthetic 48-flight CGK-inspired departure bank** generated with seed 7. It is not an observed disruption-day OpenSky extract.
- **Baseline scenario.**
  - Reopening at 21:00 and an experimental recovery cutoff at 22:20.
  - 5,000 objective evaluations per run.
  - 20 common seeds (2000–2019).
- **Continuous benchmark diagnostics.** 10,000 evaluations per run, seeds 3000–3019.

## Repository structure

| Folder | Content |
|---|---|
| `data/` | Frozen primary instance (`CGK_48_FLIGHT_SEMISYNTHETIC_INSTANCE_SEED7.csv`) and data dictionary |
| `notebooks/` | Canonical Colab notebooks for every experiment in the revised manuscript (Phases 2–3G) |
| `results/` | Authoritative result archives used in the revised manuscript |
| `supplement/` | Flight-by-flight 48-flight feasibility detail and detailed statistical tables |
| `audit/` | Manuscript/code synchronisation and reference audits |
| `legacy/` | Single-file script of the original (pre-revision) submission, kept for provenance only |

## Experiment map

| Phase | Notebook | Purpose |
|---|---|---|
| 2 | `PHASE2_FAIR_BUDGET.ipynb` | Equal-budget primary comparison (GA, SA, PSO, ZOA, dZOA, AMQSA) and continuous diagnostics |
| 2B | `PHASE2B_FORMULATION_SENSITIVITY.ipynb` | Disappointment, spillover, and scalarization sensitivity |
| 2C | `PHASE2C_TEQ_CUTOFF_SENSITIVITY.ipynb` | Equivalent-cancellation delay (T_eq) and recovery-cutoff sensitivity |
| 3A | `PHASE3A_CONTROLLED_ABLATION.ipynb` | Controlled mechanism ablation |
| 3B | `PHASE3B_VNS_BASELINE.ipynb` | Permutation-native VNS comparator |
| 3C | `PHASE3C_MULTI_INSTANCE_SCALABILITY.ipynb` | Ten-instance suite and increasing-density scalability study |
| 3D | `PHASE3D_FLIGHT_FEASIBILITY.ipynb` | Flight-by-flight model-defined feasibility audit |
| 3E | `PHASE3E_HYPERPARAMETER_SENSITIVITY.ipynb` | Post-hoc hyperparameter robustness (not retuning) |
| 3F | `PHASE3F_EXACT_OPTIMALITY_GAP.ipynb` | Exact dynamic programming for N = 8–16 and heuristic optimality gaps |
| 3G | `PHASE3G_CANCELLATION_ECONOMICS.ipynb` | Cancellation-economics sensitivity |

Run the notebooks in the order shown. Each notebook contains its own environment lock, seed definitions, evaluation-budget assertions, and output-archive generation. The ZIP files in `results/` are the outputs reported in the manuscript.

## Interpretation limits

1. Separation values are **model-defined proxies**, not a certification of the complete CGK ATC rule set. "Feasible" means that a schedule satisfies the constraints implemented in the decoder.
2. Monetary coefficients are calibrated scenario parameters, not audited airline accounting values.
3. The multi-instance suite is semi-synthetic.
4. Exact optimality certificates apply only to the solved N = 8–16 subsets, not to N = 48.
5. The Phase-3C scalability study keeps the schedule-generation window fixed while N increases, so it is also an increasing-density stress test.

## Integrity

`MANIFEST_SHA256.txt` lists the SHA-256 hash of every file in this release.

## Contact

Hilal H. Nuha, School of Computing, Telkom University, Bandung, Indonesia — hilalnuha@ieee.org
