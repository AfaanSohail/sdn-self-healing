# Computer Networks Phase 3 Final Submission

Safety-constrained SDN self-healing: detection, diagnosis, recovery and verification. Student IDs: **24i0081 and 24i0005**.


## Submission contents

This archive includes the implementation and evidence supporting requirements mentioned in requirement document.

| Requirement | Evidence |
|---|---|
| 1. Implement the proposed methodology and document changes | `src/`, trained `checkpoints/`, report Sections 1-3 |
| 2. Dataset, preprocessing, features, setup and splits | `data/`, `src/dataset.py`, `src/features.py`, `results/split_manifest.csv`, `results/spotlight_split_manifest.csv`, `results/preprocessing_trace.csv`, `results/feature_schema.json`, `results/topologies.json` |
| 3. Baseline evaluation | Detection metrics in `results/detection_metrics.json` and `results/spotlight_metrics.json`; B0-B4 outcomes in `results/episodes.csv` |
| 4. Graphical results | `results/figures/` and the figures embedded in the report |
| 5. Comparison with existing research | Report Section 14, the separate public-sample study and matched local baseline results |
| 6. Ablation studies | A1-A6 and oracle outcomes, `results/ablation_statistics.json`, ablation figures; A3 is B2 rather than a duplicated result |
| 7. Results analysis and discussion | Report, `results/summary.json`, `results/paired_statistics.json`, per-seed tables, stage/failure records and verified replay examples |

### Files and purpose

| Path | Purpose |
|---|---|
| `Phase3_Implementation_Report.pdf` | Main report for assessment |
| `Phase3_Implementation_Report.docx` | Editable version of the report |
| `README.md` | Submission guide, run instructions and metric definitions |
| `requirements.txt` | Compatible Python dependencies |
| `requirements-lock.txt` | Recorded versions of the main dependencies |
| `SHA256SUMS.txt` | Integrity hashes for every other packaged file |
| `src/` | Complete implementation, baselines, training, evaluation, analysis, checks and report generation |
| `data/` | Synthetic feature matrices and official SpotLight sample inputs |
| `checkpoints/` | Selected/final DQN weights, learning histories, supervised models and preprocessing |
| `results/` | Raw episode/action outputs, split/seed manifests, metrics, statistics, figures, failure examples and validation records |


## Executed experiment and scope

- 23,760 recovery/control episodes: 30 matched blocks x 11 scenario families x 3 loads x 2 topologies x 12 methods.
- 20 independently seeded DQN training runs: Full/A1/A2/A5 x five seeds, 50,000 environment steps per run.
- All five trained seeds are evaluated in balanced blocks, six blocks per seed per stratum.
- Binary detection, component diagnosis, B0-B4 recovery baselines, A1-A6 ablations and oracle diagnosis.
- 38 coverage/consistency checks and 21 representative failure/stress replays were verified in the delivered study.

Recovery is a **Python service-KPI/routing simulation**, not a Mininet/Open vSwitch/OpenFlow or physical-network experiment. The separate external detection experiment uses the **official SpotLight platform sample**, not the full dataset or RANGAN's exact network-contention benchmark. No 5G-NIDD result or production safety guarantee is claimed. These limits are discussed in the report. The student confirmed that the teacher gave no specific class feedback.

## Setup

Python 3.12 was used. Run commands from the extracted `CN_Phase3_Submission` folder.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. For the recorded main dependency versions, use `requirements-lock.txt` with the recorded Python version. If a CPU-only PyTorch installation is needed:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Times New Roman must be available to reproduce the report typography. LibreOffice is required to regenerate the PDF; without it, the report script writes the DOCX for export through Word. The supplied PDF is ready to read.

## Check the delivered implementation

```bash
python src/selftest.py
python src/audit.py
```

`selftest.py` checks seeded resets, stochastic variation, causal/vectorized features, stale-data shielding, counterfactual harm/rollback, controls and disconnected endpoint handling. `audit.py` checks coverage, balanced training seeds, split separation, checkpoint histories and result consistency. It refreshes structured audit metadata; it does not retrain models or rerun the complete recovery evaluation.

Read `results/validation.json` for the saved check results. Linux/macOS users with `sha256sum` can verify the untouched archive contents using:

```bash
sha256sum -c SHA256SUMS.txt
```

Verify hashes before regenerating outputs, since regenerated files may differ in timing or document metadata.

## Rebuild analysis and the report

```bash
python src/run.py --analysis-only
```

This recalculates statistical tables and figures, replays representative examples using frozen checkpoints, refreshes the audit and rebuilds the report. It does not retrain the DQN or repeat all 23,760 evaluation episodes. Replayed wall-clock timings can differ from the original run.

## Reproduce the complete experiment

This command overwrites generated data, models and results:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/run.py --full --workers 8
```

For Windows, set those environment variables using the shell's syntax, then run `python src/run.py --full --workers 8`. Reduce `--workers` for a smaller machine. The full command downloads/verifies the public sample, prepares synthetic data, trains supervised models and DQN policies, evaluates the complete grid, and rebuilds the analysis/report. It can take substantial time.

Eight worker processes were used in the delivered run. Methods execute sequentially in seeded random order within each block, while independent blocks run concurrently. Service outcomes follow the recorded seeds; shared-host wall-clock timings are not isolated hardware benchmarks. `results/run_manifest.json` records the execution details and limitations.

## Data dictionary and metric definitions

An episode lasts 180 one-second bins. Injection begins at 60, clearance at 120, and the cascading secondary fault starts at 70. Congestion has a pre-injection demand ramp beginning at 50; A4 tests anticipatory recovery on this precursor only. The two topologies have 12 or 20 switches, two host attachments per switch and three continuously protected flows of 9/11/13 Mbps. The host attachments are metadata; no host processes run.

`feasible` is 1 for fault episodes with all protected endpoints structurally connected at seconds 65 and 80; controls are excluded and physical cuts retained separately. This is structural, not a proof of sufficient SLA-compliant capacity. `restored` requires all three flows healthy for five consecutive seconds beginning before fault clearance. `restoration_seconds` is the start of that interval minus 60, and is empty for unrecovered or infeasible cases. `capped_unrecovered_seconds` is the same latency for recovered feasible faults and 60 otherwise. Conditional restoration median/p95 exclude failures, while the CDF includes them in its denominator.

`sla_burden` is violating flow-seconds / 180 protected-flow-seconds during [60,120). `collateral_burden` uses only flows whose original paths avoid the injected components. Loss and goodput are modeled service values, not packet-capture measurements. RTT is milliseconds; goodput is Mbps. `clean_sla_burden` denotes the entire pre-injection interval [0,60), including the congestion precursor at [50,60); it is not purely clean warm-up. Exploratory sensitivity selects whole blocks using passive B0 pre-injection health and does not replace the primary comparison. SLA: RTT <=20 ms, loss <=1%, goodput >=90% of flow demand.

`harmful` counts route changes that introduce an unaffected-flow violation, increase affected-flow loss by >2 percentage points or RTT by >5 ms at the next sample relative to **keeping the previous routes under the exact same exogenous traffic**. The verifier then restores the prior route and increments `rollbacks`. This conservative single-sample test differs from Phase 2's two-sample collateral-only criterion; it is explicitly disclosed. `route_changes` includes route installs and verifier/agent rollbacks; `route_changes-rollbacks` counts forward installs. No physical-world safety guarantee follows.

Byte counters are differenced across received measurements, divided by elapsed time, and reset if a decrease is observed. Short dropouts carry the last received values, with `missing` and age inputs. Longer gaps retain stale values explicitly and the shield rejects paths using observations older than two seconds. This is an age-gated carry-forward implementation, not the training-median fallback originally planned. Continuous link measurements include seeded noise; service probes are the model's observable physical KPI calculation. The same exogenous arrays are reused across paired methods. No ground-truth fault labels drive deployable decisions. Only the separately named oracle has privileged labels.

Detection tables report row/window metrics from episode-disjoint synthetic test data. Their rows are not independent replications. The operational control files report deduplicated consecutive alarm events. `localisation_*_t65` is a snapshot top-one correctness audit at t=65, separately unconditional and conditional on triggering; it is not a complete event-level root-cause validation. Unknown flags are diagnostic only, uncalibrated and not an action veto.

The main inference uses 30 reset schedule bundles, each averaging seven fault scenarios x three loads. Tests condition on five fixed trained policies. Hierarchical intervals resample five model-seed groups and six environmental bundles per seed, retaining all matched methods and strata. This avoids counting thousands of overlapping telemetry windows as independent training or evaluation replicates. The five-seed limitation remains.

