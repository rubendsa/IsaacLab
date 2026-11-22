# Native N16 learner measurements

[Summary and camera audit](summary.json) · [Artifact hashes](manifest.json) ·
[Technical report](../../SRAM_TRAINING.md#complete-native-learner-measurements-at-16-environments)

The primary synthetic record is `synthetic_full16_four.json.gz`, independently checked in
`synthetic_four_independent_audit.json.gz`. Eight balanced, screened cycles compare complete native
`PPO.update` calls with matched fused Adam and identical resumed state, observations and RNG.
Median host times are 246.02 ms native, 78.53 ms separate persistent networks, 79.25 ms combined
networks and 89.03 ms combined networks with resident inputs. GAE and collection are outside this
measurement. All state, loss and exact CPU/CUDA RNG checks pass. The resident path retains more
inputs but takes 12.34% longer than the combined path.

This record was collected from a prepared working tree over `beecb2dc4`, with 66 unchanged source
hashes, not a clean commit. The raw collector's `publication_allowed: false` placeholder is
preserved; acceptance of this bounded comparison is recorded in the independent audit and summary.
The collector and oracle snapshots are archived alongside it.

The older six-cycle record remains as `synthetic_full16_fused.json.gz`. It has a clean-commit source
record but omitted the native `agents/models.py` hash before execution. A later match to the commit
does not establish that file's immutability during the run. The newer four-way record includes it.

The camera pair uses the native RGB64 task at 16 environments, compact histories, IEEE float32,
seed 42 and the native reset bank. Both 30-iteration runs passed supervisor screening; rows 5–29
give median learning times of 213.43 → 79.82 ms and complete iteration times of 575.01 → 434.13 ms.
Here learning includes GAE and PPO update. Native Adam uses its default implementation; the combined
backend uses fused Adam. These are separate trajectories in one fixed run order, not repeated paired
trials. Both logged zero training success. The records establish throughput and execution, not
convergence or held-out task success.

Screening uses sampled GPU process queries, with their timing gaps audited; it cannot rule out
activity between samples. CPU load and affinity were not controlled. The report records timing
boundaries and these limits in full.

Raw JSON, logs, configuration and source snapshots are gzip archives with timestamp zero, preserving
the original bytes against text formatting hooks. For example:

```bash
gzip -dc docs/sram/data/unified_performance/synthetic_full16_four.json.gz | python -m json.tool
```

The manifest records each archive's SHA-256 and the original uncompressed SHA-256. Summaries are
derived records; raw source paths identify where evidence was collected and need not exist elsewhere.
