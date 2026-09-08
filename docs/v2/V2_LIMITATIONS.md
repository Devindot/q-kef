# Q-KEF v2 Limitations

- The implemented experiment is retrospective on the frozen 300-event synthetic FiQA benchmark; it is not the planned 2,400-event confirmatory benchmark.
- CAL is deterministically reserved from old TRAIN ancestry, reducing fitting data.
- Equal-weight RRF underperformed dense retrieval; no TEST tuning is permitted.
- Hierarchical models did not outperform flat B0 in the pilot.
- Q-full did not beat B0 and its matched-control advantage was small and non-significant.
- Witness queries are deterministic lexical probes, not observed enterprise workloads; current-evidence miss was high.
- In-memory atomic publication is not a distributed two-phase commit implementation.
- Authority tests are synthetic; FiQA lacks reliable authority metadata.
- Bitemporal fields are supported, but FiQA mutations are controlled rather than naturally longitudinal.
- Local latency/memory numbers do not generalize to production scale.
- Prior-art notes are technical research aids, not legal advice or a patentability opinion.
