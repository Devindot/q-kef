# Real Temporal Data Plan

`TemporalDatasetAdapter` defines local JSONL and CSV ingestion for source document ID, version ID, content, valid/system time, optional authority, predecessor references, and optional annotations. Toy fixtures validate parsing only and must not be reported as real-world evidence.

A future dataset must have a documented license, naturally occurring versions, immutable source capture, authenticated timestamps where possible, and an annotation protocol separating known lineage from inferred lifecycle action. Candidate sources include permissively licensed public policy/document revision histories and local Git histories owned by the researcher. No private or copyrighted corpus should be downloaded silently.

The confirmatory split must group complete version families and shared query ancestry. TRAIN/DEV/CAL/TEST source and query intersections must be zero. Unknown effective dates and authority remain null rather than inferred from text.
