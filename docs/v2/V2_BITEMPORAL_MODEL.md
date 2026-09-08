# Bitemporal Model

Each v2 record may hold `valid_from`, `valid_to`, `system_from`, and `system_to`. Valid time represents when content applies in the domain; system time represents when Q-KEF stored it. Unknown values remain null.

Intervals are half-open: a record is visible when `start ≤ query_time < end`. Active retrieval additionally requires `RetrievalState.ACTIVE`. Historical/as-of retrieval includes ACTIVE and HISTORICAL_ONLY records but excludes QUARANTINED and BLOCKED records. Constructors reject reversed intervals, and transition validation exposes temporal checks in certificates.

FiQA authority and domain-effective dates are not fabricated. Its existing construction timestamps remain controlled benchmark metadata, not naturally observed business history.
