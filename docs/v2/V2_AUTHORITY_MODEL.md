# Authority-Aware Conflict Model

Optional fields are `source_authority`, `approval_level`, `department`, `document_type`, `jurisdiction`, and integer `source_priority`. They are separate from classifier labels and default to unknown.

The implemented conservative rule rejects automatic REPLACE when both priorities are known and the incoming priority is lower than the predecessor. Other actions remain subject to policy extension. Synthetic tests cover regulator-versus-blog conflict; no FiQA result is represented as natural authority evidence.

Production policy would need organization-specific approval mappings, jurisdiction interactions, authenticated provenance, and human escalation. Numeric priority alone must not be interpreted as a universal truth ranking.
