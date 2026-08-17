# Quantum-Inspired State Model

Q-KEF's third system is entirely classical. “Quantum-inspired” describes a real-valued state representation and comparison features motivated by the geometry of normalized states; it does not use quantum hardware, circuits, amplitudes, or claim quantum advantage.

The implementation first embeds text with `sentence-transformers/all-MiniLM-L6-v2`. A PCA basis is fit only on knowledge units whose benchmark split is TRAIN. DEV selects the PCA dimension from 8, 16, and 32, jointly with the same logistic-regression hyperparameter grid used by the conventional baseline. The held-out TEST split is evaluated only after all choices are written to `locked_config.json`.

Each PCA projection is L2-normalized. Candidate-pair features include squared inner product (fidelity-like similarity), Shannon entropy of squared components, an L1-derived coherence-like score, and Hellinger distance between the induced probability vectors. Aggregate features use the identical top-five candidate set supplied to the conventional system. PCA fit identifiers are retained in the manifest for provenance validation.

These quantities are engineered classical features. Their scientific value is assessed through the controlled B-versus-C ablation, not assumed from terminology.
