# Final Demo Guide (5–8 minutes)

## 0:00–0:45 — Problem

Open **Overview**. Explain that append-only RAG may keep obsolete knowledge retrievable and that Q-KEF adds a pre-indexing lifecycle decision.

## 0:45–1:30 — Architecture

Walk through sanitize → MiniLM → candidates → B/Q-state features → evolution KB → evidence. State that all quantum-inspired calculations are classical.

## 1:30–3:00 — REPLACE

Open **Evolution Demo**, choose `Stored Replace benchmark example`, select Q-KEF, and press **Analyze Evolution**. Discuss actual probabilities, candidate similarity, fidelity-like value, and effect preview. Press **Apply to Demo KB** and show the isolated state. **Reset Demo** restores it.

## 3:00–4:00 — B vs C

Open **B vs C**, press **Compare Conventional and Q-KEF**, and emphasize identical input/candidates.

## 4:00–5:00 — Retrieval and graph

Open **Retrieval** to compare obsolete@5; explain Oracle relevance honestly. Open **Semantic Graph** and show active/superseded symbols.

## 5:00–6:00 — Q-State

Open **Q-State**, press **Compute Q-State**, and show 16 normalized amplitudes and squared probabilities. Explicitly say “no hardware or simulator.”

## 6:00–7:00 — Results

Open **Research Results**: B 0.8455, C 0.8933, +0.0479 absolute macro F1. Mention paired statistical uncertainty.

## 7:00–8:00 — Limits

Open **Methodology & Limits**. Disclose controlled updates, 60-item TEST, COEXIST weakness, generic MiniLM, and human review.

## Backup path

If MiniLM interactive loading is unavailable, demo Overview, Retrieval, Graph (if artifacts load), Research Results, Error Analysis, and generated figures. Do not present a lexical fallback as official Q-KEF inference.
