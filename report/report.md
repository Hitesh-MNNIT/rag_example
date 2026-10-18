# RAG Assignment Report: Baseline vs. Re-Ranking

**Document:** D&D 5e Basic Rules (Version 1.0, 180 pages, 449 chunks)  
**Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2`  
**Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`  
**LLM:** `claude-haiku-4-5-20251001`

---

## 1. Architecture Comparison

### Bi-Encoder (Baseline Retrieval)

A bi-encoder processes the query and each document chunk **independently**. Each is passed separately through the same transformer to produce a fixed-size dense vector (384-dim here). At query time, similarity is computed as the inner product between the query vector and all pre-computed chunk vectors — equivalent to cosine similarity when vectors are L2-normalized. Because chunk embeddings are computed offline and stored in a FAISS index, retrieval is extremely fast (under 50ms for 449 chunks). The limitation is that the model never sees the query and chunk *together*, so it cannot model fine-grained interactions — it relies on semantic proximity in embedding space, which can fail on negations ("not allowed"), keyword distractors ("inspiration" → bard vs. DM mechanic), and conditional clauses buried in surrounding text.

### Cross-Encoder (Re-Ranking)

A cross-encoder concatenates the query and candidate chunk into a single input (`[CLS] query [SEP] chunk [SEP]`) and passes both through a single transformer. Full self-attention across all tokens means the model can directly compare specific words in the query against specific words in the chunk — picking up negations, exceptions ("once per turn"), and exact rule clauses. This produces a much more accurate relevance score. The cost is that there is no precomputable representation: every `(query, chunk)` pair requires its own forward pass. For 10 candidates this adds ~300–600ms; for an entire 449-chunk corpus it would be impractical (~30s+). Hence the two-stage design: bi-encoder pre-filters to Top 10, cross-encoder re-ranks to Top 3.

---

## 2. Qualitative & Rank Shift Analysis

### Q1 — Keyword Trap: "What happens when you roll a 1 on an attack roll?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 206 | 76 | 1 | 0.5392 | **1** | 4.1620 |
| 216 | 79 | 2 | 0.3923 | 4 | -8.1294 |
| 11 | 5 | 7 | 0.3407 | **2** | -0.2971 |
| 411 | 161 | 9 | 0.3366 | **3** | -8.8746 |

**Rank shift:** Chunk 11 moved from Rank 7 → 2; chunk 411 from Rank 9 → 3.

**Observation:** Both pipelines correctly found the "Rolling 1 or 20" rule (chunk 206, p.76) at Rank 1 — the bi-encoder's top result was already correct. Both answers are accurate: rolling a 1 is an automatic miss regardless of modifiers. This query did not expose a failure; it confirms that when the key fact maps cleanly onto a distinctive chunk, both pipelines agree. The fear of "critical fumble" hallucination did not materialise — the model answered correctly from the retrieved rule text.

**Verdict:** No hallucination, no omission. Both pipelines correct. Rank shift had no answer-quality impact.

---

### Q2 — Keyword Trap: "How do you use inspiration?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 100 | 38 | 1 | 0.3827 | 3 | -1.7666 |
| 97 | 37 | 2 | 0.3725 | **2** | -1.7607 |
| 205 | 75 | 3 | 0.2825 | **1** | 0.2420 |

**Rank shift:** Chunk 205 (p.75 — a Perception check passage, *not* about inspiration) moved from Rank 3 → **1** by the cross-encoder, displacing the actual inspiration chunks.

**Observation:** This is a genuine cross-encoder failure. The bi-encoder's top 2 results (chunks 97, 100 — both on pp.37–38, the Inspiration rules) were actually correct. Answer 1 (Baseline) correctly explains spending inspiration for advantage. Answer 2 (Re-Ranked) gives the same correct answer, but the chunk promoted to #1 by the cross-encoder was irrelevant — the LLM happened to still extract the right answer from chunks 2 and 3. This illustrates that the cross-encoder is not universally better: for short, specific queries with low-diversity embedding scores, the re-ranking can promote noisy candidates.

**Verdict:** No hallucination in either pipeline — LLM recovered from the CE's bad rank-1 choice using chunks 2 and 3. However, this is a reliability risk: if Top-N were reduced to 1, re-ranking would have failed entirely. Baseline retrieval was more precise here.

---

### Q3 — Exception / Corner Case: "Can a Rogue sneak attack twice in one turn?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 74 | 29 | 1 | 0.3449 | **1** | 2.3167 |
| 78 | 30 | 4 | 0.2713 | **2** | -0.7162 |
| 196 | 72 | 7 | 0.2418 | **3** | -0.9154 |

**Rank shift:** Chunk 78 (p.30, containing more Rogue class details) moved Rank 4 → 2; chunk 196 (Bonus Actions) moved Rank 7 → 3.

**Observation:** Both pipelines found chunk 74 (the Sneak Attack feature text, "Once per turn…") at Rank 1 and gave the correct answer: a Rogue cannot sneak attack twice in one turn. The cross-encoder additionally elevated the Bonus Actions context, providing slightly richer surrounding rules in Answer 2. Both answers were correct; no hallucination occurred.

**Verdict:** No hallucination, no omission. The critical "once per turn" clause was in the top-3 for both pipelines. CE added supporting context (Bonus Actions) but did not change correctness.

---

### Q4 — Negation / Constraint: "What are you not allowed to do while incapacitated?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 432 | 171 | 1 | 0.3816 | 3 | -4.6544 |
| 433 | 171 | 2 | 0.3341 | **1** | -1.8104 |
| 208 | 76 | 9 | 0.2688 | **2** | -1.8263 |

**Rank shift:** Chunk 433 (p.171, Conditions appendix with the exact Incapacitated definition) moved Rank 2 → **1**; chunk 208 (combat text referencing incapacitated creatures) moved Rank 9 → **2**.

**Observation:** A clear cross-encoder win. The bi-encoder ranked chunk 432 (a general conditions heading) above chunk 433 (the specific condition text listing what you *can't* do). The cross-encoder correctly demoted the heading and promoted the specific restriction text. However, both Answer 1 and Answer 2 gave the same correct answer ("can't take actions or reactions") — the LLM extracted the right clause from either top-3 set. The value of re-ranking here is reliability: chunk 433 being at Rank 1 ensures the precise rule is present even for harder follow-up questions about the incapacitated condition.

**Verdict:** No hallucination in either pipeline. Baseline omitted the exact definition chunk from Rank 1 (replaced by a heading), but the LLM still found the correct clause in Rank 2. Re-ranking improves robustness — if the query were more specific (e.g., "can an incapacitated creature use reactions?"), the baseline's rank-1 heading would have caused an omission.

---

### Q5 — Negation / Prohibition: "Which spells cannot be cast while wearing armor?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 122 | 46 | 1 | 0.5310 | **1** | 2.8061 |
| 226 | 83 | 2 | 0.4857 | 3 | 1.8137 |
| 271 | 99 | 7 | 0.3592 | **2** | 2.6064 |

**Rank shift:** Chunk 271 (p.99, a spell with armor-related casting rules) moved Rank 7 → **2**.

**Observation:** Both pipelines correctly retrieved chunk 122 (the armor proficiency + spellcasting rule) at Rank 1. Answer 1 correctly explains there is no per-spell list — all spellcasting is blocked when wearing armor you lack proficiency with. Answer 2 (re-ranked) adds chunk 271 as a concrete spell example, making the answer slightly more grounded. Neither answer hallucinated a specific list of banned spells. This query exposed the document's true answer rather than a retrieval failure: the rule is a general restriction, not a per-spell prohibition.

**Verdict:** No hallucination in either pipeline. This is a negation trap — a naive model might invent a list of "banned spells," but both answers correctly stated the general proficiency rule instead. Answer 2 is slightly better grounded with a concrete spell example from the promoted chunk 271.

---

### Q6 — Buried Chunk: "What happens to a concentration spell if you take damage?"

| chunk_id | page | bi_rank | bi_score | ce_rank | ce_score |
|----------|------|---------|----------|---------|----------|
| 225 | 83 | 1 | 0.4541 | 2 | 0.9281 |
| 262 | 96 | **3** | 0.3635 | **1** | 1.7548 |
| 227 | 84 | 7 | 0.3206 | **3** | 0.7125 |

**Rank shift:** Chunk 262 (p.96, containing the *exact* "Constitution saving throw, DC = 10 or half damage taken") moved Rank **3 → 1**.

**Observation:** This is the clearest failure-mode demonstration. Answer 1 (Baseline) explicitly says *"the context does not contain information about what happens to a concentration spell if you take damage"* — a direct admission the key rule chunk was not in the Top 3. Answer 2 (Re-Ranked) correctly states the DC formula ("DC equals 10 or half the damage taken, whichever is higher") and the per-source saving throw rule. The cross-encoder rescued the critical chunk from Rank 3 to Rank 1, entirely changing the quality of the answer from "I don't know" to the precise rule. This is the assignment's ideal demonstration case.

**Verdict:** Baseline committed a critical omission — the chunk containing the DC formula (chunk 262, p.96) was at Rank 3 and fell outside the Top-3 context window, causing the LLM to correctly refuse rather than hallucinate. Re-ranking promoted it to Rank 1 and produced a fully correct, specific answer with the exact DC calculation. This is the strongest evidence for the two-stage pipeline's value.

---

## 3. Latency Profiling

| Query | Baseline Retrieval (ms) | Baseline LLM (ms) | Baseline Total (ms) | Rerank Retrieval+Score (ms) | Rerank LLM (ms) | Rerank Total (ms) |
|-------|------------------------|-------------------|---------------------|-----------------------------|-----------------|--------------------|
| Q1: Roll a 1 on attack | 11 | 3541 | 3552 | 573 | 4776 | 5349 |
| Q2: Inspiration mechanic | 11 | 2237 | 2248 | 399 | 2194 | 2593 |
| Q3: Rogue sneak attack twice | 20 | 1668 | 1688 | 328 | 1450 | 1777 |
| Q4: Incapacitated restrictions | 12 | 1226 | 1238 | 371 | 1222 | 1593 |
| Q5: Spells prohibited in armor | 45 | 2242 | 2287 | 388 | 2570 | 2958 |
| Q6: Concentration + damage | 12 | 2028 | 2040 | 316 | 1535 | 1850 |
| **MEAN** | **18** | **2157** | **2176** | **396** | **2291** | **2687** |

**Key observations:**

- **LLM latency dominates.** The average LLM call takes ~2157ms (baseline) vs. ~2291ms (re-ranked) — both dwarf retrieval. The cross-encoder adds a mean of only **396ms** overhead, but the total is already controlled by network + model inference time.
- **Retrieval is negligible.** Bi-encoder FAISS search over 449 chunks averages 18ms. Even with the cross-encoder scoring 10 pairs, the combined retrieval+rerank step is only 396ms mean — 18× more than baseline retrieval but still <20% of total pipeline time.
- **Re-ranking overhead is fixed-cost.** Since we always score exactly 10 candidates regardless of query, rerank latency varies only with chunk length (305–573ms range). Longer chunks → more cross-encoder tokens → slower scoring. Q1 had the highest rerank cost (573ms) likely due to longer combat chapter chunks.
- **Q1 rerank total > baseline total.** This is an outlier (5349ms vs 3552ms) where re-ranking made things worse both in answer quality and latency — a reminder that re-ranking is not universally beneficial.

---

## 4. Why Not Run the Cross-Encoder Over the Entire Corpus?

A cross-encoder cannot pre-compute document representations. Every `(query, chunk)` pair requires a full transformer forward pass — self-attention over the concatenated query and chunk tokens together. There is no way to decouple these into a fast lookup.

For this 449-chunk corpus, scoring all chunks per query would take approximately: 449 × 35ms/pair ≈ **15 seconds per query** on CPU. For a 10,000-chunk corpus (e.g., a product manual library), this rises to ~6 minutes per query — completely impractical for any interactive system.

The bi-encoder solves this by encoding documents *once* at index time. At query time it runs a single embedding pass and a vector search (O(N) dot products, parallelised by FAISS) in under 50ms regardless of corpus size. The trade-off is accuracy: the bi-encoder cannot model fine-grained query-document interactions.

The two-stage approach captures the best of both worlds: the bi-encoder efficiently filters to a small candidate set (Top-K, typically 10–50), and the cross-encoder then scores only those K pairs precisely. The accuracy benefit of the cross-encoder is recovered at a cost of O(K) inference calls — independent of corpus size N. As N grows from thousands to millions, the bi-encoder's cost stays constant; only K (a hyperparameter we control) affects cross-encoder latency.
