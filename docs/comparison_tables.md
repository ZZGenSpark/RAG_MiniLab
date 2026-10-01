# Comparison tables

A labeled question is recalled when every expected section is in the top 3. 12 of the 13 questions have an expected section.

## Route

Same numbered-rule index, MiniLM, and cross-encoder.

| Setup | Recalled |
| --- | --- |
| Always vector | 12/12 |
| Always hybrid | 12/12 |
| Section-code router | 12/12 |

## Reranker

Section-code router on the numbered-rule index. Off keeps the shortlist order. On scores it with the cross-encoder.

| Reranker | Recalled |
| --- | --- |
| Off | 12/12 |
| On | 12/12 |

## Jev

Each router chooses vector or hybrid. Qwen is qwen3:8b-q4_K_M.

| Router | Section questions hybrid | Other questions vector | Mean call time |
| --- | --- | --- | --- |
| Section-code rule | 4/4 | 9/9 | 0.0 ms |
| Jev | 4/4 | 9/9 | 172.6 ms |
| Qwen | 4/4 | 9/9 | 299.1 ms |

Produced by `scripts/compare_retrieval.py`.
