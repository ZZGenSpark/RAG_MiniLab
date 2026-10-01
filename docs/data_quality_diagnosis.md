# Data quality diagnosis

Question asked through `ask`:

> How many tokens does each employee receive at the start of a cycle?

Answer:

> The provided policy does not answer this question.

The response has no citations. Retrieval was `vector`. These chunk ids reached the model, in rerank order:

- `time-and-usage-policy:v1.0:section-5.1` — 1,000,000 tokens per cycle
- `time-and-usage-policy:v2.0:section-6.1` — 500,000 tokens per cycle
- `time-and-usage-policy:v2.0:section-7.1` — how to request more tokens

Retrieval worked. Version 2.0 restates the allocation rule as section 6.1 and changes the amount from 1,000,000 to 500,000. Both copies are kept. Generation asks the model to refuse when excerpts from the same document disagree, and it refuses a one-sided answer when those amounts differ. Agreeing copies, such as both foosball allowances at 20 minutes, still produce an answer.

`source_conflicts` on that response:

```json
[
  {
    "document": "Time & Usage Policy",
    "versions": ["1.0", "2.0"]
  }
]
```

The same title is stored at 1.0 and 2.0. The refusal is the tested result of that conflict, not a retrieval miss.
