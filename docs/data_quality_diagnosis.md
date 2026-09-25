# Data quality diagnosis

Question asked through `ask`:

> How many tokens does each employee receive at the start of a cycle?

Answer:

> The provided policy does not answer this question.

The response has no citations. Retrieval was `vector`. These chunk ids reached the model, in rerank order:

- `time-and-usage-policy:v1.0:section-5.1` — 1,000,000 tokens per cycle
- `time-and-usage-policy:v2.0:section-6.1` — 500,000 tokens per cycle
- `time-and-usage-policy:v2.0:section-7.1` — how to request more tokens

Retrieval worked. The allocation rule is in the shortlist. Version 2.0 restates it as section 6.1 and changes the amount from 1,000,000 to 500,000. Both amounts were in the prompt, so the excerpts do not support one figure. Generation refuses when it cannot ground a single answer, which is this response.

`source_conflicts` on that response:

```json
[
  {
    "document": "Time & Usage Policy",
    "versions": ["1.0", "2.0"]
  }
]
```

The same title is stored at 1.0 and 2.0. Dropping the outdated 1.0 copy would leave 6.1 as the only allocation rule: 500,000 tokens. The failure is the corpus, not retrieval or generation.
