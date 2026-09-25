# Data quality diagnosis

Question asked through `ask`:

> How many tokens does each employee receive at the start of a cycle?

Answer:

> The provided policy does not answer this question.

The response has no citations. Retrieval used the vector strategy. These are the chunk ids sent to the model, in rerank order:

- `time-and-usage-policy:v1.0:section-5.1`
- `time-and-usage-policy:v2.0:section-6.1`
- `time-and-usage-policy:v2.0:section-7.1`

`source_conflicts` on that response:

```json
[
  {
    "document": "Time & Usage Policy",
    "versions": ["1.0", "2.0"]
  }
]
```

Both versions of Time & Usage Policy were retrieved and given to the model. The same document title is present at version 1.0 and version 2.0, which is what `source_conflicts` records. The failure is the corpus.
