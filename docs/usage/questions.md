# Questions

`ask` takes a **state** — any JSON-serializable value: a string, a mapping, a
list — and a mapping of question name to question. Every question is one of
three types, discriminated by `type`.

Questions can be plain dicts or the model classes; both validate to the same
thing.

=== "Dicts"

    ```python
    agent.ask(state, {"is_spam": {"type": "noul", "instructions": "Is this spam?"}})
    ```

=== "Models"

    ```python
    from system_one import Noul

    agent.ask(state, {"is_spam": Noul(instructions="Is this spam?")})
    ```

## `noul` — yes/no

The answer is a probability that the statement holds, not a boolean.

```python
{
    "type": "noul",
    "instructions": "Does this need a human now?",
}
```

`criteria` is optional and describes what each outcome means, which helps when
"true" is ambiguous:

```python
{
    "type": "noul",
    "instructions": "Should this be escalated?",
    "criteria": {
        "true": "a human must respond within the hour",
        "false": "the queue can handle it",
    },
}
```

## `choice` — pick one

`criteria` is required and holds at least one label. Pass a list when the labels
speak for themselves:

```python
{
    "type": "choice",
    "instructions": "Which queue?",
    "criteria": ["billing", "technical", "other"],
}
```

…or a mapping when they need describing. The list form is sugar for a mapping
with `None` descriptions, so the two are interchangeable:

```python
{
    "type": "choice",
    "instructions": "Which queue?",
    "criteria": {
        "billing": "charges, refunds, invoices",
        "technical": "the product is broken",
        "other": None,
    },
}
```

## `score` — ordinal

`criteria` is an **ordered** sequence, and a criterion's index *is* its score.
Order matters: `["minor", "normal", "major", "critical"]` scores 0 through 3.

```python
{
    "type": "score",
    "instructions": "How severe?",
    "criteria": ["minor", "normal", "major", "critical"],
}
```

The returned score is a probability-weighted expectation, so `1.7` is a real
answer sitting between `normal` and `major` — see [Answers](answers.md).

## Batching

Ask everything you need about one state in a single call. Each question is
answered independently, but they share one request and one usage record:

```python
response = agent.ask(ticket, {"urgent": …, "topic": …, "severity": …})
```

## Overriding the model per call

```python
agent.ask(state, questions, model="jev-latest")
```

Without `model`, the agent's configured model is used — see
[Configuration](configuration.md).
