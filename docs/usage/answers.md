# Answers

`ask` returns a [`SystemOneOutput`][system_one.schemas.SystemOneOutput]:

```python
response.model      # the model that answered
response.usage      # input_tokens, output_tokens, cost (when reported)
response.id         # provider request id, when there is one
response.answers    # {question name: answer}, one entry per question asked
```

`answers` is heterogeneous — a `noul` question yields a `NoulAnswer`, a `choice`
question a `ChoiceAnswer`, and so on. Three cached views narrow it by type so
you get a precise type without an `isinstance` dance:

```python
response.nouls["urgent"].noul
response.choices["topic"].choice
response.scores["severity"].score
```

## `NoulAnswer`

```python
answer = response.nouls["urgent"]
answer.noul         # 0.0–1.0: the probability the statement holds
answer.confidence   # max(p, 1 - p) — the probability of the reported outcome
```

`noul` is a probability, not a boolean. Threshold it where your application
wants the cut, and use `confidence` to route the uncertain middle to a human:

```python
if answer.confidence < 0.7:
    escalate()
elif answer.noul > 0.5:
    page_someone()
```

## `ChoiceAnswer`

```python
answer = response.choices["topic"]
answer.choice          # the selected label
answer.probabilities   # {label: probability}, when the provider reports them
answer.confidence
```

## `ScoreAnswer`

```python
answer = response.scores["severity"]
answer.score           # probability-weighted expectation, not an argmax
answer.legend          # {index: criterion}, when reported
answer.probabilities   # {index: probability}, when reported
answer.confidence
```

!!! note "The score is an expectation"
    With criteria `["minor", "normal", "major", "critical"]`, a score of `1.7`
    means the mass sits between `normal` (1) and `major` (2) and leans toward
    `major`. It is *not* the index of the most likely criterion — a model torn
    between `minor` and `critical` returns roughly `1.5`, which no single
    criterion claims. Check `confidence` before treating a mid-range score as a
    verdict.

## Confidence

`confidence` is whatever the provider reports; the hosted API always sends one
for `choice` and `score`. When an answer arrives with probabilities but no
confidence, it is derived from them, and every answer type puts it on the same
scale: **how certain is the value this answer reports?**

| Answer | Derived as | `1.0` | `0.0` |
| --- | --- | --- | --- |
| `noul` | `max(p, 1 - p)` | certain either way | never — a coin flip is `0.5` |
| `choice` | the probability of the selected label | the label is certain | never — `1/k` is the floor |
| `score` | `1 - 2 * sd / (k - 1)` | all mass on one level | mass split across the end levels |

`score` cannot use the choice rule, because the score is an expectation rather
than a level. It uses dispersion instead, so a distribution split between the
end levels — whose expectation lands in a valley no level claims — scores `0.0`,
while one split between neighbours scores `0.5`.

!!! warning "Unusable probabilities leave confidence unset"
    Probabilities that are not a distribution — logits, a truncated top-k, a
    vector that does not sum to `1` — leave `confidence` as `None` rather than
    producing a number. Treat `None` as "unknown", never as confident.

## Serialization

Every schema is a frozen pydantic model, so the usual methods apply. Pass
`exclude_none=True` to keep unset optional fields off the wire:

```python
response.model_dump(exclude_none=True)
response.model_dump_json(exclude_none=True)
```
