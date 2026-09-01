# Example request payloads

Ready-to-POST bodies for `/score`. The window length must match the
loaded model, which is 50 points; query `/readyz` to confirm.

```bash
curl -s localhost:7860/score -H 'Content-Type: application/json' \
  -d @examples/step_change.json | jq
```

| File | Shape |
| --- | --- |
| `flat_window.json` | constant 85 |
| `noisy_window.json` | high variance, no level shift |
| `step_change.json` | 85 then 20 |

A level shift is the machine-temperature failure signature and should
score high; the noisy window checks the detector is not merely
reacting to variance.

Regenerate with `python scripts/make_examples.py`.
