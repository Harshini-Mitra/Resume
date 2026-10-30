# Resume chatbot

A small intent-classification model, trained from scratch on Harshini's resume and her answers to common interview questions. It runs entirely in the visitor's browser: no API key, no server, and it only ever replies with approved answers.

## Files

| File | What it is |
| --- | --- |
| `intents.json` | The knowledge base: one entry per topic with example questions (`patterns`), the `answer`, suggested `follow` questions, and held-out `tests` used only to measure accuracy. |
| `train.py` | Trains the model (hashed word, word-pair and character-trigram features → softmax classifier) and writes `model.js`. Needs only Python 3 and numpy. |
| `model.js` | Generated weights and answers. Don't edit by hand. |
| `chat.js` | The chat widget. Add `<script src="chatbot/chat.js" data-theme="dashboard|rpg|avatar" defer></script>` to a page; add `data-mount="someId"` to draw it inside an element instead of as a floating bubble. |

## Changing an answer

Edit the `answer` text in `intents.json`, then retrain so `model.js` picks it up:

```bash
python3 chatbot/train.py
```

## Teaching it a new question

Add a new entry to `intents` (or more `patterns` to an existing one), retrain, and check the printed accuracy and the "suggested questions" list: every suggestion should map to a real answer. Questions the model isn't confident about get a "Sorry! I'm still in training" reply with Harshini's email.
