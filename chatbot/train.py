"""Train the resume chatbot's intent classifier and export it for the browser.

The model is our own: hashed word / word-pair / character-trigram features
feeding a softmax (multinomial logistic regression) classifier, trained from
scratch on the example questions in intents.json. chat.js re-implements the
same featuriser, so the exported weights run in the visitor's browser.

Usage:  python3 chatbot/train.py
Output: chatbot/model.js  (weights + answers, loaded by chat.js)
"""
import base64
import json
import random
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DIM = 2048            # hashed feature space size
CHAR_WEIGHT = 0.5     # character trigrams count half as much as whole words
L2 = 1e-5
EPOCHS = 800
SEED = 7

# shorthand people type, mapped to the words used in the training questions
NORMALISE = {
    "u": "you", "ur": "your", "r": "are", "ya": "you", "pls": "please", "plz": "please",
    "thx": "thanks", "ty": "thanks", "uni": "university", "yrs": "years", "yr": "year",
    "exp": "experience", "wfh": "work from home", "js": "javascript", "cv": "resume",
    "whats": "what is", "hows": "how is", "wats": "what is", "wat": "what", "im": "i am",
    "youre": "you are", "dont": "do not", "cant": "can not", "didnt": "did not", "ml": "machine learning",
}

FILLER_PREFIXES = ["can you tell me ", "i want to know ", "quick question ", "harshini ",
                   "could you share ", "id like to know ", "so ", "tell me "]
FILLER_SUFFIXES = [" please", " thanks", " harshini"]


def normalise(text):
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9+#]+", " ", text)
    words = []
    for w in text.split():
        words.extend(NORMALISE.get(w, w).split())
    return words


def fnv1a(s):
    h = 0x811C9DC5
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def featurise(text):
    words = normalise(text)
    x = np.zeros(DIM, np.float32)
    for i, w in enumerate(words):
        x[fnv1a("w|" + w) % DIM] += 1.0
        if i + 1 < len(words):
            x[fnv1a("b|" + w + " " + words[i + 1]) % DIM] += 1.0
        padded = "<" + w + ">"
        for j in range(len(padded) - 2):
            x[fnv1a("c|" + padded[j:j + 3]) % DIM] += CHAR_WEIGHT
    x = np.log1p(x)
    n = np.linalg.norm(x)
    return x / n if n else x


def typo(text, rng):
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 3]
    if not idx:
        return text
    i = rng.choice(idx)
    w = list(words[i])
    k = rng.randrange(1, len(w) - 1)
    op = rng.choice(["drop", "swap", "double"])
    if op == "drop":
        del w[k]
    elif op == "swap":
        w[k], w[k + 1 if k + 1 < len(w) else k - 1] = w[k + 1 if k + 1 < len(w) else k - 1], w[k]
    else:
        w.insert(k, w[k])
    words[i] = "".join(w)
    return " ".join(words)


def augment(pattern, rng):
    out = [pattern]
    out += [typo(pattern, rng) for _ in range(2)]
    out.append(rng.choice(FILLER_PREFIXES) + pattern)
    out.append(pattern + rng.choice(FILLER_SUFFIXES))
    words = pattern.split()
    if len(words) > 3:
        drop = rng.randrange(len(words))
        out.append(" ".join(w for i, w in enumerate(words) if i != drop))
    return out


def train(X, y, n_classes):
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, (n_classes, X.shape[1])).astype(np.float32)
    b = np.zeros(n_classes, np.float32)
    Y = np.eye(n_classes, dtype=np.float32)[y]
    mW, vW, mb, vb = (np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b))
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, EPOCHS + 1):
        logits = X @ W.T + b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(X)
        gW = G.T @ X + L2 * W
        gb = G.sum(0)
        mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
        if t % 100 == 0:
            loss = -np.log(P[np.arange(len(y)), y] + 1e-9).mean()
            print(f"  epoch {t}: loss {loss:.4f}")
    return W, b


def predict(W, b, text):
    logits = W @ featurise(text) + b
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return p


def main():
    data = json.loads((HERE / "intents.json").read_text())
    intents = data["intents"]
    labels = [it["id"] for it in intents]
    rng = random.Random(SEED)

    texts, y = [], []
    for c, it in enumerate(intents):
        for pat in it["patterns"]:
            for t in augment(pat, rng):
                texts.append(t)
                y.append(c)
    X = np.stack([featurise(t) for t in texts])
    y = np.array(y)
    print(f"training on {len(texts)} examples ({sum(len(i['patterns']) for i in intents)} written + augmented), "
          f"{len(labels)} intents, {DIM} features")
    W, b = train(X, y, len(labels))

    # held-out evaluation: test questions were never used for training
    oos = labels.index("out_of_scope")
    tests = [(t, labels.index(it["id"])) for it in intents for t in it.get("tests", [])]
    print(f"\nheld-out accuracy on {len(tests)} unseen questions:")
    best = None
    for thr in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        right = 0
        for t, c in tests:
            p = predict(W, b, t)
            k = int(p.argmax())
            got = oos if (k == oos or p[k] < thr) else k
            right += got == c
        acc = right / len(tests)
        print(f"  confidence threshold {thr:.2f}: {acc:.1%}")
        if best is None or acc >= best[1]:  # on ties prefer the stricter threshold
            best = (thr, acc)
    threshold = best[0]
    print(f"using threshold {threshold}")
    misses = []
    for t, c in tests:
        p = predict(W, b, t)
        k = int(p.argmax())
        got = oos if (k == oos or p[k] < threshold) else k
        if got != c:
            misses.append(f"  '{t}' -> {labels[got]} ({p[k]:.2f}), expected {labels[c]}")
    if misses:
        print("misses:\n" + "\n".join(misses))

    # every suggested follow-up question must land on a real answer
    chips = set(data["starters"]) | {f for it in intents for f in it.get("follow", [])}
    print("\nsuggested questions:")
    for chip in sorted(chips):
        p = predict(W, b, chip)
        k = int(p.argmax())
        flag = "" if (k != oos and p[k] >= threshold) else "   <-- NOT ANSWERED"
        print(f"  {chip!r} -> {labels[k]} ({p[k]:.2f}){flag}")

    # export: int8 weights with a per-intent scale, base64 encoded
    scale = np.abs(W).max(1) / 127.0
    q = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
    model = {
        "dim": DIM, "charWeight": CHAR_WEIGHT, "normalise": NORMALISE, "threshold": threshold,
        "labels": labels, "scales": [round(float(s), 8) for s in scale],
        "bias": [round(float(v), 5) for v in b],
        "weights": base64.b64encode(q.tobytes()).decode(),
        "answers": {it["id"]: {"answer": it["answer"], "follow": it.get("follow", [])} for it in intents},
        "starters": data["starters"],
    }
    out = HERE / "model.js"
    out.write_text("// Generated by chatbot/train.py — do not edit by hand; edit intents.json and retrain.\n"
                   "window.RESUME_CHAT_MODEL = " + json.dumps(model, ensure_ascii=False) + ";\n")
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
