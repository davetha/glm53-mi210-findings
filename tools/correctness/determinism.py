"""Is this prompt even reproducible at temperature 0 on ONE server?

The equivalence test compares two servers. That only means something if a single
server repeats itself, so run the same prompt several times against the CURRENT
process and see whether it agrees with itself before blaming any code change.
"""
import json
import sys
import urllib.request

LONG = ("Consider the following technical notes on distributed storage. " * 90).strip()
PROMPT = LONG + "\n\nIn one sentence, what topic do these notes concern?"
N = 3


def ask():
    body = json.dumps({"model": "glm53", "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 1500, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8145/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    t = (m.get("reasoning_content") or m.get("reasoning") or "") + (m.get("content") or "")
    return t, d["usage"]["completion_tokens"], d["choices"][0].get("finish_reason")


runs = []
for i in range(N):
    t, tok, fin = ask()
    runs.append(t)
    print(f"  run{i+1}: {len(t):>6} chars, {tok:>4} tokens, finish={fin}  first60={t[:60]!r}")

same = all(r == runs[0] for r in runs)
print("SELF-CONSISTENT" if same else "NOT DETERMINISTIC - the same server disagrees with itself")
sys.exit(0 if same else 2)
