"""Numerical noise measured on a FIXED token, not on whichever token happens to rank first.

The earlier sweep recorded the top-1 token's logprob. When the top-1 token changes
identity between samples -- which it does here, because these prompts have close
competitors -- that number compares two DIFFERENT tokens and reports their gap as
"noise". It inflated the spread and made the stack look far less stable than it is.

Tracking one token that appears in every sample's top-k gives the real perturbation,
and makes an honest between-arm comparison possible.
"""
import json
import statistics
import sys
import urllib.request

UNIT = "Consider the following technical notes on distributed storage. "
LENGTHS = {"tiny": 1, "medium": 30, "large": 90}
N = 7
TRACK = "The"


def probe(prompt):
    body = json.dumps({"model": "glm53", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0,
                       "logprobs": True, "top_logprobs": 20}).encode()
    req = urllib.request.Request("http://127.0.0.1:8145/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return {t["token"]: t["logprob"] for t in top}


arm = sys.argv[1]
out = {}
for name, rep in LENGTHS.items():
    p = (UNIT * rep).strip() + "\n\nIn one sentence, what topic do these notes concern?"
    vals = []
    for _ in range(N):
        d = probe(p)
        if TRACK in d:
            vals.append(d[TRACK])
    if len(vals) < 2:
        print(f"  {name:<7} {TRACK!r} not in top-20 often enough ({len(vals)}/{N}); skipping")
        continue
    out[name] = {"vals": vals, "mean": statistics.fmean(vals),
                 "spread": max(vals) - min(vals)}
    print(f"  {name:<7} {TRACK!r} over {len(vals)}/{N} samples: mean={statistics.fmean(vals):+.6f} "
          f"spread={max(vals)-min(vals):.6f}")
json.dump(out, open(f"/tmp/ft_{arm}.json", "w"))
