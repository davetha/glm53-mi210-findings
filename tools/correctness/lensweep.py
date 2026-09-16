"""Do the two arms track each other as prompt length grows?

A single comparison cannot settle this: prefill is nondeterministic, and how much
depends on prompt length. So sweep the length, take several first-token samples per
length per arm, and ask whether the BETWEEN-arm spread looks like the WITHIN-arm
spread. If the scratch read the wrong weights, its distribution would sit apart from
the stock one at every length, including the short ones where the stock path is
perfectly reproducible.
"""
import json
import statistics
import sys
import urllib.request

UNIT = "Consider the following technical notes on distributed storage. "
LENGTHS = {"tiny": 1, "small": 8, "medium": 30, "large": 90}
N = 5


def probe(prompt):
    body = json.dumps({"model": "glm53", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0,
                       "logprobs": True, "top_logprobs": 5}).encode()
    req = urllib.request.Request("http://127.0.0.1:8145/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    c = d["choices"][0]["logprobs"]["content"][0]
    return c["token"], c["logprob"]


arm = sys.argv[1]
out = {}
for name, rep in LENGTHS.items():
    p = (UNIT * rep).strip() + "\n\nIn one sentence, what topic do these notes concern?"
    samples = [probe(p) for _ in range(N)]
    toks = {t for t, _ in samples}
    lps = [lp for _, lp in samples]
    out[name] = {"tokens": sorted(toks), "lps": lps,
                 "spread": max(lps) - min(lps), "mean": statistics.fmean(lps)}
    print(f"  {name:<7} top1={sorted(toks)}  mean={statistics.fmean(lps):+.6f}  "
          f"within-arm spread={max(lps)-min(lps):.6f}")
json.dump(out, open(f"/tmp/sweep_{arm}.json", "w"))
