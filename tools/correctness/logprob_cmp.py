"""Compare the two arms on NUMBERS rather than on sampled text.

Sampled text is a terrible correctness signal for a long greedy chain: one token that
flips near a tie changes everything after it, so "the text differs" cannot distinguish a
wrong expert index from a last-bit rounding difference. Token logprobs can.

Reads the top-k logprobs of the FIRST generated token, where nothing has compounded yet.
  * agreement to ~1e-2 with the same ranking  -> a rounding-order difference, expected
    when the same weights are read from a different memory space
  * a different top token, or gaps far larger than that -> the weights being read are
    not the same weights, i.e. a real indexing bug
"""
import json
import sys
import urllib.request

LONG = ("Consider the following technical notes on distributed storage. " * 90).strip()
PROMPTS = {
    "short": "What is 17 times 23? Reply with just the number.",
    "long": LONG + "\n\nIn one sentence, what topic do these notes concern?",
}


def probe(prompt):
    body = json.dumps({"model": "glm53", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0,
                       "logprobs": True, "top_logprobs": 10}).encode()
    req = urllib.request.Request("http://127.0.0.1:8145/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    lp = d["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        sys.exit("FAIL: server returned no logprobs; cannot compare numerically")
    top = lp["content"][0]["top_logprobs"]
    return [(t["token"], t["logprob"]) for t in top]


arm = sys.argv[1]
# Probe each prompt TWICE on this one server. The within-arm spread is the noise floor
# that any between-arm difference has to clear to mean anything -- established the hard
# way: long-form text turned out not to reproduce run to run on an unchanged server.
out = {}
for k, p in PROMPTS.items():
    a, b = probe(p), probe(p)
    da, db = dict(a), dict(b)
    shared = set(da) & set(db)
    within = max((abs(da[t] - db[t]) for t in shared), default=0.0)
    out[k] = {"probe": a, "within_arm_gap": within, "top1_repeats": a[0][0] == b[0][0]}
    print(f"  {k:<6} top1={a[0][0]!r} {a[0][1]:.6f} | repeats={a[0][0] == b[0][0]} | "
          f"within-arm gap {within:.6f}")
json.dump(out, open(f"/tmp/lp_{arm}.json", "w"))
