"""Compare the statistics file of the clean re-run (v6.0.0 code, /tmp/rr7) with the one used for the manuscript."""
import json, re
A = json.load(open("/home/claude/results/paper_stats_r6.json")); B = json.load(open("/tmp/rr7/results/paper_stats_r6.json"))
skip = re.compile(r"(sec|seconds|time|_sec$|zenodo|repo_url)", re.I)
keys = [k for k in A if not skip.search(k)]
diff = [k for k in keys if A.get(k) != B.get(k)]
missing = [k for k in keys if k not in B]
st = (f"of {len(keys):,} statistics written by the analyses, {len(keys) - len(diff):,} were identical"
      + (f"; {len(diff)} differed ({', '.join(diff[:5])})" if diff else ", and none differed apart from computing times"))
json.dump({"n": len(keys), "n_diff": len(diff), "diff": diff[:50], "missing": missing[:20], "statement": st},
          open("/home/claude/results/secondary/rerun_compare.json", "w"), indent=1)
print(st)
