#!/usr/bin/env python
"""How well-supported is each held-out scene's vocabulary in the REST of the data?

For a leave-one-scene-out split, the model trains on every scene except S. This asks:
when we hold out scene S, how often do the objects / (object,part) pairs that S's
questions ask about actually occur in the training (rest) data? A scene whose objects
are RARE in the rest is "far OOD" — the model saw few examples of them.

Per held-out scene S we report, occurrence-weighted over S's own questions:
  obj_support   = median #rest-questions sharing the asked OBJECT
  pair_support  = median #rest-questions sharing the asked (OBJECT,PART) pair
  %obj_novel    = share of S's questions whose object never appears in the rest
  %pair_rare    = share of S's questions whose (obj,part) pair occurs <10x in the rest
Lower support / higher rarity ⇒ more genuinely far-OOD.
"""
import csv
import re
import statistics
from collections import Counter, defaultdict

CSV = "FG-datset/paco_questions_with_scene.csv"
PART_OBJ = re.compile(r"color of the (.+?) of the (.+?)\s*\?", re.I)
OBJ_ONLY = re.compile(r"color of the (.+?)\s*\?", re.I)


def parse(q):
    m = PART_OBJ.search(q)
    if m:
        return m.group(2).strip().lower(), m.group(1).strip().lower()
    m = OBJ_ONLY.search(q)
    if m:
        return m.group(1).strip().lower(), ""
    return "", ""


rows = []
for r in csv.DictReader(open(CSV)):
    o, p = parse(r["query"])
    rows.append((r["scene"], o, p))

scenes = sorted({s for s, _, _ in rows} - {"other", "unknown"})
results = []
for S in scenes:
    in_S = [(o, p) for s, o, p in rows if s == S]
    rest = [(o, p) for s, o, p in rows if s != S]
    obj_rest = Counter(o for o, p in rest)
    pair_rest = Counter((o, p) for o, p in rest)
    obj_sup = [obj_rest[o] for o, p in in_S]
    pair_sup = [pair_rest[(o, p)] for o, p in in_S]
    novel_obj = sum(obj_rest[o] == 0 for o, p in in_S) / len(in_S)
    rare_pair = sum(pair_rest[(o, p)] < 10 for o, p in in_S) / len(in_S)
    results.append((S, len(in_S), len(rest),
                    statistics.median(obj_sup), statistics.mean(obj_sup),
                    statistics.median(pair_sup), statistics.mean(pair_sup),
                    100 * novel_obj, 100 * rare_pair))

# rank by median pair support ascending (most far-OOD first)
results.sort(key=lambda r: r[5])
print(f"{'scene':13s} {'qS':>5s} {'objMed':>7s} {'objMean':>8s} "
      f"{'pairMed':>8s} {'pairMean':>9s} {'%objNovel':>10s} {'%pairRare<10':>13s}")
for S, qs, _, om, oa, pm, pa, no, rp in results:
    print(f"{S:13s} {qs:5d} {om:7.0f} {oa:8.0f} {pm:8.0f} {pa:9.0f} "
          f"{no:9.1f}% {rp:12.1f}%")

# spotlight: sports' actual objects and their rest-support
print("\n=== sports: objects asked about, count-in-sports vs count-in-rest ===")
sp = [(o, p) for s, o, p in rows if s == "sports"]
rest_sp = Counter(o for s, o, p in rows if s != "sports")
in_sp = Counter(o for o, p in sp)
for o, c in in_sp.most_common(15):
    print(f"  {o:18s} in_sports={c:4d}  in_rest={rest_sp[o]:5d}  "
          f"rest_rate={100*rest_sp[o]/sum(rest_sp.values()):.2f}%")
