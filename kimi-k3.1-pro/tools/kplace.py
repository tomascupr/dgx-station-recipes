#!/usr/bin/env python3
"""kplace.py STATS.pt OUT.json   each MoE layer's experts ranked by how many tokens a pass of the published model
(kstream.py's stats.pt) routed to them, most used first. The serving hook cuts this ranking into its tiers."""
import json
import sys

import torch

counts = torch.load(sys.argv[1])["counts"][0]
layers = {str(i): counts[i].argsort(descending=True, stable=True).tolist()
          for i in range(counts.shape[0]) if counts[i].sum() > 0}
share = {k: [round(float(counts[int(k)][v[:n]].sum() / counts[int(k)].sum()), 4) for n in (100, 150, 250, 400, 800)]
         for k, v in layers.items()}
json.dump({"source": sys.argv[1], "layers": layers}, open(sys.argv[2], "w"))
mean = [sum(s[i] for s in share.values()) / len(share) for i in range(5)]
print(f"{len(layers)} layers; mean share of calls for the top 100/150/250/400/800 experts: {mean}")
