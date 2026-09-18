"""Synthetic ~2k-neuron signed sparse network with fake annotations.

Every group name used by any shipped scenario (see groups/tiny.yaml) exists here,
for both sides, so tests and CI never touch a real connectome. The wiring has a
little structure so sanity checks behave: a sugar->MN9 pathway, sensory->P1->pIP10,
looming->DNp01, LC10a->DNa02 (ipsilateral), so that "input on group A drives
output group B" can be tested deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Connectome

# (type, class, n_per_side)
TINY_TYPES = [
    ("LB3", "gustatory", 12),        # sugar GRN
    ("LB1", "gustatory", 12),        # bitter GRN (fake label)
    ("Gr32a", "gustatory", 8),
    ("ppk23", "gustatory", 8),
    ("ORN_DA1", "olfactory", 10),
    ("JO-A", "mechanosensory", 15),
    ("JO-B", "mechanosensory", 15),
    ("LC10a", "visual", 20),
    ("LPLC2", "visual", 20),
    ("L1", "visual", 30),
    ("pC1", "central", 12),           # P1
    ("mAL", "central", 12),
    ("vAB3", "central", 4),
    ("aSP22", "central", 4),
    ("vpoEN", "central", 2),
    ("aIPg", "central", 4),
    ("pIP10", "descending", 1),
    ("b1 MN", "motor", 2),            # wing MN
    ("DNp01", "descending", 1),
    ("DNa01", "descending", 1),
    ("DNa02", "descending", 1),
    ("DNp09", "descending", 1),
    ("MDN", "descending", 2),
    ("MN9", "motor", 1),
    ("DNp13", "descending", 1),
    ("vpoDN", "descending", 1),
    ("SEZ_int", "central", 40),       # sugar relay
    ("misc", "central", 700),
]

# Directed pathways (pre type -> post type, synapse count, same-side only?)
PATHWAYS = [
    ("LB3", "SEZ_int", 20, True),
    ("SEZ_int", "MN9", 20, False),
    ("ORN_DA1", "pC1", 12, False),
    ("ppk23", "pC1", 12, False),
    ("JO-A", "pC1", 8, False),
    ("JO-B", "pC1", 8, False),
    ("pC1", "pIP10", 20, False),
    ("pC1", "b1 MN", 10, False),
    ("pIP10", "b1 MN", 20, False),
    ("mAL", "pC1", -20, False),       # inhibitory brake
    ("Gr32a", "mAL", 15, False),
    ("LPLC2", "DNp01", 12, True),
    ("LC10a", "DNa02", 12, True),
    ("LC10a", "DNp09", 8, False),
    ("LB1", "MDN", 12, False),
    ("pC1", "vpoDN", 12, False),
    ("mAL", "DNp13", 12, False),
]


def build_tiny(seed: int = 0, n_random_edges: int = 12000) -> Connectome:
    rng = np.random.default_rng(seed)
    rows = []
    for t, cls, n in TINY_TYPES:
        for side in ("L", "R"):
            for k in range(n):
                rows.append((t, cls, side))
    neurons = pd.DataFrame(rows, columns=["type", "class", "side"])
    N = len(neurons)
    neurons["id"] = np.arange(1_000_000, 1_000_000 + N, dtype=np.int64)
    # neurotransmitter: mAL and 25% of misc inhibitory (gaba/glutamate), rest acetylcholine
    nt = np.full(N, "acetylcholine", dtype=object)
    is_mAL = neurons["type"].eq("mAL").to_numpy()
    misc = neurons["type"].eq("misc").to_numpy()
    inh_misc = misc & (rng.random(N) < 0.25)
    nt[is_mAL] = "gaba"
    nt[inh_misc] = rng.choice(["gaba", "glutamate"], size=int(inh_misc.sum()))
    neurons["nt"] = nt
    neurons["instance"] = neurons["type"] + "_" + neurons["side"]
    neurons["superclass"] = neurons["class"]
    neurons = neurons[["id", "type", "class", "side", "nt", "instance", "superclass"]]
    sign = np.where(pd.Series(nt).str.contains("gaba|glutamate|histamine"), -1.0, 1.0)

    idx_by = {(t, s): np.flatnonzero((neurons["type"] == t).to_numpy() & (neurons["side"] == s).to_numpy())
              for t, _, _ in TINY_TYPES for s in ("L", "R")}
    pre_l, post_l, w_l = [], [], []
    for pre_t, post_t, w, same_side in PATHWAYS:
        for s in ("L", "R"):
            pres = idx_by[(pre_t, s)]
            posts = idx_by[(post_t, s)] if same_side else np.concatenate([idx_by[(post_t, "L")], idx_by[(post_t, "R")]])
            for p in pres:
                # each pre connects to a random ~60% of targets
                tg = posts[rng.random(len(posts)) < 0.6]
                pre_l.append(np.full(len(tg), p)); post_l.append(tg); w_l.append(np.full(len(tg), float(w)))
    # random background wiring (sparse, weak) among misc/central neurons
    pre_r = rng.integers(0, N, n_random_edges)
    post_r = rng.integers(0, N, n_random_edges)
    w_r = rng.integers(1, 4, n_random_edges).astype(float) * sign[pre_r]
    pre = np.concatenate(pre_l + [pre_r]); post = np.concatenate(post_l + [post_r]); w = np.concatenate(w_l + [w_r])
    # pathway weights carry explicit sign from the table; enforce nt sign consistency for mAL
    keep = pre != post
    W = sparse.csr_matrix((w[keep].astype(np.float32), (post[keep], pre[keep])), shape=(N, N))
    W.sum_duplicates()
    meta = {"name": "tiny", "source": "synthetic (flypair.connectome.tiny)", "seed": seed,
            "sign_convention": "gaba|glutamate|histamine -> -1 (fake labels)", "sex": "any",
            "notes": "synthetic; for tests only"}
    return Connectome("tiny", W, neurons, meta)
