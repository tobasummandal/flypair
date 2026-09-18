"""Controls: degree-preserving shuffled connectome, and running all controls + coupling metric."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .connectome.base import Connectome


def shuffle_edges(indptr, indices, data, seed: int = 0, max_rounds: int = 200):
    """Degree-preserving shuffle (idea from drosophila-brain-mlx shuffle_pack): every row keeps its
    edges' values (out-degree and signed counts of each source), the column targets are permuted
    across all edges (in-degree preserved), duplicates within a row repaired by swapping."""
    rng = np.random.default_rng(seed)
    e = len(indices)
    n = len(indptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    dst = indices.astype(np.int64)[rng.permutation(e)]
    rounds = 0
    for rounds in range(max_rounds + 1):
        flat = src * n + dst
        order = np.argsort(flat, kind="stable")
        key = flat[order]
        rep = order[1:][key[1:] == key[:-1]]
        if rep.size == 0:
            break
        partner = rng.integers(0, e, rep.size)
        free = ~np.isin(partner, rep)
        partner, first = np.unique(partner[free], return_index=True)
        rep = rep[free][first]
        swap = np.arange(e); swap[rep], swap[partner] = partner, rep
        dst = dst[swap]
    return dst[order].astype(indices.dtype), data[order], rounds


def shuffle_connectome(c: Connectome, seed: int = 0) -> Connectome:
    """W is [post, pre] CSR; we shuffle on W^T so each PRE neuron keeps its outgoing signed counts."""
    WT = c.W.T.tocsr(); WT.sort_indices()
    dst, data, rounds = shuffle_edges(WT.indptr, WT.indices, WT.data, seed)
    WT2 = sparse.csr_matrix((data, dst, WT.indptr), shape=WT.shape)
    W2 = WT2.T.tocsr()
    meta = dict(c.meta); meta.update({"shuffled": True, "shuffle_seed": seed, "repair_rounds": rounds,
                                      "note": "NOT the connectome: degree-preserving random rewiring control"})
    return Connectome(c.name, W2, c.neurons.copy(), meta)


def run_controls(scn: dict, connectomes: dict, live, controls=None, **kw) -> dict:
    """Run the requested controls alongside a live run. Returns {control_name: Run}."""
    from .scenario import run_scenario
    controls = controls if controls is not None else scn.get("controls", [])
    out = {"live": live}
    for ctl in controls:
        out[ctl] = run_scenario(scn, connectomes, control=ctl, playback_from=live if ctl == "playback" else None, **kw)
    return out
