"""MaleCNS v1.0 loader (Janelia FlyEM, CC-BY 4.0). Male; brain + ventral nerve cord.

URLs, column names and the node-selection rule come from fly.ai/flybrain/build.py
and drosophila-brain-mlx/compile_pack_malecns.py (both verified 2026-09-17):
  nodes  : annotation rows with non-null `superclass` (166,700 neurons), ascending bodyId
  edges  : connectome-weights table (body_pre, body_post, weight), both endpoints selected
  sign   : from the PRESYNAPTIC neuron's `consensus_nt` (fly.ai convention):
           label contains gaba|glutamate|histamine -> -1, everything else +1.
           unknown_nt="drop" instead drops edges whose source is not
           acetylcholine/gaba/glutamate (mlx convention).
Memory: the 1 GB weights feather is streamed in record batches (never fully
materialised as pandas). Peak RSS on Colab ~3 GB.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Connectome, normalize_side

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connectivity": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
EXPECTED_BYTES = {"annotations": 14483314, "neurotransmitters": 43282834, "connectivity": 1051241946}
INHIBITORY = re.compile("gaba|glutamate|histamine")
ANN_COLS = ["bodyId", "type", "flywireType", "hemibrainType", "class", "superclass", "instance",
            "somaSide", "rootSide", "synonyms", "receptorType", "dimorphism", "fruDsx"]


def download_raw(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        target = raw_dir / name
        if target.exists() and target.stat().st_size == EXPECTED_BYTES[key]:
            paths[key] = target; continue
        print(f"[malecns] downloading {name} ({EXPECTED_BYTES[key]/1e6:,.0f} MB)")
        part = target.with_suffix(".part")

        def hook(blocks, bs, total, _n=name):
            if total > 0 and blocks % 200 == 0:
                print(f"\r  {min(blocks*bs, total)/1e6:,.0f} / {total/1e6:,.0f} MB", end="")
        urllib.request.urlretrieve(f"{BUCKET}/{name}", part, hook)
        print()
        part.replace(target)
        paths[key] = target
    return paths


def neuron_table(ann_path: Path) -> pd.DataFrame:
    """Selected neurons (superclass non-null), ascending bodyId, with normalized columns."""
    import pyarrow.feather as feather
    ann = feather.read_table(ann_path, columns=ANN_COLS).to_pandas()
    ann = ann[ann["superclass"].notna() & ann["superclass"].ne("")]
    ann = ann.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)
    inst = ann["instance"].fillna("").astype(str)
    inst_side = inst.str.extract(r"_([LRM])$")[0]
    side = ann["somaSide"].fillna(ann["rootSide"]).fillna(inst_side).map(normalize_side)
    df = pd.DataFrame({
        "id": ann["bodyId"].astype(np.int64),
        "type": ann["type"].fillna("").astype(str),
        "class": ann["class"].fillna("").astype(str),
        "side": side.fillna(""),
        "nt": "",   # filled by build
        "superclass": ann["superclass"].astype(str),
        "instance": inst,
        "flywireType": ann["flywireType"].fillna("").astype(str),
        "hemibrainType": ann["hemibrainType"].fillna("").astype(str),
        "synonyms": ann["synonyms"].fillna("").astype(str),
        "receptorType": ann["receptorType"].fillna("").astype(str),
        "dimorphism": ann["dimorphism"].fillna("").astype(str),
        "fruDsx": ann["fruDsx"].fillna("").astype(str),
    })
    return df


def nt_labels(nt_path: Path, ids: np.ndarray) -> np.ndarray:
    import pyarrow.feather as feather
    nt = feather.read_table(nt_path, columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    return nt.reindex(ids).fillna("unclear").astype(str).str.lower().to_numpy()


def signs_from_nt(labels: np.ndarray, unknown_nt: str = "excitatory") -> np.ndarray:
    inh = np.array([bool(INHIBITORY.search(l)) for l in labels])
    sign = np.where(inh, -1.0, 1.0).astype(np.float32)
    if unknown_nt == "drop":
        known = np.array([l.startswith(("acetylcholine", "gaba", "glutamate", "ach", "glu")) for l in labels])
        sign[~known] = 0.0
    elif unknown_nt != "excitatory":
        raise ValueError("unknown_nt must be 'excitatory' (fly.ai) or 'drop' (mlx)")
    return sign


def build_malecns(cache_dir: Path | str = "cache", raw_dir: Path | str | None = None,
                  unknown_nt: str = "excitatory", batch_rows: int = 4_000_000) -> Connectome:
    import pyarrow.feather as feather
    raw_dir = Path(raw_dir) if raw_dir else Path(cache_dir) / "raw" / "malecns"
    paths = download_raw(raw_dir)
    neurons = neuron_table(paths["annotations"])
    ids = neurons["id"].to_numpy()
    n = len(ids)
    labels = nt_labels(paths["neurotransmitters"], ids)
    neurons["nt"] = labels
    sign = signs_from_nt(labels, unknown_nt)
    print(f"[malecns] {n:,} neurons; nt: " + ", ".join(f"{k} {v}" for k, v in
          pd.Series(labels).value_counts().items()))
    edges = feather.read_table(paths["connectivity"], columns=["body_pre", "body_post", "weight"], memory_map=True)
    pre_parts, post_parts, w_parts = [], [], []
    seen = 0; dropped_endpoint = 0; dropped_unsigned = 0
    for batch in edges.to_batches(max_chunksize=batch_rows):
        pre_id = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        post_id = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        w = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float32)
        pre = np.minimum(np.searchsorted(ids, pre_id), n - 1)
        post = np.minimum(np.searchsorted(ids, post_id), n - 1)
        ok = (ids[pre] == pre_id) & (ids[post] == post_id) & (w > 0)
        dropped_endpoint += int((~ok).sum())
        s = sign[pre]
        ok2 = ok & (s != 0)
        dropped_unsigned += int((ok & (s == 0)).sum())
        pre_parts.append(pre[ok2].astype(np.int32)); post_parts.append(post[ok2].astype(np.int32))
        w_parts.append((w[ok2] * s[ok2]).astype(np.float32))
        seen += len(pre_id)
        print(f"\r[malecns] scanned {seen:,} / {edges.num_rows:,} edge rows", end="")
    print()
    del edges
    pre = np.concatenate(pre_parts); post = np.concatenate(post_parts); w = np.concatenate(w_parts)
    del pre_parts, post_parts, w_parts
    W = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    meta = {"name": "malecns", "source": BUCKET, "license": "CC-BY 4.0 (Janelia FlyEM MaleCNS v1.0)",
            "sex": "male", "regions": "brain + VNC", "n_neurons": int(n), "n_edges": int(W.nnz),
            "sign_convention": f"presynaptic consensus_nt; gaba|glutamate|histamine -> -1; unknown_nt={unknown_nt}",
            "dropped_edges_endpoint_or_zero": int(dropped_endpoint), "dropped_edges_unsigned": int(dropped_unsigned),
            "node_rule": "superclass non-null, ascending bodyId (fly.ai / mlx)"}
    return Connectome("malecns", W, neurons, meta)
