"""FlyWire v783 loader (female, BRAIN ONLY) using the files the Shiu repo ships.

  Completeness_783.csv       index = root_id, N = 138,639 neurons; row order = model index
  Connectivity_783.parquet   Presynaptic_Index, Postsynaptic_Index, 'Excitatory x Connectivity'
                             (already signed by Shiu's neurotransmitter predictions)
  annotations                Schlegel et al. 2024 TSV (flyconnectome/flywire_annotations),
                             joined on root_id for type/class/side. Side labels there are
                             mirrored w.r.t. Shiu's naming (their 'right' sugar GRNs are side=left).
No VNC: motor output must be read from descending neurons (DNa02, DNp01, MDN, ...)
and MN9 (a brain motor neuron, cell_type CB0701) — see groups/flywire.yaml.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .base import Connectome, normalize_side

SHIU_RAW = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
FILES = {
    "completeness": (f"{SHIU_RAW}/Completeness_783.csv", 3327347),
    "connectivity": (f"{SHIU_RAW}/Connectivity_783.parquet", 100804642),
    "annotations": ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
                    "supplemental_files/Supplemental_file1_neuron_annotations.tsv", None),
}
# Shiu et al. example.ipynb: the 21 right-labellum sugar GRNs and MN9 (for the sanity check only;
# the registry uses cell types, never IDs).
SHIU_SUGAR_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
SHIU_MN9_ID = 720575940660219265
SHIU_REFERENCE = {"sugar_150Hz_MN9_Hz": 93.3, "sugar_100Hz_MN9_Hz": 67.0,
                  "note": "from reference/Drosophila_brain_model/results/example/*.parquet, 30 trials x 1 s"}


def download_raw(raw_dir: Path) -> dict:
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, (url, size) in FILES.items():
        target = raw_dir / url.rsplit("/", 1)[-1]
        if target.exists() and (size is None or target.stat().st_size == size):
            out[key] = target; continue
        print(f"[flywire] downloading {target.name}")
        part = target.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.replace(target)
        out[key] = target
    return out


def neuron_table(completeness_csv: Path, annotations_tsv: Path) -> pd.DataFrame:
    comp = pd.read_csv(completeness_csv, index_col=0)
    ids = comp.index.to_numpy(np.int64)
    ann = pd.read_csv(annotations_tsv, sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type", "hemibrain_type",
                               "side", "top_nt", "synonyms"])
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    df = pd.DataFrame({
        "id": ids,
        "type": ann["cell_type"].fillna("").astype(str).to_numpy(),
        "class": ann["cell_class"].fillna("").astype(str).to_numpy(),
        "side": ann["side"].map(normalize_side).fillna("").to_numpy(),
        "nt": ann["top_nt"].fillna("").astype(str).to_numpy(),
        "superclass": ann["super_class"].fillna("").astype(str).to_numpy(),
        "hemibrain_type": ann["hemibrain_type"].fillna("").astype(str).to_numpy(),
        "synonyms": ann["synonyms"].fillna("").astype(str).to_numpy(),
        "completed": comp["Completed"].to_numpy() if "Completed" in comp.columns else True,
    })
    return df


def build_flywire(cache_dir: Path | str = "cache", raw_dir: Path | str | None = None) -> Connectome:
    import pyarrow.parquet as pq
    raw_dir = Path(raw_dir) if raw_dir else Path(cache_dir) / "raw" / "flywire"
    paths = download_raw(raw_dir)
    neurons = neuron_table(paths["completeness"], paths["annotations"])
    n = len(neurons)
    n_ann = int((neurons["type"] != "").sum() + (neurons["superclass"] != "").sum() > 0)
    pf = pq.ParquetFile(paths["connectivity"])
    pre_parts, post_parts, w_parts = [], [], []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        pre_parts.append(t.column(0).to_numpy().astype(np.int32))
        post_parts.append(t.column(1).to_numpy().astype(np.int32))
        w_parts.append(t.column(2).to_numpy().astype(np.float32))
    pre = np.concatenate(pre_parts); post = np.concatenate(post_parts); w = np.concatenate(w_parts)
    if pre.max() >= n or post.max() >= n:
        raise ValueError("connectivity indices exceed the completeness table; wrong file pairing?")
    W = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    missing = int((neurons["superclass"] == "").sum())
    meta = {"name": "flywire", "source": SHIU_RAW, "annotations": FILES["annotations"][0],
            "version": "783", "sex": "female", "regions": "brain only (no VNC)",
            "n_neurons": int(n), "n_edges": int(W.nnz), "neurons_without_annotation": missing,
            "sign_convention": "Shiu 'Excitatory x Connectivity' column (pre-signed)",
            "shiu_reference": SHIU_REFERENCE}
    return Connectome("flywire", W, neurons, meta)
