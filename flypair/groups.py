"""Neuron group registry: groups/<connectome>.yaml -> neuron indices.

A group is a set of regex filters on annotation columns of the neuron table
(ANDed), plus an optional `side` (L/R/M). Body IDs are never hardcoded.

groups:
  sugar_grn:
    type: "^LB3[bc]$"        # regex on the `type` column
    side: R                  # optional; normalized L/R/M
    role: sensory            # sensory | internal | output (documentation only)
    split_side: true         # also register sugar_grn_L / sugar_grn_R
    desc: "..."
  vAB3:
    synonyms: "vAB3"         # any other column name works the same way
  Gr32a:
    unresolved: "no Gr32a label in this dataset"   # documented absence: reported, not fatal

resolve() FAILS LOUDLY (GroupResolutionError) for any group that matches 0
neurons unless the group is declared `unresolved`.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import GROUPS_DIR
from .connectome.base import normalize_side

META_KEYS = {"side", "role", "split_side", "desc", "unresolved", "source"}


class GroupResolutionError(RuntimeError):
    pass


@dataclass
class ResolvedGroup:
    name: str
    query: dict
    indices: np.ndarray
    examples: list
    role: str = ""
    unresolved_reason: str | None = None

    @property
    def n(self) -> int:
        return int(self.indices.size)

    @property
    def ok(self) -> bool:
        return self.n > 0


@dataclass
class GroupRegistry:
    connectome_name: str
    groups: dict = field(default_factory=dict)      # name -> ResolvedGroup
    spec: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        g = self.groups.get(name)
        if g is None:
            raise KeyError(f"group {name!r} not in registry for {self.connectome_name}; "
                           f"known: {sorted(self.groups)}")
        if not g.ok:
            raise GroupResolutionError(f"group {name!r} is unresolved on {self.connectome_name}: {g.unresolved_reason}")
        return g.indices

    def __contains__(self, name: str) -> bool:
        return name in self.groups and self.groups[name].ok

    def has(self, name: str) -> bool:
        return name in self

    def names(self, only_ok: bool = True):
        return [k for k, v in self.groups.items() if v.ok or not only_ok]

    def report(self) -> str:
        lines = [f"Group resolution report — connectome '{self.connectome_name}'",
                 f"{'group':28s} {'n':>6s}  {'role':9s} query -> example types"]
        for name, g in self.groups.items():
            q = ", ".join(f"{k}={v!r}" for k, v in g.query.items())
            if g.ok:
                lines.append(f"{name:28s} {g.n:6d}  {g.role:9s} {q} -> {g.examples}")
            else:
                lines.append(f"{name:28s} {'UNRES':>6s}  {g.role:9s} {q} -> {g.unresolved_reason}")
        n_ok = sum(g.ok for g in self.groups.values())
        lines.append(f"{n_ok}/{len(self.groups)} groups resolved")
        return "\n".join(lines)


def load_group_spec(connectome_name: str, path: Path | str | None = None) -> dict:
    p = Path(path) if path else GROUPS_DIR / f"{connectome_name}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no group file for connectome {connectome_name!r}: {p}")
    spec = yaml.safe_load(p.read_text())
    if not isinstance(spec, dict) or "groups" not in spec:
        raise ValueError(f"{p}: expected a top-level 'groups:' mapping")
    return spec


def _side_col(neurons: pd.DataFrame) -> pd.Series:
    return neurons["side"].map(normalize_side)


def _match(neurons: pd.DataFrame, q: dict, side_series: pd.Series) -> np.ndarray:
    mask = np.ones(len(neurons), dtype=bool)
    for col, pat in q.items():
        if col in META_KEYS:
            continue
        if col not in neurons.columns:
            raise GroupResolutionError(f"query column {col!r} not in neuron table; columns: {list(neurons.columns)}")
        s = neurons[col].astype("string").fillna("")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mask &= s.str.contains(str(pat), regex=True, na=False).to_numpy()
    if q.get("side"):
        mask &= (side_series == normalize_side(q["side"])).to_numpy()
    return mask


def resolve(neurons: pd.DataFrame, spec: dict, connectome_name: str = "?",
            strict: bool = True, verbose: bool = True) -> GroupRegistry:
    reg = GroupRegistry(connectome_name, spec=spec)
    side_series = _side_col(neurons)
    failures = []
    for name, q in spec["groups"].items():
        if not isinstance(q, dict):
            raise ValueError(f"group {name!r}: expected a mapping of column -> regex, got {q!r}")
        role = q.get("role", "")
        if q.get("unresolved"):
            reg.groups[name] = ResolvedGroup(name, {k: v for k, v in q.items() if k not in META_KEYS},
                                             np.zeros(0, dtype=np.int64), [], role, str(q["unresolved"]))
            continue
        query = {k: v for k, v in q.items() if k not in META_KEYS or k == "side"}
        variants = [(name, q.get("side"))]
        if q.get("split_side"):
            variants += [(f"{name}_L", "L"), (f"{name}_R", "R")]
        for vname, side in variants:
            qq = dict(query)
            if side:
                qq["side"] = side
            elif "side" in qq:
                del qq["side"]
            mask = _match(neurons, qq, side_series)
            idx = np.flatnonzero(mask).astype(np.int64)
            ex = sorted(neurons.loc[mask, "type"].astype(str).unique().tolist())[:6]
            g = ResolvedGroup(vname, qq, idx, ex, role)
            if idx.size == 0:
                g.unresolved_reason = "0 matches"
                failures.append(vname)
            reg.groups[vname] = g
    if verbose:
        print(reg.report())
    if failures and strict:
        raise GroupResolutionError(
            f"{len(failures)} group(s) matched 0 neurons on {connectome_name}: {failures}. "
            f"Fix groups/{connectome_name}.yaml or mark them `unresolved: <reason>`.")
    return reg


def resolve_for(connectome, path: Path | str | None = None, strict: bool = True, verbose: bool = True) -> GroupRegistry:
    spec = load_group_spec(connectome.name, path)
    return resolve(connectome.neurons, spec, connectome.name, strict=strict, verbose=verbose)
