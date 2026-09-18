import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flypair.connectome.base import Connectome
from flypair.connectome.tiny import build_tiny
from flypair.groups import resolve_for


@pytest.fixture(scope="session")
def tiny():
    return build_tiny()


@pytest.fixture(scope="session")
def tiny_groups(tiny):
    return resolve_for(tiny, verbose=False)


def micro_connectome(edges, n=4, nt=None):
    """Tiny hand-made connectome: edges = [(pre, post, signed_count), ...]."""
    pre = np.array([e[0] for e in edges]); post = np.array([e[1] for e in edges])
    w = np.array([e[2] for e in edges], dtype=np.float32)
    W = sparse.csr_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    neurons = pd.DataFrame({"id": np.arange(n), "type": [f"n{i}" for i in range(n)],
                            "class": "x", "side": "L", "nt": nt or ["acetylcholine"] * n})
    return Connectome("micro", W, neurons)
