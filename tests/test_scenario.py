import copy

import numpy as np
import pytest

from flypair.scenario import ScenarioError, load_scenario, run_scenario, validate


def tiny_scn(name, duration=400):
    scn = load_scenario(name)
    for f in scn["flies"]:
        if f.get("connectome"):
            f["connectome"] = "tiny"
    scn["duration_ms"] = duration
    return scn


def test_all_shipped_scenarios_validate():
    for n in ("male_male_intact", "male_male_brake_removed", "courtship_chain", "male_female",
              "clone_mirror", "custom_template"):
        scn = load_scenario(n)
        assert scn["flies"] and scn["channels"]


def test_clone_mirror_symmetry(tiny):
    scn = tiny_scn("clone_mirror", 600)
    run = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    a, b = run.fly_frames("A"), run.fly_frames("B")
    assert np.allclose(a.x.to_numpy(), -b.x.to_numpy(), atol=1e-6)
    assert np.allclose(a.y.to_numpy(), -b.y.to_numpy(), atol=1e-6)
    assert np.allclose((a.heading.to_numpy() - b.heading.to_numpy() - 180) % 360, 0, atol=1e-6) or \
        np.allclose((a.heading.to_numpy() - b.heading.to_numpy() - 180) % 360, 360, atol=1e-6)
    for c in run.rate_columns():
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), c
    assert a.x.abs().sum() > 0     # the flies actually moved


def test_gain_zero_equals_open_loop(tiny):
    scn = tiny_scn("male_male_intact", 400)
    scn_g0 = copy.deepcopy(scn)
    for ch in scn_g0["channels"]:
        ch["gain"] = 0.0
    r_open = run_scenario(scn, {"tiny": tiny}, control="open_loop", progress=False, verbose=False)
    r_g0 = run_scenario(scn_g0, {"tiny": tiny}, progress=False, verbose=False)
    cols = ["x", "y", "heading", "song"] + r_open.rate_columns()
    for f in ("A", "B"):
        for c in cols:
            assert np.array_equal(r_open.fly_frames(f)[c].to_numpy(), r_g0.fly_frames(f)[c].to_numpy()), (f, c)
    assert not any(c.startswith("drive_") for c in r_open.frames.columns)


def test_channels_change_the_run(tiny):
    scn = tiny_scn("male_male_intact", 400)
    live = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    open_ = run_scenario(scn, {"tiny": tiny}, control="open_loop", progress=False, verbose=False)
    assert any(c.startswith("drive_") for c in live.frames.columns)
    diff = sum(not np.array_equal(live.fly_frames("A")[c], open_.fly_frames("A")[c]) for c in live.rate_columns())
    assert diff > 0


def test_silence_in_scenario_zeroes_group(tiny):
    scn = tiny_scn("male_male_brake_removed", 400)
    # give both flies constant P1 drive; brake (mAL) silenced -> mal rate must be 0
    for f in scn["flies"]:
        f["activate"] = {"gr32a": 100}
    run = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    assert run.frames["rate_mal"].max() == 0.0
    scn2 = tiny_scn("male_male_intact", 400)
    for f in scn2["flies"]:
        f["activate"] = {"gr32a": 100}
    run2 = run_scenario(scn2, {"tiny": tiny}, progress=False, verbose=False)
    assert run2.frames["rate_mal"].max() > 0.0


def test_playback_control_runs(tiny):
    scn = tiny_scn("male_male_intact", 300)
    live = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    pb = run_scenario(scn, {"tiny": tiny}, control="playback", playback_from=live, progress=False, verbose=False)
    a_live, a_pb = live.fly_frames("A"), pb.fly_frames("A")
    assert np.allclose(a_live.x, a_pb.x) and np.allclose(a_live.song, a_pb.song)
    assert a_pb["rate_p1"].isna().all() or "rate_p1" not in a_pb.columns  # A had no brain in playback


def test_shuffled_control_preserves_degrees(tiny):
    from flypair.controls import shuffle_connectome
    s = shuffle_connectome(tiny, seed=1)
    W, W2 = tiny.W.T.tocsr(), s.W.T.tocsr()
    assert np.array_equal(np.diff(W.indptr), np.diff(W2.indptr))              # out-degree per pre neuron
    assert np.array_equal(np.sort(W.data), np.sort(W2.data))                   # same signed counts
    assert np.array_equal(np.bincount(W.indices, minlength=tiny.n), np.bincount(W2.indices, minlength=tiny.n))  # in-degree
    assert not np.array_equal(W.indices, W2.indices)
    assert s.meta["shuffled"] is True


def test_scripted_source_without_brain(tiny):
    scn = tiny_scn("male_male_intact", 200)
    scn["flies"][1] = {"name": "B", "source": "scripted", "sex": "female", "pose": {"x": 3, "y": 0, "heading": 180},
                       "timeline": {2: {"song": 0.8}}}
    run = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    b = run.fly_frames("B")
    assert b.song.iloc[0] == 0 and b.song.iloc[-1] == 0.8
    assert np.allclose(b.x, 3.0)   # stationary
    assert run.fly_frames("A")["drive_jo_ab_L"].fillna(0).sum() + run.fly_frames("A")["drive_jo_ab_R"].fillna(0).sum() > 0


# ---- schema errors --------------------------------------------------------
def _bad(doc, match):
    with pytest.raises(ScenarioError, match=match):
        validate(doc, "t.yaml")


def test_schema_errors_are_helpful():
    base = {"flies": [{"name": "A", "connectome": "tiny"}]}
    _bad({**base, "bogus": 1}, "unknown top-level keys")
    _bad({"flies": []}, "non-empty list")
    _bad({"flies": [{"name": "A", "connectome": "mouse"}]}, "connectome must be one of")
    _bad({"flies": [{"name": "A", "connectome": "tiny", "sex": "yes"}]}, "sex must be")
    _bad({"flies": [{"name": "A", "connectome": "tiny"}, {"name": "A", "connectome": "tiny"}]}, "duplicate fly name")
    _bad({**base, "duration_ms": -5}, "duration_ms must be")
    _bad({**base, "world_dt_ms": 0.15}, "multiple of the brain dt")
    _bad({**base, "controls": ["magic"]}, "unknown control")
    _bad({**base, "channels": [{"source": "song_intensity"}]}, "missing required key 'target'")
    _bad({**base, "channels": [{"source": "telepathy", "target": "jo_ab"}]}, "source 'telepathy'")
    _bad({**base, "channels": [{"source": "song_intensity", "target": "jo_ab", "falloff": {"type": "magic"}}]}, "falloff.type")
    _bad({"flies": [{"name": "A", "connectome": "tiny", "activate": {"p1": -1}}]}, "activate must map")
    _bad({"flies": [{"name": "A", "connectome": "tiny", "pose": {"z": 1}}]}, "pose must be")
    _bad({**base, "on_unresolved": "ignore"}, "on_unresolved")
    ok = validate(base, "t.yaml")
    assert ok["channels"] and ok["duration_ms"] == 1000
