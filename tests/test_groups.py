import pytest

from flypair.groups import GroupResolutionError, resolve


def test_all_tiny_groups_resolve(tiny_groups):
    assert all(g.ok for g in tiny_groups.groups.values())
    assert len(tiny_groups["sugar_grn_R"]) == 12


def test_zero_match_fails_loudly(tiny):
    spec = {"groups": {"ghost": {"type": "^DoesNotExist$"}}}
    with pytest.raises(GroupResolutionError, match="ghost"):
        resolve(tiny.neurons, spec, "tiny", verbose=False)


def test_declared_unresolved_is_reported_not_fatal(tiny):
    spec = {"groups": {"gr32a": {"unresolved": "not annotated", "role": "sensory"},
                       "mn9": {"type": "^MN9$"}}}
    reg = resolve(tiny.neurons, spec, "tiny", verbose=False)
    assert not reg.has("gr32a") and reg.has("mn9")
    with pytest.raises(GroupResolutionError):
        reg["gr32a"]
    assert "UNRES" in reg.report()


def test_unknown_column_is_an_error(tiny):
    with pytest.raises(GroupResolutionError, match="not in neuron table"):
        resolve(tiny.neurons, {"groups": {"x": {"nope": "a"}}}, "tiny", verbose=False)


def test_real_group_files_parse():
    from flypair.groups import load_group_spec
    for name in ("tiny", "malecns", "flywire"):
        spec = load_group_spec(name)
        assert "sugar_grn" in spec["groups"] and "mn9" in spec["groups"]
