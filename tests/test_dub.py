def test_window_summaries_read_only(tiny):
    from flypair.dub import window_summaries, WATERMARK
    from flypair.scenario import load_scenario, run_scenario
    scn = load_scenario("male_male_intact")
    for f in scn["flies"]:
        f["connectome"] = "tiny"
    scn["duration_ms"] = 300
    run = run_scenario(scn, {"tiny": tiny}, progress=False, verbose=False)
    w = window_summaries(run, window_ms=100)
    assert len(w) == 3 and set(w[0]["flies"]) == {"A", "B"}
    assert "dist_to_other_mm" in w[0]["flies"]["A"]
    assert "not fly output" in WATERMARK
