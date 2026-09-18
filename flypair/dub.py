"""Optional post-hoc "LLM dub": caption the flies' states with the Anthropic API.

Strictly read-only: takes a finished Run, summarises windows of rates/poses, asks Claude for a
short line of "dialogue" per fly per window, and returns {tick: {fly: text}} for video.render().
Never feeds back into the simulation. The video is watermarked "LLM dub — not fly output".
"""
from __future__ import annotations

import json

import numpy as np

from .record import Run

WATERMARK = "LLM dub — not fly output"
MODEL = "claude-sonnet-5"


def window_summaries(run: Run, window_ms: float = 500.0, groups=("p1", "pip10", "dnp01", "dna02_L", "dna02_R", "mdn", "vpodn", "dnp13")) -> list:
    dt = float(run.meta.get("world_dt_ms", 20))
    per = max(1, int(round(window_ms / dt)))
    out = []
    n = len(run.fly_frames(run.flies[0]))
    for start in range(0, n, per):
        w = {"tick": start, "t_s": round(start * dt / 1000, 2), "flies": {}}
        for f in run.flies:
            ff = run.fly_frames(f).iloc[start:start + per]
            d = {"x": round(float(ff.x.mean()), 1), "y": round(float(ff.y.mean()), 1),
                 "speed_mm_s": round(float(ff.speed.mean()), 1), "song": round(float(ff.song.mean()), 2),
                 "jumps": int(ff.jump.sum())}
            for g in groups:
                c = f"rate_{g}"
                if c in ff and ff[c].notna().any():
                    d[f"{g}_hz"] = round(float(ff[c].mean()), 1)
            others = [o for o in run.flies if o != f]
            if others:
                oo = run.fly_frames(others[0]).iloc[start:start + per]
                d["dist_to_other_mm"] = round(float(np.hypot(ff.x.to_numpy() - oo.x.to_numpy(), ff.y.to_numpy() - oo.y.to_numpy()).mean()), 1)
            w["flies"][f] = d
        out.append(w)
    return out


def dub(run: Run, api_key: str | None = None, window_ms: float = 500.0, model: str = MODEL, style: str = "deadpan nature documentary") -> dict:
    """Returns captions {tick: {fly: text}}. Requires `pip install anthropic` and an API key
    (Colab: userdata.get('ANTHROPIC_API_KEY'))."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    windows = window_summaries(run, window_ms)
    prompt = (
        "You are captioning a simulation of fruit-fly brains (leaky integrate-and-fire on a connectome) "
        "interacting in an arena. For each time window and each fly, write ONE short line (max 10 words) of "
        f"imagined inner monologue in a {style} style, grounded ONLY in the numbers given (song 0-1, rates in Hz, "
        "p1 = courtship drive, pip10 = song command, dnp01 = escape, vpodn = female acceptance, dnp13 = rejection). "
        "Return STRICT JSON: a list of {\"tick\": int, \"captions\": {fly_name: text}}. Nothing else.\n\n"
        + json.dumps(windows)
    )
    msg = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1] if "\n" in text else text.strip("`")
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    return {int(w["tick"]): dict(w["captions"]) for w in data}


def dubbed_video(run: Run, out, captions: dict, **kw):
    from .video import render
    return render(run, out, captions=captions, watermark=WATERMARK, **kw)
