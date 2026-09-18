"""Headless 3D render: viewer/index.html driven by Playwright (Chromium) -> PNG frames -> ffmpeg MP4.

    pip install playwright && playwright install chromium   (Colab: also `playwright install-deps`)
    from flypair.video3d import render3d; render3d(run, "arena3d.mp4", camera="orbit")
"""
from __future__ import annotations

import shutil


def ffmpeg_exe():
    """System ffmpeg, else the binary bundled with imageio-ffmpeg, else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
import subprocess
import tempfile
from pathlib import Path

from . import PKG_ROOT
from .record import Run

VIEWER = PKG_ROOT / "viewer" / "index.html"


def render3d(run: Run, out: Path | str, fps: int = 30, speed: float = 1.0, width: int = 1280, height: int = 720,
             camera: str = "orbit", captions: dict | None = None, watermark: str | None = None,
             tick_stride: int | None = None, verbose: bool = True) -> Path:
    from playwright.sync_api import sync_playwright
    out = Path(out)
    dt = float(run.meta.get("world_dt_ms", 20))
    stride = tick_stride or max(1, int(round(1000.0 / fps / dt * speed)))
    n_ticks = len(run.fly_frames(run.flies[0]))
    ticks = list(range(0, n_ticks, stride))
    data = run.to_viewer_json(captions=captions, watermark=watermark)
    tmp = Path(tempfile.mkdtemp(prefix="flypair3d_"))
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(VIEWER.as_uri() + "?headless=1")
        page.wait_for_function("window.flypair !== undefined", timeout=60000)
        page.evaluate("d => window.flypair.loadRun(d)", __import__("json").loads(data))
        page.evaluate(f"window.flypair.setCamera({camera!r})")
        for i, t in enumerate(ticks):
            page.evaluate(f"window.flypair.setTick({t}); window.flypair.animate({1.0/fps}); window.flypair.render();")
            page.screenshot(path=str(tmp / f"f{i:06d}.png"))
            if verbose and i % 50 == 0:
                print(f"\r[video3d] frame {i}/{len(ticks)}", end="")
        browser.close()
    if verbose:
        print()
    exe = ffmpeg_exe()
    if exe:
        subprocess.run([exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%06d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return out
    raise RuntimeError(f"ffmpeg not found; PNG frames left in {tmp}")
