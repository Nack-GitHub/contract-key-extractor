#!/usr/bin/env python
"""Walk through the demo in a headless browser and record it: a video plus one screenshot per step.

The recording is the fallback for a presentation — if the laptop, the projector
or the model misbehaves on the day, the same story can be played or shown as
slides. It follows the order of ``docs/demo_script_th.md``.

Start the server first, then record:

    contract-ner serve                                  # in one terminal
    pip install playwright && python -m playwright install chromium
    python scripts/record_demo.py                       # → demo/recording/

Screenshots are PNG at the window size (1440×900 by default); the video is
WebM, and also MP4 when ``ffmpeg`` is on the path (PowerPoint and Keynote play MP4).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# step: (screenshot name, caption shown in the video, actions)
CAPTION_CSS = """
#demo-caption { position: fixed; left: 20px; bottom: 20px; z-index: 100; max-width: 760px;
  background: rgba(11,11,11,0.86); color: #fff; font: 600 17px/1.45 system-ui, sans-serif;
  padding: 12px 16px; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
#demo-caption small { display: block; font-weight: 400; font-size: 14px; color: #c3c2b7; margin-top: 2px; }
"""


def caption(page, title: str, detail: str = "") -> None:
    page.evaluate(
        """([title, detail]) => {
            let el = document.getElementById('demo-caption');
            if (!el) { el = document.createElement('div'); el.id = 'demo-caption'; document.body.append(el); }
            el.innerHTML = '';
            el.append(title);
            if (detail) { const s = document.createElement('small'); s.textContent = detail; el.append(s); }
        }""",
        [title, detail],
    )


def wait_idle(page) -> None:
    page.wait_for_timeout(250)
    page.wait_for_function("() => !document.querySelector('#loading').classList.contains('show')", timeout=60_000)
    page.wait_for_timeout(300)


def tab(page, name: str) -> None:
    page.click(f"[aria-controls=panel-{name}]")
    page.wait_for_timeout(500)


def extract(page, sample: str, gold: bool = False) -> None:
    tab(page, "extract")
    page.select_option("#sample-select", sample)
    wait_idle(page)
    page.click("#run-btn")
    wait_idle(page)
    if gold and not page.is_checked("#gold-toggle"):
        page.click("#gold-toggle")
        page.wait_for_timeout(300)


def click_cell(page, label: str) -> None:
    page.click(f'#cells .cell[data-label="{label}"][data-start]')
    page.wait_for_timeout(900)


def run(args) -> None:
    from playwright.sync_api import sync_playwright

    out = ROOT / args.out
    shots = out / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    width, height = (int(x) for x in args.size.split("x"))
    pause = args.pause
    n = 0

    def shot(name: str) -> None:
        nonlocal n
        n += 1
        page.screenshot(path=str(shots / f"{n:02d}-{name}.png"))
        page.wait_for_timeout(pause)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height}, color_scheme=args.theme, device_scale_factor=args.scale,
            record_video_dir=str(out / "video") if not args.no_video else None,
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        page.goto(args.url)
        page.wait_for_selector("#sample-select option", state="attached")
        page.add_style_tag(content=CAPTION_CSS)

        caption(page, "A contract is long; a procurement team needs six cells.",
                "Pick a contract and press Extract.")
        extract(page, "en-example")
        caption(page, "One pass: parties, dates and governing law, each with its confidence.",
                "Every highlight in the text is a span the model tagged; colour = label.")
        shot("extract-table")
        click_cell(page, "JURISDICTION")
        caption(page, "Click a cell — the page jumps to the clause it came from.",
                "Offsets are kept end to end, so every value can be checked against the source.")
        shot("click-to-source")

        extract(page, "cuad-clean", gold=True)
        caption(page, "A held-out CUAD contract with its gold table: all five cells right.",
                "Start date 2008-01-01 differs from the signing date — the effective-date split rule.")
        shot("gold-all-right")

        tab(page, "compare")
        page.select_option("#cmp-sample", "cuad-parties")
        wait_idle(page)
        page.click("#cmp-run")
        wait_idle(page)
        caption(page, "Same contract, three taggers, identical post-processing.",
                "Both baselines take “Trust” as the second party; the fine-tuned model finds “PFS Funds”.")
        shot("compare-parties")

        tab(page, "eval")
        page.wait_for_selector("#chart-models svg")
        caption(page, "Slot F1 0.855 on 47 held-out contracts — above the 0.80 “usable with review” target.",
                "Regex 0.762 · conditional random field 0.690 · fine-tuned RoBERTa 0.855")
        shot("evaluation")
        page.locator("#chart-cells").scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        caption(page, "Per cell: the gap opens on Parties and End date.",
                "Local decisions (a governing-law clause) are easy for every model.")
        shot("evaluation-per-cell")

        extract(page, "cuad-end-date", gold=True)
        page.evaluate("""() => {
            const i = [...document.querySelectorAll('#doc .gold')].find(g => g.closest('[data-label=END_DATE]'));
            (i || document.querySelector('#doc .gold')).scrollIntoView({block: 'center'});
        }""")
        page.wait_for_timeout(500)
        caption(page, "An honest error: the right term clause is tagged (outlined = gold) …",
                "… but the table picks a more confident licence clause. The fix belongs in selection, not training.")
        shot("end-date-error")

        extract(page, "cuad-gold-gap", gold=True)
        click_cell(page, "JURISDICTION")
        caption(page, "“Not in gold”: the model reads a governing-law clause CUAD did not annotate.",
                "Read the clause: it is right. Gold gaps like this make the CUAD score a floor, not a ceiling.")
        shot("gold-gap")

        extract(page, "th-gold", gold=True)
        caption(page, "Same API, same page, Thai model: one language seam.",
                "Buddhist-era dates normalised to ISO · blind Thai gold slot F1 0.903")
        shot("thai")

        page.wait_for_timeout(pause)
        video_path = Path(page.video.path()) if page.video else None
        context.close()  # the video file is complete once the context is closed
        browser.close()

    print(f"screenshots → {shots}")
    if video_path is not None:
        webm = out / "demo_walkthrough.webm"
        video_path.replace(webm)
        shutil.rmtree(out / "video", ignore_errors=True)
        print(f"video → {webm}")
        if shutil.which("ffmpeg"):
            mp4 = webm.with_suffix(".mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(mp4)], check=True)
            print(f"video → {mp4}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:8000/")
    p.add_argument("--out", default="demo/recording")
    p.add_argument("--size", default="1440x900", help="window size, WIDTHxHEIGHT")
    p.add_argument("--scale", type=float, default=1.0, help="device scale factor for the screenshots (2 = retina)")
    p.add_argument("--theme", choices=("light", "dark"), default="light", help="projectors favour light")
    p.add_argument("--pause", type=int, default=3500, help="milliseconds each step stays on screen in the video")
    p.add_argument("--no-video", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
