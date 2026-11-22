"""Stitch evaluator episode recordings (head + wrist cams) into one labelled H.264 video.

usage: python -m robospec.tools.make_montage OUT.mp4 PATH [PATH ...] [--policy scripted_smoke]
PATH is an episode dir (has manifest.json) or any parent dir; episodes are found recursively.
Labels come from the episode's events.jsonl (the action sent at each step); --policy adds
phase names when the policy module defines phase_at(t).
"""
import argparse
import importlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from robospec import policies

W, H, SIDE = 880, 480, 240
FPS = 50  # recordings are one frame per 0.02 s control step


def label(img, text, x, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def find_episodes(paths):
    eps = []
    for p in map(Path, paths):
        eps += [p] if (p / "manifest.json").exists() else sorted(m.parent for m in p.rglob("manifest.json"))
    return eps


def load_actions(ep):
    acts = {}
    events = ep / "events.jsonl"
    if events.exists():
        for line in events.read_text().splitlines():
            e = json.loads(line)
            if e.get("type") == "action":
                acts[e["step"] - 1] = np.asarray(e["values"], np.float32)  # 0-based step
    return acts


def frames(ep):
    caps = [cv2.VideoCapture(str(ep / f"{n}_rgb.mp4"))
            for n in ("first_person_camera", "left_hand_camera", "right_hand_camera")]
    while True:
        reads = [c.read() for c in caps]
        if not all(ok for ok, _ in reads):
            return
        head, wl, wr = (im for _, im in reads)
        side = np.concatenate([cv2.resize(wl, (SIDE, SIDE)), cv2.resize(wr, (SIDE, SIDE))], 0)
        yield np.concatenate([head, side], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--policy", default=None, help="registry name or module:Class, for phase names")
    args = ap.parse_args()

    phase_at = None
    if args.policy:
        module = policies.REGISTRY.get(args.policy, args.policy).split(":")[0]
        phase_at = getattr(importlib.import_module(module), "phase_at", None)

    episodes = find_episodes(args.paths)
    if not episodes:
        raise SystemExit("no episode recordings found")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "20", args.out], stdin=subprocess.PIPE)
    for ep in episodes:
        title = f"{ep.parents[1].name} / {ep.parent.name} / {ep.name}"
        acts = load_actions(ep)
        card = np.full((H, W, 3), 30, np.uint8)
        label(card, title, 12, 230, 0.6)
        label(card, "head cam | left wrist (top right) | right wrist (bottom right)", 12, 266, 0.55)
        for _ in range(FPS):
            ff.stdin.write(card.tobytes())
        for t, im in enumerate(frames(ep)):
            label(im, f"step {t:4d}  t={t * 0.02:5.2f}s", 12, 28)
            if phase_at is not None:
                label(im, phase_at(t)[0][0], 12, 56, color=(80, 255, 255))
            a = acts.get(t)
            if a is not None and a.size >= 19:
                grip = "/".join("open" if g >= 0 else "CLOSE" for g in a[14:16])
                label(im, f"base vx={a[16]:+.2f} vy={a[17]:+.2f} wz={a[18]:+.2f}  grip L/R {grip}",
                      12, 470, 0.5)
            label(im, "L wrist", 652, SIDE - 8, 0.5)
            label(im, "R wrist", 652, H - 8, 0.5)
            ff.stdin.write(im.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out} ({len(episodes)} episode(s)), ffmpeg rc={ff.returncode}")


if __name__ == "__main__":
    main()
