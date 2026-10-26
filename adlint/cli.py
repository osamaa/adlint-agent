"""Command line: python -m adlint.cli CREATIVE [--logo LOGO] [--placements a,b] [--out DIR]"""

import argparse
import json
import logging
import os
import sys

import cv2

from .placements import DEFAULT_PLACEMENTS, PLACEMENTS
from .planner import make_planner

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adlint", description="Check an ad creative before launch and prepare it per placement.")
    parser.add_argument("creative", help="image or video file")
    parser.add_argument("--logo", help="brand logo image (optional)")
    parser.add_argument("--placements", default=",".join(DEFAULT_PLACEMENTS),
                        help="comma list from: " + ", ".join(PLACEMENTS))
    parser.add_argument("--out", default="adlint_out", help="output folder")
    parser.add_argument("--planner", choices=["auto", "rules", "llm"], default="auto",
                        help="auto = LLM when OPENROUTER_API_KEY is set, else rules")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    placements = [p.strip() for p in args.placements.split(",") if p.strip()]
    unknown = [p for p in placements if p not in PLACEMENTS]
    if unknown:
        parser.error(f"unknown placement(s): {', '.join(unknown)}")
    logo = None
    if args.logo:
        logo = cv2.imread(args.logo)
        if logo is None:
            parser.error(f"cannot read logo: {args.logo}")
    planner = make_planner(args.planner)

    if args.creative.lower().endswith(VIDEO_EXTENSIONS):
        from .video import check_video

        report = check_video(args.creative, placements, logo, planner, args.out)
    else:
        image = cv2.imread(args.creative)
        if image is None:
            parser.error(f"cannot read image: {args.creative}")
        from .agent import run_image
        from .report import write_outputs

        report = write_outputs(run_image(image, placements, logo, planner), args.out, os.path.basename(args.creative))

    summary = {p["placement"]: p["status"] for p in report["placements"]}
    print(json.dumps({"out": args.out, "needs_human": report["needs_human"], "placements": summary}, indent=2))
    return 2 if report["needs_human"] else 0


if __name__ == "__main__":
    sys.exit(main())
