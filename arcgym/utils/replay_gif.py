"""Replay GIF generation from ARC board logs."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

_BOARD_MARKERS = {"[INITIAL BOARD STATE]", "[POST-ACTION BOARD STATE]"}
_CELL_SCALE = 6
_FRAME_DURATION_MS = 120

_COLORS = {
    " ": (16, 18, 24),
    "O": (82, 86, 94),
    "$": (198, 170, 92),
    "z": (63, 128, 255),
    "G": (64, 188, 96),
    "#": (32, 32, 32),
    "C": (219, 92, 72),
    "(": (151, 99, 214),
}


def _looks_like_board_row(line: str) -> bool:
    text = line.strip("\n")
    if not text:
        return False
    if len(text) < 8:
        return False
    allowed = set(_COLORS) | {"."}
    return set(text) <= allowed


def parse_board_frames(log_path: Path) -> list[list[str]]:
    frames: list[list[str]] = []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() not in _BOARD_MARKERS:
            idx += 1
            continue
        idx += 1
        if idx < len(lines) and lines[idx].startswith("Score:"):
            idx += 1

        board: list[str] = []
        expected_width: int | None = None
        while idx < len(lines):
            line = lines[idx].rstrip("\n")
            if not _looks_like_board_row(line):
                break
            if expected_width is None:
                expected_width = len(line)
            if len(line) != expected_width:
                break
            board.append(line)
            idx += 1

        if board:
            frames.append(board)
        else:
            idx += 1
    return frames


def render_frame(board: list[str], scale: int = _CELL_SCALE) -> Image.Image:
    height = len(board)
    width = len(board[0]) if board else 0
    image = Image.new("RGB", (width * scale, height * scale), _COLORS[" "])
    draw = ImageDraw.Draw(image)
    for row_idx, row in enumerate(board):
        for col_idx, char in enumerate(row):
            color = _COLORS.get(char, _COLORS[" "])
            x0 = col_idx * scale
            y0 = row_idx * scale
            draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=color)
    return image


def generate_replay_gif(log_path: Path, output_path: Path | None = None) -> Path | None:
    frames = parse_board_frames(log_path)
    if not frames:
        return None
    images = [render_frame(frame) for frame in frames]
    dest = output_path or log_path.with_name("replay.gif")
    images[0].save(
        dest,
        save_all=True,
        append_images=images[1:],
        duration=_FRAME_DURATION_MS,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a replay GIF from an ARC logs.txt file.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = generate_replay_gif(args.log_path, args.output)
    if output is None:
        raise SystemExit("No board frames found in log.")
    print(output)


if __name__ == "__main__":
    main()
