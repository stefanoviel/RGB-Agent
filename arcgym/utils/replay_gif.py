"""Offline replay GIF generation from ARC board logs."""
from __future__ import annotations

import argparse
from pathlib import Path

_BOARD_MARKERS = {"[INITIAL BOARD STATE]", "[POST-ACTION BOARD STATE]"}
_CELL_SCALE = 6
_FRAME_DELAY_CS = 12

_PALETTE = [
    (16, 18, 24),    # background / fallback
    (82, 86, 94),    # O
    (198, 170, 92),  # $
    (63, 128, 255),  # z
    (64, 188, 96),   # G
    (32, 32, 32),    # #
    (219, 92, 72),   # C
]

_CHAR_TO_INDEX = {
    " ": 0,
    "O": 1,
    "$": 2,
    "z": 3,
    "G": 4,
    "#": 5,
    "C": 6,
}


def _parse_board_frames(log_path: Path) -> list[list[str]]:
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
        while idx < len(lines) and lines[idx].strip():
            board.append(lines[idx].rstrip("\n"))
            idx += 1
        if board:
            frames.append(board)
    return frames


def _render_frame(board: list[str], scale: int = _CELL_SCALE) -> bytes:
    height = len(board)
    width = len(board[0]) if board else 0
    pixels = bytearray(width * scale * height * scale)
    out_width = width * scale
    for row_idx, row in enumerate(board):
        for col_idx, char in enumerate(row):
            color = _CHAR_TO_INDEX.get(char, 0)
            for sy in range(scale):
                start = (row_idx * scale + sy) * out_width + col_idx * scale
                pixels[start:start + scale] = bytes([color]) * scale
    return bytes(pixels)


def _gif_palette_bytes() -> bytes:
    raw = bytearray()
    for rgb in _PALETTE:
        raw.extend(rgb)
    while len(raw) < 256 * 3:
        raw.extend((0, 0, 0))
    return bytes(raw)


def _lzw_compress(indices: bytes, min_code_size: int = 8) -> bytes:
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    next_code = eoi_code + 1
    code_size = min_code_size + 1

    dictionary = {bytes([i]): i for i in range(clear_code)}
    codes: list[int] = [clear_code]
    prefix = b""

    for value in indices:
        candidate = prefix + bytes([value])
        if candidate in dictionary:
            prefix = candidate
            continue
        if prefix:
            codes.append(dictionary[prefix])
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            codes.append(clear_code)
            dictionary = {bytes([i]): i for i in range(clear_code)}
            next_code = eoi_code + 1
            code_size = min_code_size + 1
        prefix = bytes([value])

    if prefix:
        codes.append(dictionary[prefix])
    codes.append(eoi_code)

    output = bytearray()
    bit_buffer = 0
    bit_count = 0

    dictionary = {bytes([i]): i for i in range(clear_code)}
    next_code = eoi_code + 1
    code_size = min_code_size + 1
    prefix = b""

    code_iter = iter(codes)
    first = next(code_iter)
    stream_codes = [first]
    for value in indices:
        candidate = prefix + bytes([value])
        if candidate in dictionary:
            prefix = candidate
            continue
        if prefix:
            stream_codes.append(dictionary[prefix])
        if next_code < 4096:
            dictionary[candidate] = next_code
            next_code += 1
            if next_code == (1 << code_size) and code_size < 12:
                pass
        else:
            stream_codes.append(clear_code)
            dictionary = {bytes([i]): i for i in range(clear_code)}
            next_code = eoi_code + 1
        prefix = bytes([value])
    if prefix:
        stream_codes.append(dictionary[prefix])
    stream_codes.append(eoi_code)

    dictionary = {bytes([i]): i for i in range(clear_code)}
    next_code = eoi_code + 1
    code_size = min_code_size + 1
    prefix = b""
    emitted_clear = False
    for code in stream_codes:
        if code == clear_code:
            emitted_clear = True
            bit_buffer |= code << bit_count
            bit_count += code_size
            while bit_count >= 8:
                output.append(bit_buffer & 0xFF)
                bit_buffer >>= 8
                bit_count -= 8
            dictionary = {bytes([i]): i for i in range(clear_code)}
            next_code = eoi_code + 1
            code_size = min_code_size + 1
            prefix = b""
            continue
        bit_buffer |= code << bit_count
        bit_count += code_size
        while bit_count >= 8:
            output.append(bit_buffer & 0xFF)
            bit_buffer >>= 8
            bit_count -= 8
        if code == eoi_code:
            break
        if not emitted_clear:
            continue
    if bit_count:
        output.append(bit_buffer & 0xFF)
    return bytes(output)


def _pack_sub_blocks(payload: bytes) -> bytes:
    blocks = bytearray()
    for offset in range(0, len(payload), 255):
        chunk = payload[offset:offset + 255]
        blocks.append(len(chunk))
        blocks.extend(chunk)
    blocks.append(0)
    return bytes(blocks)


def write_gif(frames: list[bytes], width: int, height: int, output_path: Path, delay_cs: int = _FRAME_DELAY_CS) -> None:
    header = bytearray()
    header.extend(b"GIF89a")
    header.extend(width.to_bytes(2, "little"))
    header.extend(height.to_bytes(2, "little"))
    header.append(0xF7)  # global color table, 8-bit palette
    header.append(0)
    header.append(0)
    header.extend(_gif_palette_bytes())
    header.extend(b"!\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00")

    body = bytearray()
    for frame in frames:
        body.extend(b"!\xF9\x04\x04")
        body.extend(delay_cs.to_bytes(2, "little"))
        body.extend(b"\x00\x00")
        body.extend(b",")
        body.extend((0).to_bytes(2, "little"))
        body.extend((0).to_bytes(2, "little"))
        body.extend(width.to_bytes(2, "little"))
        body.extend(height.to_bytes(2, "little"))
        body.extend(b"\x00")
        body.append(8)
        body.extend(_pack_sub_blocks(_lzw_compress(frame, min_code_size=8)))

    output_path.write_bytes(bytes(header + body + b";"))


def generate_replay_gif(log_path: Path, output_path: Path | None = None) -> Path | None:
    frames = _parse_board_frames(log_path)
    if not frames:
        return None
    width = len(frames[0][0]) * _CELL_SCALE
    height = len(frames[0]) * _CELL_SCALE
    rendered = [_render_frame(frame) for frame in frames]
    dest = output_path or log_path.with_name("replay.gif")
    write_gif(rendered, width, height, dest)
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
