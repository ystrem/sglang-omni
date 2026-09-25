# SPDX-License-Identifier: Apache-2.0
"""Continuation envelope for MOSS-TTS Local (v1.5).

The content-factory ComfyUI client chains a multi-chunk script by carrying a
``next_prefix`` object from every hop to the next one. This module is the
single place that speaks that wire contract, so the SGLang-Omni path can offer
the identical shape instead of an invented one:

    next_prefix = {"text": str, "audio_codes": {sr, n_vq, frames, layout,
                     dtype, encoding, data}, "tail_sec": float}

``audio_codes`` is the base64 envelope ``comfyui-unified`` ``pack_audio_codes``
produces/consumes: little-endian uint16, ``frames_x_codebooks``. The tail is
the last ``tail_sec`` worth of frames of THIS hop's own output, so a client
replaying it as the next hop's ``ref_audio`` conditions the model on the
utterance it just produced.

Everything here is msgpack-safe (bytes/str/int/float/list/dict only): these
objects travel in the terminal ``CompleteMessage``, which cannot pack a
``torch.Tensor``.
"""

from __future__ import annotations

import base64
import struct
from typing import Any, Mapping

# note (Yue Yin): the ComfyUI contract's constants describe the Delay-8B codec.
# MOSS-TTS Local is a different codec: n_vq=12 (config.json), the AR engine
# emits one audio frame per step at 12.5 frames/s (request_builders.py
# _MOSS_TTS_LOCAL_AUDIO_FRAME_RATE), and the vocoder emits 48 kHz stereo
# (MossTTSLocalState.sample_rate default). The ENVELOPE layout is what must
# match the client; the constants inside it are this model's.
MOSS_LOCAL_FRAMES_PER_SEC = 12.5
MOSS_LOCAL_LAYOUT = "frames_x_codebooks"
DEFAULT_PREFIX_TAIL_SEC = 8.0
MAX_PREFIX_TAIL_SEC = 20.0
# Cap the number of frames a single prefix may carry so a long utterance cannot
# inflate the terminal msgpack message without bound.
MAX_PREFIX_FRAMES = int(MAX_PREFIX_TAIL_SEC * MOSS_LOCAL_FRAMES_PER_SEC)


class ContinuationValidationError(ValueError):
    """Raised for continuation request/response problems."""


def resolve_prefix_tail_sec(value: Any) -> float:
    if value is None:
        return DEFAULT_PREFIX_TAIL_SEC
    else:
        pass
    try:
        tail = float(value)
    except (TypeError, ValueError) as exc:
        raise ContinuationValidationError(
            f"prefix_tail_sec must be a number: {value!r}"
        ) from exc
    if tail <= 0:
        raise ContinuationValidationError("prefix_tail_sec must be > 0")
    else:
        pass
    if tail > MAX_PREFIX_TAIL_SEC:
        raise ContinuationValidationError(
            f"prefix_tail_sec max is {MAX_PREFIX_TAIL_SEC}, got {tail}"
        )
    else:
        pass
    return tail


def frames_for_tail_sec(tail_sec: float) -> int:
    return int(float(tail_sec) * MOSS_LOCAL_FRAMES_PER_SEC)


def pack_audio_codes(
    rows: Any,
    *,
    sr: int,
    n_vq: int,
) -> dict[str, Any]:
    """Pack a ``[T, n_vq]`` tensor/iterable into the client's JSON envelope."""
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    else:
        pass
    rows = list(rows)
    if rows and len(rows[0]) != n_vq:
        raise ContinuationValidationError(
            f"audio_codes width {len(rows[0])} != n_vq {n_vq}"
        )
    else:
        pass
    raw = b"".join(
        struct.pack("<" + "H" * n_vq, *[int(x) & 0xFFFF for x in row])
        for row in rows
    )
    return {
        "sr": int(sr),
        "n_vq": int(n_vq),
        "frames": len(rows),
        "layout": MOSS_LOCAL_LAYOUT,
        "dtype": "uint16",
        "encoding": "base64",
        "data": base64.b64encode(raw).decode("ascii") if raw else "",
    }


def unpack_audio_codes(obj: Mapping[str, Any]) -> Any:
    """Decode the client envelope back into a ``[T, n_vq]`` long tensor.

    Used when a client replays a previous hop's ``audio_codes`` envelope as the
    ``prefix_audio_codes`` of the next hop.
    """
    import torch

    if not isinstance(obj, Mapping):
        raise ContinuationValidationError(
            "prefix_audio_codes must be an object", status_code=422
        )
    else:
        pass
    encoding = obj.get("encoding", "base64")
    if encoding != "base64":
        raise ContinuationValidationError(
            f"prefix_audio_codes encoding must be 'base64', got {encoding!r}"
        )
    else:
        pass
    n_vq = int(obj.get("n_vq") or 0)
    if n_vq <= 0:
        raise ContinuationValidationError("prefix_audio_codes.n_vq must be > 0")
    else:
        pass
    layout = obj.get("layout", MOSS_LOCAL_LAYOUT)
    if layout != MOSS_LOCAL_LAYOUT:
        raise ContinuationValidationError(
            f"prefix_audio_codes layout must be {MOSS_LOCAL_LAYOUT!r}, "
            f"got {layout!r}"
        )
    else:
        pass
    data = obj.get("data") or ""
    if not data:
        return torch.empty((0, n_vq), dtype=torch.long)
    else:
        pass
    try:
        raw = base64.b64decode(data)
    except Exception as exc:
        raise ContinuationValidationError(
            "prefix_audio_codes.data must be base64"
        ) from exc
    expected = int(obj.get("frames") or 0) * n_vq * 2
    if len(raw) != expected:
        raise ContinuationValidationError(
            f"prefix_audio_codes.data is {len(raw)} bytes, expected {expected} "
            f"for frames={obj.get('frames')} n_vq={n_vq}"
        )
    else:
        pass
    return torch.tensor(
        list(struct.unpack("<" + "H" * (len(raw) // 2), raw)), dtype=torch.long
    ).reshape(-1, n_vq)


def compose_text(prefix_text: str, text: str) -> str:
    return prefix_text.rstrip() + " " + text.lstrip()


def next_prefix_text(
    prefix_text_used: str,
    new_text: str,
    new_duration_s: float,
    tail_sec: float,
    *,
    epsilon: float = 1e-3,
) -> str:
    """Heuristic for ``next_prefix.text`` (no STT alignment on this path).

    Mirrors the ComfyUI rule: when the whole utterance fits inside the tail
    window, the next hop must carry the composed transcript, otherwise it would
    only get the tail's worth of words.
    """
    if float(tail_sec) + epsilon >= float(new_duration_s):
        return compose_text(prefix_text_used, new_text)
    else:
        return new_text


def build_next_prefix(
    *,
    text: str,
    ref_text: str | None,
    rows: Any,
    n_vq: int,
    tail_sec: float,
    sample_rate: int,
    frames_per_sec: float = MOSS_LOCAL_FRAMES_PER_SEC,
) -> dict[str, Any]:
    """Build the ``next_prefix`` the client carries into the following hop."""
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    else:
        pass
    rows = [list(row) for row in rows]
    frames = len(rows)
    if frames == 0:
        raise ContinuationValidationError(
            "cannot build a continuation prefix from an empty audio_codes tensor"
        )
    else:
        pass
    max_frames = frames_for_tail_sec(tail_sec)
    prefix_frames = min(frames, max_frames)
    duration_s = frames / float(frames_per_sec)
    return {
        "text": next_prefix_text(ref_text or "", text, duration_s, tail_sec),
        "audio_codes": pack_audio_codes(
            rows[frames - prefix_frames :], sr=sample_rate, n_vq=n_vq
        ),
        "tail_sec": float(tail_sec),
    }


__all__ = [
    "DEFAULT_PREFIX_TAIL_SEC",
    "MAX_PREFIX_FRAMES",
    "MAX_PREFIX_TAIL_SEC",
    "MOSS_LOCAL_FRAMES_PER_SEC",
    "MOSS_LOCAL_LAYOUT",
    "ContinuationValidationError",
    "build_next_prefix",
    "compose_text",
    "frames_for_tail_sec",
    "next_prefix_text",
    "pack_audio_codes",
    "resolve_prefix_tail_sec",
    "unpack_audio_codes",
]
