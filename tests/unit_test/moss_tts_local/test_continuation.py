# SPDX-License-Identifier: Apache-2.0
"""Continuation-mode coverage for MOSS-TTS Local (v1.5)."""

from __future__ import annotations

import torch
import pytest

from sglang_omni.models.moss_tts_local.request_builders import (
    build_moss_tts_local_state,
    prepare_moss_tts_local_request,
)
from sglang_omni.proto import OmniRequest, StagePayload

N_VQ = 12
DATA_URI = "data:audio/wav;base64,AAAA"


class RecordingProcessor:
    """Records the (conversations, mode) call and returns deterministic rows."""

    model_config = type("_FakeModelConfig", (), {"n_vq": N_VQ})()

    def __init__(self) -> None:
        self.calls: list[tuple[list, str]] = []

    @staticmethod
    def build_user_message(**kwargs):
        return dict(kwargs, role="user")

    @staticmethod
    def build_assistant_message(**kwargs):
        return dict(kwargs, role="assistant")

    def __call__(self, conversations, mode):
        self.calls.append((conversations, mode))
        seq = 4
        rows = torch.full((1, seq, N_VQ + 1), 1024, dtype=torch.long)
        rows[0, :, 0] = torch.arange(seq)
        rows[0, -1, 0] = 151669
        return {"input_ids": rows}


class RecordingReferenceEncoder:
    def __init__(self) -> None:
        self.codes = torch.full((4, N_VQ), 7, dtype=torch.long)
        self.encode_calls: list[str] = []
        self.encode_data_uri_calls: list[str] = []

    def encode(self, path: str) -> torch.Tensor:
        self.encode_calls.append(path)
        return self.codes

    def encode_data_uri(self, ref_audio: str) -> torch.Tensor:
        self.encode_data_uri_calls.append(ref_audio)
        return self.codes


def make_payload(
    *,
    mode: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    text: str = "hello",
) -> StagePayload:
    tts_params: dict[str, object] = {}
    if mode is not None:
        tts_params["mode"] = mode
    if ref_audio is not None:
        tts_params["ref_audio"] = ref_audio
    if ref_text is not None:
        tts_params["ref_text"] = ref_text
    metadata = {"tts_params": tts_params} if tts_params else {}
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"text": text, "references": []},
            params={},
            metadata=metadata,
        ),
        data={},
    )


def test_default_mode_is_generation() -> None:
    processor = RecordingProcessor()
    prepared = prepare_moss_tts_local_request(make_payload(), processor=processor)

    assert prepared.state.mode == "generate"
    (conversations, mode) = processor.calls[0]
    assert mode == "generation"
    assert len(conversations) == 1
    conversation = conversations[0]
    assert len(conversation) == 1
    message = conversation[0]
    assert message["role"] == "user"
    assert message["text"] == "hello"
    assert "audio_codes_list" not in message


def test_continuation_builds_two_message_conversation() -> None:
    processor = RecordingProcessor()
    encoder = RecordingReferenceEncoder()
    prepared = prepare_moss_tts_local_request(
        make_payload(
            mode="continuation",
            ref_audio=DATA_URI,
            ref_text="prefix ",
            text="world",
        ),
        processor=processor,
        reference_encoder=encoder,
    )

    assert prepared.state.mode == "continuation"
    (conversations, mode) = processor.calls[0]
    assert mode == "continuation"
    assert len(conversations) == 1
    conversation = conversations[0]
    assert len(conversation) == 2

    user_message, assistant_message = conversation
    assert user_message["role"] == "user"
    assert user_message["text"] == "prefix world"
    assert "reference" not in user_message

    assert assistant_message["role"] == "assistant"
    audio_codes_list = assistant_message["audio_codes_list"]
    assert len(audio_codes_list) == 1
    assert torch.equal(audio_codes_list[0], encoder.codes)

    # The data-URI reference is encoded through the existing data-URI encoder.
    assert encoder.encode_data_uri_calls == [DATA_URI]
    assert encoder.encode_calls == []


def test_continuation_without_ref_audio_raises() -> None:
    processor = RecordingProcessor()
    with pytest.raises(ValueError, match="ref_audio"):
        prepare_moss_tts_local_request(
            make_payload(mode="continuation"),
            processor=processor,
        )
    assert processor.calls == []


def test_voice_clone_matches_generation_with_reference() -> None:
    processor = RecordingProcessor()
    encoder = RecordingReferenceEncoder()
    prepared = prepare_moss_tts_local_request(
        make_payload(mode="voice_clone", ref_audio=DATA_URI),
        processor=processor,
        reference_encoder=encoder,
    )

    assert prepared.state.mode == "voice_clone"
    (conversations, mode) = processor.calls[0]
    assert mode == "generation"
    assert len(conversations) == 1
    conversation = conversations[0]
    assert len(conversation) == 1
    message = conversation[0]
    assert message["role"] == "user"
    assert len(message["reference"]) == 1
    assert torch.equal(message["reference"][0], encoder.codes)


def test_continuation_text_is_ref_text_plus_text() -> None:
    state = build_moss_tts_local_state(
        make_payload(
            mode="continuation",
            ref_audio=DATA_URI,
            ref_text="A transcript. ",
            text="Continue here.",
        )
    )
    assert (state.ref_text or "") + state.text == "A transcript. Continue here."


def test_build_tts_params_propagates_non_default_mode() -> None:
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import build_tts_params

    continuation = CreateSpeechRequest.model_validate(
        {"input": "hello", "mode": "continuation"}
    )
    assert build_tts_params(continuation)["mode"] == "continuation"

    default = CreateSpeechRequest.model_validate({"input": "hello"})
    assert "mode" not in build_tts_params(default)


def test_speech_request_rejects_invalid_mode() -> None:
    from pydantic import ValidationError

    from sglang_omni.serve.protocol import CreateSpeechRequest

    with pytest.raises(ValidationError):
        CreateSpeechRequest.model_validate({"input": "hello", "mode": "bogus"})
