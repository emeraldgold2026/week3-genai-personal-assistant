import pytest
from pydantic import ValidationError

from app.providers.base import ChatParams


def test_chat_params_defaults():
    params = ChatParams()
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k is None
    assert params.max_output_tokens == 1024
    assert params.seed is None
    assert params.stop_sequence is None


def test_chat_params_rejects_temperature_out_of_range():
    with pytest.raises(ValidationError):
        ChatParams(temperature=2.5)


def test_chat_params_rejects_top_p_out_of_range():
    with pytest.raises(ValidationError):
        ChatParams(top_p=1.5)


def test_chat_params_rejects_top_k_below_one():
    with pytest.raises(ValidationError):
        ChatParams(top_k=0)


def test_chat_params_rejects_non_positive_max_output_tokens():
    with pytest.raises(ValidationError):
        ChatParams(max_output_tokens=0)
