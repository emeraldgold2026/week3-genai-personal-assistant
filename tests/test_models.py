from app.models import ConversationCreate, ConversationDetail, ConversationUpdate, MessageOut
from app.providers.base import ChatParams


def test_conversation_create_defaults():
    created = ConversationCreate()
    assert created.provider == "openai"
    assert created.model == "gpt-5.1"
    assert created.system_prompt == ""
    assert created.params == ChatParams()


def test_conversation_create_accepts_custom_params():
    created = ConversationCreate(provider="gemini", model="gemini-3-pro", params=ChatParams(temperature=0.2))
    assert created.provider == "gemini"
    assert created.params.temperature == 0.2


def test_conversation_update_fields_are_optional():
    update = ConversationUpdate()
    assert update.system_prompt is None
    assert update.provider is None
    assert update.model is None
    assert update.params is None


def test_conversation_detail_round_trip():
    detail = ConversationDetail(
        id="abc", title="Title", system_prompt="sys", provider="openai", model="gpt-5.1",
        params=ChatParams(), created_at="t1", updated_at="t2",
        messages=[MessageOut(id="m1", role="user", content="hi", created_at="t1")],
    )
    dumped = detail.model_dump()
    assert dumped["messages"][0]["content"] == "hi"
    assert dumped["params"]["temperature"] == 1.0
