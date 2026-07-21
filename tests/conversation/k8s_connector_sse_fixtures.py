"""Minimal Harness SSE responses for the k8s connector conversation flow."""

from __future__ import annotations

import json

_SESSION = {
    "conversation_id": "conv-1",
    "session_id": "sess-1",
    "interaction_id": "int-1",
}


def _sse(events: list[tuple[str, dict]]) -> str:
    return "\n\n".join(f"event: {name}\ndata: {json.dumps(payload)}" for name, payload in events) + "\n\n"


def k8s_connector_turn_responses() -> list[str]:
    """Six SSE bodies matching the harness-agent golden elicitation loop."""
    return [
        _sse(
            [
                ("stream_metadata", _SESSION),
                (
                    "elicitation_form",
                    {
                        "review_id": "ask-form",
                        "content": {
                            "fields": [
                                {
                                    "label": "Which authentication method should be used for the Kubernetes connector?",
                                    "options": [
                                        {"label": "Inherit from Delegate", "value": "Inherit from Delegate"},
                                        {"label": "Service Account Token", "value": "Service Account Token"},
                                    ],
                                },
                                {
                                    "label": "What scope should this connector be created at?",
                                    "options": [
                                        {"label": "Project (Recommended)", "value": "Project (Recommended)"},
                                        {"label": "Account", "value": "Account"},
                                    ],
                                },
                            ]
                        },
                    },
                ),
            ]
        ),
        _sse(
            [
                ("stream_metadata", _SESSION),
                (
                    "elicitation_free_text",
                    {
                        "review_id": "ask-name",
                        "content": {"question": "What name would you like for the Kubernetes connector?"},
                    },
                ),
            ]
        ),
        _sse(
            [
                ("stream_metadata", _SESSION),
                (
                    "elicitation_yaml",
                    {
                        "review_id": "rev-yaml-1",
                        "content": {"yaml": "connector:\n  identifier: testconnector\n"},
                        "entity_info": {"entity_type": "connector", "request_action": "CREATE_CONNECTOR"},
                    },
                ),
            ]
        ),
        _sse(
            [
                ("stream_metadata", _SESSION),
                (
                    "elicitation_free_text",
                    {
                        "review_id": "ask-delegate",
                        "content": {
                            "question": (
                                "Please provide the delegate selector tag to use for this connector "
                                "(e.g. the name/tag of your delegate)."
                            )
                        },
                    },
                ),
            ]
        ),
        _sse(
            [
                ("stream_metadata", _SESSION),
                (
                    "elicitation_yaml",
                    {
                        "review_id": "rev-yaml-2",
                        "content": {"yaml": "connector:\n  identifier: testconnector\n"},
                        "entity_info": {"entity_type": "connector", "request_action": "CREATE_CONNECTOR"},
                    },
                ),
            ]
        ),
        _sse(
            [
                ("stream_metadata", _SESSION),
                ("entity_mutation", {"resource_type": "connector", "identifier": "testconnector"}),
                ("assistant_message", {"v": "Connector created."}),
            ]
        ),
    ]
