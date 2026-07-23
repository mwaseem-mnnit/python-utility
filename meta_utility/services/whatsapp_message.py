"""WhatsApp template messaging service."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from meta_utility.clients.graph_facebook_api import GraphFacebookApiClient, GraphFacebookApiError
from meta_utility.core.config import MetaConfig

logger = logging.getLogger(__name__)


class WhatsAppMessageService:
    """Build and send WhatsApp template messages."""

    def __init__(self, client: GraphFacebookApiClient, config: MetaConfig) -> None:
        self.client = client
        self.config = config

    def send_template_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Send template messages and return batch summary."""
        request_bodies = self.build_request_bodies(messages)
        responses: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for index, body in enumerate(request_bodies, start=1):
            recipient = str(body.get("to", "")).strip()
            try:
                response = self.client.post(self._messages_path(), json_payload=body)
                responses.append(response)
                logger.info("Sent WhatsApp template message #%s recipient=%s", index, recipient)
            except (GraphFacebookApiError, ValueError, TypeError) as exc:
                logger.error(
                    "Failed WhatsApp template message #%s recipient=%s error=%s",
                    index,
                    recipient,
                    exc,
                )
                errors.append({"recipient": recipient, "error": str(exc)})

        return {
            "total": len(messages),
            "sent": len(responses),
            "failed": len(errors),
            "responses": responses,
            "errors": errors,
        }

    def build_request_bodies(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build Graph API request payloads for all recipients."""
        bodies: list[dict[str, Any]] = []
        for item in messages:
            bodies.append(self._build_request_body(item))
        return bodies

    def _build_request_body(self, item: Mapping[str, Any]) -> dict[str, Any]:
        recipient = str(item.get("recipient", "")).strip()
        if not recipient:
            raise ValueError("recipient is required for WhatsApp template message")

        raw_parameters = item.get("parameters")
        if not isinstance(raw_parameters, list):
            raise TypeError("parameters must be a list")

        parameters = [self._build_text_parameter(parameter) for parameter in raw_parameters]
        return {
            "messaging_product": self.config.whatsapp_messaging_product,
            "recipient_type": self.config.whatsapp_recipient_type,
            "to": "91" + recipient,
            "type": self.config.whatsapp_type,
            "template": {
                "name": self.config.whatsapp_template_name,
                "language": {"code": self.config.whatsapp_template_language_code},
                "components": [
                    {
                        "type": self.config.whatsapp_template_component_type,
                        "parameters": parameters,
                    },
                    {
                        "type": "header",
                        "parameters": [
                            {
                                "type": "image",
                                "image": {
                                    "link": "https://static.wixstatic.com/media/8c8910_593ad2b5edf245e4b5c4f467a31dc7e8~mv2.jpg"
                                }
                            }
                        ]
                    }
                ],
            },
        }

    @staticmethod
    def _build_text_parameter(parameter: Any) -> dict[str, str]:
        if isinstance(parameter, Mapping):
            if "value" in parameter:
                value = parameter["value"]
            elif "text" in parameter:
                value = parameter["text"]
            else:
                raise ValueError("parameter object must contain value or text")
        else:
            value = parameter
        return {"type": "text", "text": str(value)}

    def _messages_path(self) -> str:
        return f"/{self.config.whatsapp_phone_number_id}/messages"

