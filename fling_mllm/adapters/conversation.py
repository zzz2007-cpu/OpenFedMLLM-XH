import re
from typing import Dict, List, Optional, Tuple


_IMAGE_PLACEHOLDER_RE = re.compile(r"<image(?:_\d+)?>", re.IGNORECASE)


def strip_image_placeholders(text: Optional[str]) -> str:
    normalized = "" if text is None else str(text)
    normalized = _IMAGE_PLACEHOLDER_RE.sub("", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_user_assistant_turns(conversations: List[Dict]) -> Tuple[str, str]:
    user_text = ""
    assistant_text = ""
    for turn in conversations or []:
        role = turn.get("role")
        content = str(turn.get("content", ""))
        if role == "user" and not user_text:
            user_text = content
        if role == "assistant":
            assistant_text = content
    return user_text, assistant_text


def build_qwen2_vl_messages(
    user_text: str,
    image_path: Optional[str] = None,
    assistant_text: Optional[str] = None,
) -> List[Dict]:
    user_content: List[Dict] = []
    if image_path:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": strip_image_placeholders(user_text)})

    messages: List[Dict] = [{"role": "user", "content": user_content}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": str(assistant_text)})
    return messages
