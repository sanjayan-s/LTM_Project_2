import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Initialize the Gemini model for action item extraction.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Please check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key
    )


# ============================================================
# PROMPT
# ============================================================

ACTION_ITEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Your task is to identify explicit action items, tasks,
follow-up activities, or required actions from the document.

STRICT RULES:

1. Use ONLY information present in the document.
2. Do not invent action items.
3. Do not infer responsibilities that are not stated.
4. Do not invent deadlines.
5. Do not invent priorities.
6. If owner, deadline, or priority is not specified,
   use "Not specified".
7. Extract only genuine actionable tasks.
8. Do not treat general statements or descriptions
   as action items.
9. Return ONLY valid JSON.
10. Return an empty JSON array if no action items exist.

Each action item must have exactly these fields:

- action
- owner
- deadline
- priority

PRIORITY RULE:

Use the priority explicitly stated in the document.

If priority is not stated, use:

"Not specified"

OUTPUT FORMAT:

[
    {{
        "action": "Task description",
        "owner": "Person or team, or Not specified",
        "deadline": "Deadline, or Not specified",
        "priority": "High, Medium, Low, or Not specified"
    }}
]

DOCUMENT:
{document}
"""
        ),
        (
            "human",
            "Extract the action items from this document."
        )
    ]
)


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_response(response_text):
    """
    Clean Gemini output before JSON parsing.
    """

    text = response_text.strip()

    # Remove Markdown code fences if Gemini returns them.
    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# ACTION ITEM EXTRACTION
# ============================================================

def extract_action_items(text):
    """
    Extract structured action items from document text.

    Returns:
        list[dict]
    """

    if not text or not text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    llm = get_llm()

    prompt = ACTION_ITEM_PROMPT.invoke(
        {
            "document": text
        }
    )

    response = llm.invoke(prompt)

    response_text = response.text

    cleaned_response = clean_json_response(
        response_text
    )

    try:

        action_items = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned an invalid JSON response."
        ) from e

    if not isinstance(action_items, list):

        raise ValueError(
            "Invalid action item format. "
            "Expected a JSON list."
        )

    # --------------------------------------------------------
    # Validate each action item
    # --------------------------------------------------------

    validated_items = []

    for item in action_items:

        if not isinstance(item, dict):
            continue

        validated_item = {
            "action": item.get(
                "action",
                "Not specified"
            ),
            "owner": item.get(
                "owner",
                "Not specified"
            ),
            "deadline": item.get(
                "deadline",
                "Not specified"
            ),
            "priority": item.get(
                "priority",
                "Not specified"
            )
        }

        # Ignore completely empty actions
        if (
            validated_item["action"]
            and
            validated_item["action"] != "Not specified"
        ):
            validated_items.append(
                validated_item
            )

    return validated_items