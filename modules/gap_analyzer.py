import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


# ============================================================
# DOCUMENT TYPE CONFIGURATION
# ============================================================

DOCUMENT_SECTIONS = {

    "Technical / Project Document": [
        "Project Overview",
        "System Architecture",
        "Functional Requirements",
        "Security",
        "Monitoring",
        "Logging",
        "Backup Strategy",
        "Rollback Plan",
        "Disaster Recovery",
        "Testing Strategy",
        "Deployment Strategy"
    ],

    "Requirements Document": [
        "Project Overview",
        "Business Requirements",
        "Functional Requirements",
        "Non-Functional Requirements",
        "User Requirements",
        "Security Requirements",
        "Performance Requirements",
        "Acceptance Criteria",
        "Constraints",
        "Assumptions"
    ],

    "Architecture / Design Document": [
        "System Overview",
        "Architecture",
        "Components",
        "Data Flow",
        "Technology Stack",
        "Security",
        "Scalability",
        "Monitoring",
        "Logging",
        "Deployment",
        "Disaster Recovery"
    ],

    "SOP / Process Document": [
        "Purpose",
        "Scope",
        "Roles and Responsibilities",
        "Prerequisites",
        "Procedure",
        "Exception Handling",
        "Security",
        "Monitoring",
        "Documentation",
        "Review and Maintenance"
    ],

    "Meeting / Project Report": [
        "Meeting Overview",
        "Objectives",
        "Discussion Points",
        "Decisions",
        "Action Items",
        "Risks",
        "Issues",
        "Next Steps"
    ]
}


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Initialize Gemini for knowledge gap analysis.
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
# DOCUMENT TYPE PROMPT
# ============================================================

DOCUMENT_TYPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Determine the most appropriate document type based ONLY
on the provided document content.

Allowed document types:

1. Technical / Project Document
2. Requirements Document
3. Architecture / Design Document
4. SOP / Process Document
5. Meeting / Project Report

Rules:

- Select exactly one type.
- Do not invent information.
- Choose the type that best represents the overall document.
- Return ONLY valid JSON.

Output format:

{{
    "document_type": "one of the allowed document types"
}}

DOCUMENT:
{document}
"""
        ),
        (
            "human",
            "Classify this document."
        )
    ]
)


# ============================================================
# GAP ANALYSIS PROMPT
# ============================================================

GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Your task is to evaluate whether expected sections are
present in the provided document.

You will receive:

1. A document
2. A list of expected sections

For EACH expected section, classify it as:

- Present
- Partial
- Missing

Definitions:

Present:
The document clearly contains sufficient information
about the section.

Partial:
The document contains some relevant information,
but the section is incomplete or insufficiently detailed.

Missing:
No meaningful information about the section
can be found in the document.

STRICT RULES:

1. Use ONLY the provided document.
2. Do not assume information exists.
3. Do not invent missing content.
4. Do not mark a section Present based only on
   a keyword appearing without meaningful content.
5. Return every expected section exactly once.
6. Return ONLY valid JSON.

For each section provide:

- section
- status
- evidence
- risk
- recommendation

Risk must be one of:

- Low
- Medium
- High

Output format:

[
    {{
        "section": "Security",
        "status": "Present",
        "evidence": "Brief evidence from the document",
        "risk": "Low",
        "recommendation": "No major gap identified"
    }}
]

DOCUMENT:
{document}

EXPECTED SECTIONS:
{sections}
"""
        ),
        (
            "human",
            "Analyze the document for knowledge gaps."
        )
    ]
)


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_response(response_text):
    """
    Remove Markdown code fences from Gemini output.
    """

    text = response_text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# DOCUMENT TYPE CLASSIFICATION
# ============================================================

def classify_document(text):
    """
    Classify the document into one supported category.
    """

    if not text or not text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    llm = get_llm()

    prompt = DOCUMENT_TYPE_PROMPT.invoke(
        {
            "document": text
        }
    )

    response = llm.invoke(prompt)

    cleaned_response = clean_json_response(
        response.text
    )

    try:

        result = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned invalid JSON "
            "during document classification."
        ) from e

    document_type = result.get(
        "document_type"
    )

    if document_type not in DOCUMENT_SECTIONS:

        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    return document_type


# ============================================================
# GAP ANALYSIS
# ============================================================

def analyze_gaps(
    text,
    document_type=None
):
    """
    Analyze a document for missing or incomplete
    knowledge sections.

    Returns:
        dict
    """

    if not text or not text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    # --------------------------------------------------------
    # Determine document type
    # --------------------------------------------------------

    if document_type is None:

        document_type = classify_document(
            text
        )

    if document_type not in DOCUMENT_SECTIONS:

        raise ValueError(
            f"Unsupported document type: {document_type}"
        )

    expected_sections = DOCUMENT_SECTIONS[
        document_type
    ]

    # --------------------------------------------------------
    # Generate gap analysis
    # --------------------------------------------------------

    llm = get_llm()

    prompt = GAP_ANALYSIS_PROMPT.invoke(
        {
            "document": text,
            "sections": json.dumps(
                expected_sections,
                indent=2
            )
        }
    )

    response = llm.invoke(prompt)

    cleaned_response = clean_json_response(
        response.text
    )

    try:

        analysis = json.loads(
            cleaned_response
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Gemini returned invalid JSON "
            "during gap analysis."
        ) from e

    if not isinstance(analysis, list):

        raise ValueError(
            "Invalid gap analysis format. "
            "Expected a JSON list."
        )

    # --------------------------------------------------------
    # Validate results
    # --------------------------------------------------------

    validated_results = []

    for item in analysis:

        if not isinstance(item, dict):
            continue

        section = item.get(
            "section",
            "Unknown"
        )

        status = item.get(
            "status",
            "Missing"
        )

        evidence = item.get(
            "evidence",
            "No evidence identified."
        )

        risk = item.get(
            "risk",
            "Medium"
        )

        recommendation = item.get(
            "recommendation",
            "Review and document this section."
        )

        if status not in [
            "Present",
            "Partial",
            "Missing"
        ]:
            status = "Missing"

        if risk not in [
            "Low",
            "Medium",
            "High"
        ]:
            risk = "Medium"

        validated_results.append(
            {
                "section": section,
                "status": status,
                "evidence": evidence,
                "risk": risk,
                "recommendation": recommendation
            }
        )

    # --------------------------------------------------------
    # Calculate overall risk
    # --------------------------------------------------------

    high_risk = sum(
        1
        for item in validated_results
        if item["risk"] == "High"
        and item["status"] in [
            "Missing",
            "Partial"
        ]
    )

    medium_risk = sum(
        1
        for item in validated_results
        if item["risk"] == "Medium"
        and item["status"] in [
            "Missing",
            "Partial"
        ]
    )

    if high_risk >= 2:

        overall_risk = "High"

    elif high_risk >= 1 or medium_risk >= 2:

        overall_risk = "Medium"

    else:

        overall_risk = "Low"

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    return {
        "document_type": document_type,
        "overall_risk": overall_risk,
        "sections": validated_results
    }