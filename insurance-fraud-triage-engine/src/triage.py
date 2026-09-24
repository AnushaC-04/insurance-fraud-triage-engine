import os
import time
from typing import List, Literal

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class TriageAudit(BaseModel):
    risk_category: Literal[
        "Low Risk",
        "Medium Risk",
        "High Risk / Fraud Suspected"
    ]
    action_trigger: Literal[
        "Auto-Pass",
        "Adjuster Review",
        "SIU Escalation"
    ]
    supporting_evidence: List[str] = Field(
        ...,
        description="Top 3 to 5 SHAP factors justifying the risk classification."
    )
    audit_notes: str = Field(
        ...,
        description="Concise investigation narrative synthesizing risk features."
    )


class LLMTriageEngine:
    """Uses Google Gemini API to generate structured fraud triage reports."""

    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=key)

        # Primary + fallback models
        self.models = [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
        ]

    def generate_audit(
        self,
        claim_details: dict,
        fraud_probability: float,
        optimal_threshold: float,
        top_shap_features: List[dict]
    ) -> TriageAudit:

        prompt = f"""
Evaluate the following insurance claim for fraud potential.

Claim Details:
{claim_details}

ML Fraud Probability:
{fraud_probability:.4f}

Optimal Action Threshold:
{optimal_threshold:.4f}

Top SHAP Feature Attributions:
{top_shap_features}

Task:
Synthesize the quantitative ML outputs with the SHAP attributions
and produce a structured audit brief.
"""

        last_error = None

        for model_name in self.models:

            for attempt in range(2):

                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                "You are an expert Special Investigation Unit "
                                "(SIU) claims auditor."
                            ),
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=TriageAudit,
                        ),
                    )

                    return TriageAudit.model_validate_json(response.text)

                except Exception as exc:
                    last_error = exc

                    error_text = str(exc)

                    # Retry temporary Gemini availability errors
                    if "503" in error_text or "UNAVAILABLE" in error_text:
                        wait_time = 2 ** attempt
                        print(
                            f"Gemini model {model_name} unavailable. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue

                    # Do not retry other errors
                    raise

        raise RuntimeError(
            "All Gemini models are temporarily unavailable. "
            "Please try the request again later."
        ) from last_error