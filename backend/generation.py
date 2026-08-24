"""Generation step: turn retrieved candidates into a natural-language explanation.

Uses the Anthropic API when ANTHROPIC_API_KEY is set; otherwise (or if the call
fails for any reason) falls back to a deterministic, evidence-grounded summary
so the system always returns a usable explanation.
"""

import os

DEFAULT_MODEL = "claude-sonnet-5"
WEAK_EVIDENCE_THRESHOLD = 0.3


def _describe_modality(has_symptoms: bool, has_image: bool) -> str:
    if has_symptoms and has_image:
        return "the described symptoms and the uploaded image"
    if has_symptoms:
        return "the described symptoms"
    return "the uploaded image"


def _fallback_explanation(candidates, has_symptoms: bool, has_image: bool) -> str:
    if not candidates:
        return (
            "No confident matches were found for the provided input. "
            "Consider adding more detail or consulting a dermatologist directly."
        )

    top = candidates[0]
    modality = _describe_modality(has_symptoms, has_image)

    if top["final_score"] < WEAK_EVIDENCE_THRESHOLD:
        lead = (
            f"Based on {modality}, no single condition stood out with strong confidence. "
            f"The closest match was {top['name']}, but the evidence is weak"
        )
    else:
        lead = f"Based on {modality}, the closest match is {top['name']}"

    parts = [
        f"{lead} (risk level: {top['risk_level'].rstrip('.')})."
        f" Recommended next step: {top['next_steps']}"
    ]

    runner_ups = [c["name"] for c in candidates[1:3]]
    if runner_ups:
        parts.append(f"Other possibilities worth ruling out: {', '.join(runner_ups)}.")

    return " ".join(parts)


def _build_prompt(candidates, symptoms, has_symptoms: bool, has_image: bool) -> str:
    modality = _describe_modality(has_symptoms, has_image)
    evidence_lines = []
    for candidate in candidates:
        evidence_lines.append(
            f"- {candidate['name']}: final_score={candidate['final_score']}, "
            f"text_score={candidate['text_score']}, image_score={candidate['image_score']}, "
            f"risk_level={candidate['risk_level']}, next_steps={candidate['next_steps']}"
        )
    evidence = "\n".join(evidence_lines)

    symptoms_line = f'Reported symptoms: "{symptoms}"' if has_symptoms else "No symptoms were described."

    return (
        "You are summarizing retrieval results from a skin-disease reference system for a patient. "
        f"The system compared {modality} against a knowledge base and image index and returned these "
        "ranked candidate conditions, most likely first:\n\n"
        f"{evidence}\n\n{symptoms_line}\n\n"
        "Write a short (3-5 sentence), cautious, plain-language explanation of what these results suggest. "
        "Refer to the top candidate(s) by name, note the retrieval evidence is not a diagnosis, and repeat "
        "the top candidate's recommended next step. Do not invent symptoms, causes, or facts not present above."
    )


def _anthropic_explanation(candidates, symptoms, has_symptoms: bool, has_image: bool) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.environ.get("GENERATION_MODEL", DEFAULT_MODEL),
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": _build_prompt(candidates, symptoms, has_symptoms, has_image),
                }
            ],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def generate_explanation(candidates, symptoms=None, has_image=False):
    """Return a natural-language explanation for the ranked candidates."""
    has_symptoms = bool(symptoms and symptoms.strip())
    if not candidates:
        return _fallback_explanation(candidates, has_symptoms, has_image)

    explanation = _anthropic_explanation(candidates, symptoms, has_symptoms, has_image)
    if explanation:
        return explanation

    return _fallback_explanation(candidates, has_symptoms, has_image)
