from typing import Any
from uuid import UUID


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_profile_text(row: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    """
    Build a single searchable document per candidate from structured DB fields.

    Sections are fixed-order so similar profiles embed consistently.
    Text is truncated at max_chars (end cut) if needed; truncation is flagged in metadata.
    """
    parts: list[str] = []
    name = f"{_clean(row.get('first_name'))} {_clean(row.get('last_name'))}".strip()
    parts.append(f"Name: {name}")
    if row.get("headline"):
        parts.append(f"Headline: {_clean(row.get('headline'))}")

    loc_bits = [
        _clean(row.get("city_name")),
        _clean(row.get("country_name")),
    ]
    loc = ", ".join(b for b in loc_bits if b)
    if loc:
        parts.append(f"Location: {loc}")
    if _clean(row.get("nationality")):
        parts.append(f"Nationality: {_clean(row.get('nationality'))}")

    yoe = row.get("years_of_experience")
    if yoe is not None:
        parts.append(f"Years of experience (overall): {yoe}")

    skills = _clean(row.get("skills_text"))
    if skills:
        parts.append(f"Skills: {skills}")

    langs = _clean(row.get("languages_text"))
    if langs:
        parts.append(f"Languages: {langs}")

    work = _clean(row.get("work_text"))
    if work:
        parts.append("Work experience:\n" + work)

    edu = _clean(row.get("education_text"))
    if edu:
        parts.append("Education:\n" + edu)

    text = "\n\n".join(parts)
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, truncated


def row_to_chroma_metadata(row: dict[str, Any], truncated: bool) -> dict[str, str | int | float | bool]:
    """Chroma metadata must be scalar types only."""    
    return {
        "candidate_id": str(row["id"]) if isinstance(row["id"], UUID) else str(row["id"]),
        "first_name": _clean(row.get("first_name"))[:256],
        "last_name": _clean(row.get("last_name"))[:256],
        "email": _clean(row.get("email"))[:256],
        "headline": _clean(row.get("headline"))[:512],
        "city": _clean(row.get("city_name"))[:128],
        "country": _clean(row.get("country_name"))[:128],
        "nationality": _clean(row.get("nationality"))[:128],
        "years_of_experience": int(row["years_of_experience"]) if row.get("years_of_experience") is not None else -1,        
        "text_truncated": truncated,
    }
