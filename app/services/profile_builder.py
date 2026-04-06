from typing import Any
from uuid import UUID


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Helpers: derive structured signals from the denormalized work_text string.
# The SQL aggregate produces segments of the form:
#   "Job Title at Company Name | industry: XYZ | 2020-01 to present\n<description>"
# separated by "\n---\n".
# ---------------------------------------------------------------------------

def _parse_work_segments(work_text: str) -> list[dict[str, str]]:
    """Return a list of {title, industry} dicts for each work segment."""
    segments: list[dict[str, str]] = []
    for block in work_text.split("\n---\n"):
        block = block.strip()
        if not block:
            continue
        first_line = block.split("\n")[0]
        title = ""
        industry = ""
        if " at " in first_line:
            title = first_line.split(" at ")[0].strip()
        if "| industry: " in first_line:
            raw = first_line.split("| industry: ", 1)[1]
            industry = raw.split(" | ")[0].strip()
        if title or industry:
            segments.append({"title": title, "industry": industry})
    return segments


def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


_EXECUTIVE_KW = {"chief", "ceo", "cto", "coo", "cfo", "president", "founder", "owner", "partner", "managing director"}
_VP_KW = {"vice president", "svp", "evp", "vp "}
_DIRECTOR_KW = {"director", "head of", "head,"}
_SENIOR_KW = {"senior", "sr.", "sr ", "lead", "principal", "staff engineer", "manager"}
_JUNIOR_KW = {"junior", "jr.", "associate", "entry level", "intern", "trainee", "assistant"}


def _derive_seniority(titles: list[str]) -> str:
    text = " ".join(titles).lower()
    if any(k in text for k in _EXECUTIVE_KW):
        return "Executive (C-suite / Founder)"
    if any(k in text for k in _VP_KW):
        return "VP-level"
    if any(k in text for k in _DIRECTOR_KW):
        return "Director / Head"
    if any(k in text for k in _SENIOR_KW):
        return "Senior / Manager"
    if any(k in text for k in _JUNIOR_KW):
        return "Junior / Associate"
    return "Mid-level"


# ---------------------------------------------------------------------------
# Main document builder
# ---------------------------------------------------------------------------

def build_profile_text(row: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    """
    Build a searchable expert profile document optimised for semantic similarity.

    Design principles:
    - No name: proper nouns add noise, not signal, for skill/role queries.
    - Derived summary fields (Primary roles, Industries, Seniority) are placed
      at the top so they dominate the embedding and directly match query intent.
    - Geographic background is kept but moved to the bottom so it supports
      region-based queries without drowning out functional expertise.
    - Sections are fixed-order for embedding consistency across re-ingestions.
    """
    parts: list[str] = []

    # 1. Headline — candidate's own expertise summary; high signal.
    headline = _clean(row.get("headline"))
    if headline:
        parts.append(f"Headline: {headline}")

    # 2–4. Derived fields from work history — anchor the embedding to role/function.
    work_raw = _clean(row.get("work_text"))
    segments = _parse_work_segments(work_raw) if work_raw else []
    titles = _unique_ordered([s["title"] for s in segments if s["title"]])
    industries = _unique_ordered([s["industry"] for s in segments if s["industry"]])

    if titles:
        parts.append("Primary roles: " + ", ".join(titles))
    if industries:
        parts.append("Industries: " + ", ".join(industries))
    if titles:
        parts.append(f"Seniority: {_derive_seniority(titles)}")

    # 5. Years of experience — quantified expertise depth.
    yoe = row.get("years_of_experience")
    if yoe is not None:
        parts.append(f"Years of experience: {yoe}")

    # 6. Skills — explicit, structured expertise signals.
    skills = _clean(row.get("skills_text"))
    if skills:
        parts.append(f"Skills: {skills}")

    # 7. Full work experience — rich semantic context from descriptions.
    if work_raw:
        parts.append("Work experience:\n" + work_raw)

    # 8. Education — domain background and academic specialisation.
    edu = _clean(row.get("education_text"))
    if edu:
        parts.append("Education:\n" + edu)

    # 9. Languages — useful for multilingual / regional searches.
    langs = _clean(row.get("languages_text"))
    if langs:
        parts.append(f"Languages: {langs}")

    # 10. Geographic background — kept last so region queries still resolve,
    #     but functional expertise dominates the vector representation.
    geo_parts: list[str] = []
    loc_bits = [_clean(row.get("city_name")), _clean(row.get("country_name"))]
    loc = ", ".join(b for b in loc_bits if b)
    if loc:
        geo_parts.append(loc)
    nationality = _clean(row.get("nationality"))
    if nationality:
        geo_parts.append(f"Nationality: {nationality}")
    if geo_parts:
        parts.append("Geographic background: " + " | ".join(geo_parts))

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
