from typing import Any

import asyncpg

# One row per candidate with denormalized text for embedding.
_CANDIDATE_PAGE_SQL = """
SELECT
  c.id,
  c.first_name,
  c.last_name,
  c.email,
  c.headline,
  c.years_of_experience,
  nat.name AS nationality,
  city.name AS city_name,
  ctry.name AS country_name,
  (
    SELECT string_agg(
      s.name || COALESCE(' (' || cs.proficiency_level || ')', ''),
      ', ' ORDER BY s.name
    )
    FROM candidate_skills cs
    JOIN skills s ON s.id = cs.skill_id
    WHERE cs.candidate_id = c.id
  ) AS skills_text,
  (
    SELECT string_agg(
      lang.name || ' (' || pl.name || ')',
      ', ' ORDER BY lang.name
    )
    FROM candidate_languages cl
    JOIN languages lang ON lang.id = cl.language_id
    JOIN proficiency_levels pl ON pl.id = cl.proficiency_level_id
    WHERE cl.candidate_id = c.id
  ) AS languages_text,
  (
    SELECT string_agg(
      wx.job_title
        || ' at ' || comp.name
        || COALESCE(' | industry: ' || NULLIF(btrim(comp.industry), ''), '')
        || ' | '
        || to_char(wx.start_date, 'YYYY-MM')
        || ' to '
        || CASE WHEN wx.is_current THEN 'present' ELSE COALESCE(to_char(wx.end_date, 'YYYY-MM'), 'unknown') END
        || COALESCE(E'\n' || left(wx.description, 1200), ''),
      E'\n---\n' ORDER BY wx.start_date DESC NULLS LAST
    )
    FROM work_experience wx
    JOIN companies comp ON comp.id = wx.company_id
    WHERE wx.candidate_id = c.id
  ) AS work_text,
  (
    SELECT string_agg(
      deg.name || ' in ' || fos.name
        || ' — ' || inst.name
        || CASE WHEN ey.graduation_year IS NOT NULL
           THEN ', graduated ' || ey.graduation_year::text ELSE '' END,
      '; ' ORDER BY ey.graduation_year DESC NULLS LAST
    )
    FROM education ey
    JOIN degrees deg ON deg.id = ey.degree_id
    JOIN fields_of_study fos ON fos.id = ey.field_of_study_id
    JOIN institutions inst ON inst.id = ey.institution_id
    WHERE ey.candidate_id = c.id
  ) AS education_text
FROM candidates c
LEFT JOIN countries nat ON nat.id = c.nationality_id
LEFT JOIN cities city ON city.id = c.city_id
LEFT JOIN countries ctry ON ctry.id = city.country_id
ORDER BY c.id
LIMIT $1 OFFSET $2
"""


def _row_to_dict(record: asyncpg.Record) -> dict[str, Any]:
    return dict(record)


async def iter_candidate_pages(
    pool: asyncpg.Pool,
    *,
    page_size: int,
    limit: int | None = None,
    start_offset: int = 0,
):
    """
    Yield pages of candidate rows as dicts. Stops after `limit` total rows if set.
    `start_offset` is the initial SQL OFFSET (for pagination into the ordered result set).
    """
    offset = start_offset
    total_yielded = 0
    while True:
        cap = page_size
        if limit is not None:
            remaining = limit - total_yielded
            if remaining <= 0:
                break
            cap = min(cap, remaining)

        async with pool.acquire() as conn:
            rows = await conn.fetch(_CANDIDATE_PAGE_SQL, cap, offset)
        if not rows:
            break
        page = [_row_to_dict(r) for r in rows]
        total_yielded += len(page)
        yield page
        offset += len(page)
        if len(rows) < cap or (limit is not None and total_yielded >= limit):
            break
