"""Stage 2: Claude reads the job against your resumes and a written rubric,
and returns a structured assessment via a forced tool call.
"""
from __future__ import annotations

import logging

from .models import Job

log = logging.getLogger(__name__)

# These models return a 400 on forced tool_choice ("tool"/"any"); ask with "auto" instead.
_NO_FORCED_TOOL = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1")


def build_tool(variants: list[str]) -> dict:
    return {
        "name": "record_fit",
        "description": "Record the fit assessment for this job posting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fit_score": {"type": "integer", "minimum": 0, "maximum": 10,
                              "description": "Calibrated per the rubric anchors."},
                "recommend": {"type": "string", "enum": ["apply", "maybe", "skip"]},
                "resume_variant": {"type": "string", "enum": variants},
                "strongest_evidence": {"type": "string",
                    "description": "The single most relevant thing on the resume for this role, stated concretely."},
                "gaps": {"type": "array", "items": {"type": "string"},
                    "description": "Stated requirements the resume does not show evidence for."},
                "dealbreakers": {"type": "array", "items": {"type": "string"},
                    "description": "Hard requirements clearly not met (degree, clearance, years, citizenship, sector)."},
                "pitch": {"type": "string",
                    "description": "1-2 sentences the candidate could open an application note with. Specific, no hype."},
            },
            "required": ["fit_score", "recommend", "resume_variant", "strongest_evidence",
                         "gaps", "dealbreakers", "pitch"],
        },
    }


def build_system(rubric: str, resumes: dict[str, str]) -> list[dict]:
    blocks = "\n\n".join(f'<resume variant="{k}">\n{v.strip()}\n</resume>' for k, v in resumes.items())
    # Rubric + resumes are identical across every call, so cache them.
    # (Caching silently no-ops if the block is under the model's minimum cacheable length.)
    return [{"type": "text", "text": f"{rubric.strip()}\n\n{blocks}",
             "cache_control": {"type": "ephemeral"}}]


def job_message(job: Job, max_chars: int) -> str:
    desc = job.description
    if len(desc) > max_chars:
        desc = desc[:max_chars] + "\n[truncated]"
    meta = ""
    if job.source == "usajobs":
        e = job.extra
        meta = (f"\nGrade: GS-{e.get('low_grade')} to GS-{e.get('high_grade')}"
                f"\nHiring paths: {', '.join(e.get('hiring_paths') or [])}")
    return (f"<job>\nTitle: {job.title}\nOrganization: {job.org}\nLocation: {job.location}{meta}\n\n"
            f"{desc}\n</job>\n\nAssess this job against the rubric and resumes. Call record_fit.")


def score_job(client, model: str, system: list[dict], tool: dict, job: Job, max_chars: int = 8000) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=1000,
        system=system,
        tools=[tool],
        tool_choice=({"type": "auto"} if model.startswith(_NO_FORCED_TOOL)
                     else {"type": "tool", "name": tool["name"]}),
        messages=[{"role": "user", "content": job_message(job, max_chars)}],
    )
    for block in msg.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError(f"no tool_use block for {job.key}")
