"""Stage 2: Claude reads the job against your resumes and a written rubric,
and returns a structured assessment via a forced tool call.
"""
from __future__ import annotations

import logging

from .models import Job

log = logging.getLogger(__name__)

# Thinking is on by default for these, so the reply needs room beyond the tool call, and
# forced tool_choice ("tool"/"any") returns a 400. Ask with "auto"; the prompt names the tool.
_THINKS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-haiku-5-5")
_NO_FORCED_TOOL = _THINKS

# USD per million tokens, standard API (claude-api skill pricing table, cached 2026-10-06).
# Cache writes (5 min TTL) cost 1.25x input and reads 0.1x; the Batch API halves everything.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
# Shortest system prompt that will cache, by model (prompt caching docs).
MIN_CACHEABLE = {"claude-haiku-4-5": 4096, "claude-haiku-5-5": 512, "claude-sonnet-5-5": 512}


def _family(model: str) -> str:
    for k in PRICES:
        if model.startswith(k):
            return k
    raise KeyError(f"no price for {model}; add it to rerank.PRICES")


def cost_usd(model: str, usage: dict, batch: bool = False) -> float:
    """usage: input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens."""
    pin, pout = PRICES[_family(model)]
    c = (usage.get("input_tokens", 0) * pin + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
         + usage.get("cache_read_input_tokens", 0) * pin * 0.1 + usage.get("output_tokens", 0) * pout) / 1e6
    return c * (0.5 if batch else 1.0)


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
                    "description": "Hard requirements clearly not met "
                                   "(degree, clearance, years, citizenship, sector)."},
                "pitch": {"type": "string",
                    "description": "1-2 sentences the candidate could open an application note with. "
                                   "Specific, no hype."},
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


def request_params(model: str, system: list[dict], tool: dict, job: Job, max_chars: int = 8000) -> dict:
    """One Messages API request. Shared by the daily run and the eval batches so both ask the same thing."""
    return dict(
        model=model,
        max_tokens=16000 if model.startswith(_THINKS) else 1000,
        system=system,
        tools=[tool],
        tool_choice=({"type": "auto"} if model.startswith(_NO_FORCED_TOOL)
                     else {"type": "tool", "name": tool["name"]}),
        messages=[{"role": "user", "content": job_message(job, max_chars)}],
    )


def parse_result(msg, key: str = "") -> dict:
    if getattr(msg, "stop_reason", None) == "refusal":
        raise RuntimeError(f"model declined {key}")
    for block in msg.content:
        if block.type == "tool_use":
            r = dict(block.input)
            r["fit_score"] = int(r["fit_score"])
            if not 0 <= r["fit_score"] <= 10 or r.get("recommend") not in ("apply", "maybe", "skip"):
                raise RuntimeError(f"out-of-range assessment for {key}: {r}")
            return r
    raise RuntimeError(f"no tool_use block for {key}")


def usage_dict(msg) -> dict:
    u = getattr(msg, "usage", None)
    return {k: int(getattr(u, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def score_job(client, model: str, system: list[dict], tool: dict, job: Job, max_chars: int = 8000) -> dict:
    msg = client.messages.create(**request_params(model, system, tool, job, max_chars))
    return parse_result(msg, job.key)
