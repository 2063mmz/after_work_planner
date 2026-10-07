"""FastAPI application for the LifeOps evening-planning agent."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import agent_trace
import tools

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


class PlanRequest(BaseModel):
    """Payload accepted from the browser."""

    message: str = Field(default="", max_length=8000)
    energy_level: str = "medium"
    location: str | None = None
    saved_places: dict[str, str] = Field(default_factory=dict)
    api_key: str | None = Field(default=None, max_length=300)


app = FastAPI(title="LifeOps Agent", version="1.0.0")

origins = [
    value.strip()
    for value in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials="*" not in origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return backend and model configuration status."""
    configured = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    return {
        "status": "ok",
        "llm_configured": configured,
        "byo_key_required": not configured,
        "model": agent.model_name(),
    }


def _record_failure(recorder: agent_trace.TraceRecorder, detail: str) -> None:
    recorder.record("recommendation", "failed", detail, {})


def _error_response(
    recorder: agent_trace.TraceRecorder,
    code: str,
    message: str,
) -> dict[str, Any]:
    _record_failure(recorder, message)
    return {
        "status": "error",
        "error": {"code": code, "message": message},
        "summary": message,
        "agent_trace": list(recorder.entries),
        "recommended_plan": None,
        "alternatives": [],
        "tradeoff_explanation": "",
        "coach_notes": [],
        "tomorrow_readiness_score": 0,
        "weather": None,
        "travel": None,
    }


def _visitor_key_error(value: str | None) -> str | None:
    supplied = (value or "").strip()
    if supplied and (not supplied.startswith("AIza") or len(supplied) < 20):
        return (
            "The supplied Gemini API key does not look like a valid Google API key. "
            "Create or copy a key from https://aistudio.google.com/app/apikey."
        )
    return None


def _safe_agent_error(error: Exception, api_key: str) -> str:
    raw = str(error).replace(api_key, "[redacted]")
    lowered = raw.lower()
    if (
        "api key" in lowered
        and any(word in lowered for word in ("invalid", "not valid", "rejected", "unauthorized"))
    ) or "api_key_invalid" in lowered:
        return "The Gemini API key was rejected. Check the key and try again."
    if any(word in lowered for word in ("quota", "resource_exhausted", "rate limit", "429")):
        return "The Gemini API quota exceeded its limit. Check the quota or try again later."
    if not raw.strip():
        return "The Gemini agent failed without returning an error message."
    return f"The Gemini agent could not complete the request: {raw}"


def _agent_prompt(request: PlanRequest) -> str:
    context = {
        "energy_level": request.energy_level,
        "location": request.location,
        "saved_places": request.saved_places,
    }
    return (
        "Plan this user's evening and tomorrow morning.\n\n"
        f"User request:\n{request.message.strip()}\n\n"
        f"Interface context:\n{json.dumps(context, ensure_ascii=False)}"
    )


def _message_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result or "").strip()
    messages = result.get("messages") or []
    if not messages:
        structured = result.get("structured_response")
        return json.dumps(structured, ensure_ascii=False) if structured else ""
    message = messages[-1]
    content = getattr(message, "content", None)
    if content is None:
        content = getattr(message, "text", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _narrative(text: str) -> dict[str, Any]:
    """Extract the model's final JSON while preserving a prose fallback."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"summary": text.strip()} if text.strip() else {}


def _fallback_context(request: PlanRequest) -> dict[str, Any]:
    time_match = re.search(
        r"\b(?:[01]?\d|2[0-3])(?::[0-5]\d)?(?:\s*(?:a\.?m\.?|p\.?m\.?))?\b",
        request.message,
        flags=re.IGNORECASE,
    )
    arrival = time_match.group(0) if time_match else "19:00"
    return tools.parse_user_context.invoke(
        {
            "arrival_time": arrival,
            "energy_level": request.energy_level,
            "location": request.location,
        }
    )


def _ensure_plan(request: PlanRequest, recorder: agent_trace.TraceRecorder) -> dict[str, Any]:
    plan = recorder.data.get("plan")
    if plan:
        return plan

    context = recorder.data.get("context") or _fallback_context(request)
    plan = tools.create_evening_plan.invoke(
        {
            "arrival_time": context["arrival_time"],
            "mandatory_tasks": context.get("mandatory_tasks") or [],
            "optional_tasks": context.get("optional_tasks") or [],
            "energy_level": context.get("energy_level", request.energy_level),
            "preferred_bedtime": context.get("preferred_bedtime", "23:30"),
            "target_sleep_hours": context.get("target_sleep_minutes", 480) / 60,
            "tomorrow_first_event": context.get("tomorrow_first_event"),
            "tomorrow_event_importance": context.get(
                "tomorrow_event_importance", "medium"
            ),
            "commute_minutes": context.get("commute_minutes", 30),
        }
    )
    return plan


def _ensure_alternatives(recorder: agent_trace.TraceRecorder) -> list[dict[str, Any]]:
    alternatives = recorder.data.get("alternatives")
    if alternatives is None:
        result = tools.compare_alternative_plans.invoke({})
        alternatives = result.get("alternatives") or []
    return alternatives


def _default_alternative_summary(item: dict[str, Any]) -> str:
    mode = item.get("mode")
    if mode == "sleep_first":
        return "Drops optional tasks to maximize sleep and recovery."
    if mode == "productivity":
        return "Keeps more tasks tonight, with less room for rest."
    return "Balances useful progress with the sleep target."


def _merge_alternatives(
    computed: list[dict[str, Any]],
    written: Any,
) -> list[dict[str, Any]]:
    model_items = written if isinstance(written, list) else []
    by_name = {
        str(item.get("name", "")).strip().lower(): item
        for item in model_items
        if isinstance(item, dict)
    }
    by_mode = {
        str(item.get("mode", "")).strip().lower(): item
        for item in model_items
        if isinstance(item, dict)
    }
    merged = []
    for item in computed:
        model_item = by_mode.get(str(item.get("mode", "")).lower()) or by_name.get(
            str(item.get("name", "")).lower()
        )
        result = dict(item)
        result["summary"] = (
            str(model_item.get("summary", "")).strip()
            if model_item
            else _default_alternative_summary(item)
        )
        if not result["summary"]:
            result["summary"] = _default_alternative_summary(item)
        merged.append(result)
    return merged


def _tool_result(recorder: agent_trace.TraceRecorder, tool_name: str) -> Any:
    for entry in reversed(recorder.entries):
        if entry["tool"] == tool_name:
            return entry.get("result")
    return None


def _finish_trace(
    recorder: agent_trace.TraceRecorder,
    recommendation_detail: str,
) -> list[dict[str, Any]]:
    called = {entry["tool"] for entry in recorder.entries}
    for tool_name in agent_trace.STEP_ORDER:
        if tool_name == "recommendation" or tool_name in called:
            continue
        recorder.record(
            tool_name,
            "skipped",
            "The agent did not need this step for the final plan.",
            {},
        )
    recorder.record(
        "recommendation",
        "complete",
        recommendation_detail,
        {},
    )
    latest = {entry["tool"]: entry for entry in recorder.entries}
    return [latest[name] for name in agent_trace.STEP_ORDER]


def _execute_plan(
    request: PlanRequest,
    recorder: agent_trace.TraceRecorder | None = None,
) -> dict[str, Any]:
    recorder = recorder or agent_trace.TraceRecorder()
    agent_trace.start(recorder)
    recorder.remember("saved_places", request.saved_places)

    if not request.message.strip():
        return _error_response(
            recorder,
            "empty_request",
            "Describe your evening before asking the agent to build a plan.",
        )

    key_error = _visitor_key_error(request.api_key)
    if key_error:
        return _error_response(recorder, "missing_api_key", key_error)

    api_key = agent.resolve_key(request.api_key)
    if not api_key:
        return _error_response(
            recorder,
            "missing_api_key",
            "Set GOOGLE_API_KEY on the server or provide a Gemini API key from "
            "https://aistudio.google.com/app/apikey.",
        )

    try:
        executor = agent.build_agent(api_key)
        result = executor.invoke(
            {"messages": [{"role": "user", "content": _agent_prompt(request)}]},
            config={"recursion_limit": 40},
        )
    except Exception as error:
        return _error_response(
            recorder,
            "agent_failed",
            _safe_agent_error(error, api_key),
        )

    plan = _ensure_plan(request, recorder)
    if plan.get("status") == "error":
        return _error_response(
            recorder,
            "planning_failed",
            plan.get("message", "The scheduling constraints could not produce a plan."),
        )

    computed_alternatives = _ensure_alternatives(recorder)
    text = _message_text(result)
    narrative = _narrative(text)
    summary = str(narrative.get("summary") or text or "Your evening plan is ready.").strip()
    tradeoff = str(narrative.get("tradeoff_explanation") or "").strip()
    if not tradeoff:
        tradeoff = (
            f"The balanced plan protects {plan['sleep_duration']} of sleep while keeping "
            "the highest-priority work that fits safely tonight."
        )

    alternatives = _merge_alternatives(
        computed_alternatives,
        narrative.get("alternatives"),
    )
    coach_notes = narrative.get("coach_notes")
    if not isinstance(coach_notes, list):
        coach_notes = []
    coach_notes = [str(note).strip() for note in coach_notes[:3] if str(note).strip()]

    trace = _finish_trace(recorder, "The final recommendation is ready.")
    return {
        "status": "ok",
        "error": None,
        "summary": summary,
        "agent_trace": trace,
        "recommended_plan": plan,
        "alternatives": alternatives,
        "tradeoff_explanation": tradeoff,
        "coach_notes": coach_notes,
        "tomorrow_readiness_score": plan["tomorrow_readiness_score"],
        "weather": _tool_result(recorder, "get_weather_context"),
        "travel": _tool_result(recorder, "estimate_travel_time"),
    }


@app.post("/api/plan")
def create_plan(request: PlanRequest) -> dict[str, Any]:
    """Run the agent and return one complete plan response."""
    return _execute_plan(request)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/plan/stream")
async def stream_plan(request: PlanRequest) -> StreamingResponse:
    """Stream trace events while the agent builds the final response."""

    async def events():
        recorder = agent_trace.TraceRecorder()
        yield _sse({"type": "start"})
        task = asyncio.create_task(asyncio.to_thread(_execute_plan, request, recorder))

        while not task.done():
            while True:
                try:
                    event = recorder.events.get_nowait()
                except queue.Empty:
                    break
                yield _sse(event)
            await asyncio.sleep(0.02)

        result = await task
        while True:
            try:
                event = recorder.events.get_nowait()
            except queue.Empty:
                break
            yield _sse(event)
        yield _sse({"type": "result", "payload": result})
        yield _sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
