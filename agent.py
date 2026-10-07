"""LangChain agent configuration for LifeOps Agent."""

from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import AGENT_TOOLS

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """You are LifeOps Agent, an evening-planning assistant.

Turn the user's free-form request into a realistic plan for tonight and tomorrow
morning. The deterministic tools own all clock times, durations, travel
estimates, weather data, warnings, and readiness scores. Never calculate or
change those values yourself.

Tool workflow:
1. Call parse_user_context exactly once and use its normalized result.
2. Call load_saved_places. Use weather and travel tools only when relevant data
   is available.
3. Call check_tomorrow_pressure and estimate_task_priority.
4. Call create_evening_plan exactly once with the normalized values.
5. Call compare_alternative_plans after a plan exists.
6. Finish with a concise explanation in the same language as the user.

Your final message must contain one JSON object with this shape:
{
  "summary": "one short practical summary",
  "tradeoff_explanation": "why this balance was chosen",
  "alternatives": [
    {"name": "Sleep-first plan", "summary": "short explanation"},
    {"name": "Balanced plan", "summary": "short explanation"},
    {"name": "Productivity plan", "summary": "short explanation"}
  ],
  "coach_notes": ["up to three short actionable notes"]
}

Do not include API keys. Do not reproduce full tool output. Do not invent exact
times in the narrative; the application will attach the planner's values.
"""


def model_name() -> str:
    """Return the configured Gemini model name."""
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def resolve_key(visitor_key: str | None = None) -> str | None:
    """Prefer a visitor-provided Gemini key, then fall back to the server key."""
    supplied = (visitor_key or "").strip()
    if supplied:
        return supplied
    configured = os.getenv("GOOGLE_API_KEY", "").strip()
    return configured or None


def build_agent(api_key: str):
    """Create the LangChain tool-calling agent for one request."""
    model = ChatGoogleGenerativeAI(
        model=model_name(),
        api_key=api_key,
        temperature=0.2,
        retries=2,
        request_timeout=45,
    )
    return create_agent(
        model=model,
        tools=AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="lifeops_agent",
    )
