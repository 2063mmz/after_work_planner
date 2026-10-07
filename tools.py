"""LangChain tools the LifeOps agent can call.

Every tool is deterministic Python: the model chooses which tool to call and
with what arguments, but never invents times, distances or weather. Each call
is recorded so the dashboard can show the agent's actual workflow.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from langchain_core.tools import tool

import agent_trace
import planner

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 6.0

# Straight-line distance underestimates real routes; this is the usual fudge.
ROUTE_FACTOR = 1.3

# Average door-to-door speeds (km/h) and fixed overhead (minutes) for a dense
# European city. Good enough for evening planning, and free.
TRAVEL_MODES: dict[str, tuple[float, int]] = {
    "walk": (4.8, 0),
    "bike": (15.0, 3),
    "transit": (24.0, 6),
    "drive": (22.0, 6),
}
DEFAULT_TRAVEL_MINUTES = {"walk": 20, "bike": 12, "transit": 30, "drive": 20}

WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
99: "thunderstorm with hail",
}

_GEOCODE_CACHE: dict[str, dict[str, Any] | None] = {}


# --------------------------------------------------------------------------
# Free geo/weather helpers (Open-Meteo, no API key required)
# --------------------------------------------------------------------------


def geocode(place: str) -> dict[str, Any] | None:
    """Resolve a place name to coordinates using Open-Meteo's free geocoder."""
    key = place.strip().lower()
    if not key:
        return None
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]

    result: dict[str, Any] | None = None
    try:
        response = httpx.get(
            GEOCODING_URL,
            params={"name": place, "count": 1, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        matches = response.json().get("results") or []
        if matches:
            match = matches[0]
            result = {
                "name": match.get("name"),
                "country": match.get("country"),
                "latitude": match["latitude"],
                "longitude": match["longitude"],
            }
    except (httpx.HTTPError, KeyError, ValueError):
        result = None

    _GEOCODE_CACHE[key] = result
    return result


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def travel_minutes(distance_km: float, mode: str) -> int:
    speed, overhead = TRAVEL_MODES.get(mode, TRAVEL_MODES["transit"])
    return int(round(distance_km * ROUTE_FACTOR / speed * 60)) + overhead


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@tool(parse_docstring=True)
def parse_user_context(
    arrival_time: str,
    mandatory_tasks: list[str] | None = None,
    optional_tasks: list[str] | None = None,
    energy_level: str = "medium",
    tomorrow_first_event: str | None = None,
    tomorrow_event_importance: str = "medium",
    preferred_bedtime: str = "23:30",
    target_sleep_hours: float = 8.0,
    commute_minutes: int = 30,
    location: str | None = None,
) -> dict[str, Any]:
    """Normalise everything you extracted from the user's message.

    Call this first, exactly once. It converts loose clock strings such as
    "8 PM" into HH:MM, applies defaults, and returns the canonical context to
    use for every later tool call.

    Args:
        arrival_time: When the user gets home, e.g. "20:15" or "8 PM".
        mandatory_tasks: Must-do items tonight as "name" or "name|minutes".
        optional_tasks: Nice-to-have items in the same format.
        energy_level: "low", "medium" or "high".
        tomorrow_first_event: Tomorrow's first fixed commitment, e.g. "09:00".
        tomorrow_event_importance: "low", "medium" or "high".
        preferred_bedtime: The user's usual target bedtime.
        target_sleep_hours: Hours of sleep the user needs, e.g. 7.5.
        commute_minutes: Door-to-door commute to tomorrow's first event.
        location: City for weather, e.g. "Paris, France".
    """
    recorder = agent_trace.current()
    arrival = planner.parse_time(arrival_time)
    bedtime = planner.parse_time(preferred_bedtime)
    event = planner.parse_time(tomorrow_first_event)

    context = {
        "arrival_time": planner.format_time(arrival) if arrival is not None else None,
        "mandatory_tasks": [
            f"{task['name']}|{task['minutes']}"
            for task in planner.normalize_tasks(mandatory_tasks, "mandatory")
        ],
        "optional_tasks": [
            f"{task['name']}|{task['minutes']}"
            for task in planner.normalize_tasks(optional_tasks, "optional")
        ],
        "energy_level": energy_level if energy_level in planner.ENERGY_PROFILES else "medium",
        "tomorrow_first_event": planner.format_time(event) if event is not None else None,
        "tomorrow_event_importance": (
            tomorrow_event_importance
            if tomorrow_event_importance in planner.IMPORTANCE_WEIGHT
            else "medium"
        ),
        "preferred_bedtime": planner.format_time(bedtime) if bedtime is not None else "23:30",
        "target_sleep_minutes": int(max(5.0, min(float(target_sleep_hours), 12.0)) * 60),
        "commute_minutes": max(0, int(commute_minutes)),
        "location": location or None,
    }
    if context["arrival_time"] is None:
        context["needs_clarification"] = (
            "No arrival time could be read from the message; assuming 19:00."
        )
        context["arrival_time"] = "19:00"

    recorder.remember("context", context)
    detail = (
        f"Home at {context['arrival_time']}, {len(context['mandatory_tasks'])} must-do and "
        f"{len(context['optional_tasks'])} optional items, {context['energy_level']} energy."
    )
    recorder.record("parse_user_context", "complete", detail, context)
    return context


@tool(parse_docstring=True)
def load_saved_places(purpose: str = "travel") -> dict[str, Any]:
    """Read the places the user saved in the app (home, work, gym, ...).

    Use this before estimating travel so you know the real origin and
    destination instead of guessing.

    Args:
        purpose: Why the places are needed, e.g. "travel" or "weather".
    """
    recorder = agent_trace.current()
    places = recorder.data.get("saved_places") or {}
    detail = (
        "Saved places: " + ", ".join(f"{k} → {v}" for k, v in places.items())
        if places
        else "No saved places; falling back to the location field."
    )
    result = {"purpose": purpose, "saved_places": places, "count": len(places)}
    recorder.record("load_saved_places", "complete" if places else "skipped", detail, result)
    return result


@tool(parse_docstring=True)
def check_tomorrow_pressure(
    first_event_time: str | None = None,
    event_importance: str = "medium",
    commute_minutes: int = 30,
    morning_prep_minutes: int = 45,
    target_sleep_hours: float = 8.0,
) -> dict[str, Any]:
    """Compute how hard tomorrow morning constrains tonight.

    Returns the required wake-up time, the latest bedtime that still delivers
    the sleep target, and a pressure level. Use these numbers when you explain
    the trade-offs; do not compute them yourself.

    Args:
        first_event_time: Tomorrow's first fixed commitment, e.g. "09:00".
        event_importance: "low", "medium" or "high".
        commute_minutes: Door-to-door travel time to that event.
        morning_prep_minutes: Getting-ready time before leaving home.
        target_sleep_hours: Hours of sleep the user needs.
    """
    recorder = agent_trace.current()
    result = planner.assess_tomorrow_pressure(
        first_event_time=first_event_time,
        event_importance=event_importance,
        commute_minutes=commute_minutes,
        morning_prep_minutes=morning_prep_minutes,
        target_sleep_minutes=int(max(5.0, min(float(target_sleep_hours), 12.0)) * 60),
    )
    recorder.remember("tomorrow", result)
    detail = (
        f"{result['pressure_level'].capitalize()} pressure — wake at "
        f"{result['required_wake_time']}, in bed by {result['latest_healthy_bedtime']}."
    )
    recorder.record("check_tomorrow_pressure", "complete", detail, result)
    return result


@tool(parse_docstring=True)
def estimate_task_priority(
    tasks: list[str] | None = None,
    energy_level: str = "medium",
    tomorrow_pressure: str = "medium",
) -> dict[str, Any]:
    """Rank tasks so you know what to drop first when the evening is too full.

    Args:
        tasks: All candidate tasks as "name" or "name|minutes".
        energy_level: "low", "medium" or "high".
        tomorrow_pressure: Pressure level from check_tomorrow_pressure.
    """
    recorder = agent_trace.current()
    result = planner.estimate_task_priority(tasks, energy_level, tomorrow_pressure)
    top = result["ranked_tasks"][0]["task"] if result["ranked_tasks"] else "nothing"
    detail = f"Ranked {len(result['ranked_tasks'])} tasks; keep \"{top}\" first."
    recorder.record("estimate_task_priority", "complete", detail, result)
    return result


@tool(parse_docstring=True)
def get_weather_context(location: str) -> dict[str, Any]:
    """Look up this evening's and tomorrow morning's weather (Open-Meteo, free).

    Use it to decide whether outdoor activities are sensible and whether the
    morning needs extra time. If the lookup fails, keep planning without it.

    Args:
        location: City or area, e.g. "Paris, France" or "Lyon".
    """
    recorder = agent_trace.current()
    place = geocode(location)
    if not place:
        detail = f'Could not locate "{location}"; planning without weather.'
        result = {"available": False, "reason": detail}
        recorder.record("get_weather_context", "unavailable", detail, result, {"location": location})
        return result

    try:
        response = httpx.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "hourly": "temperature_2m,precipitation_probability,weather_code",
                "timezone": "auto",
                "forecast_days": 2,
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        hourly = response.json()["hourly"]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        detail = f"Weather service unavailable ({type(error).__name__}); planning without it."
        result = {"available": False, "reason": detail}
        recorder.record("get_weather_context", "unavailable", detail, result, {"location": location})
        return result

    evening = _window_summary(hourly, start=18, end=23)
    morning = _window_summary(hourly, start=24 + 7, end=24 + 9)
    result = {
        "available": True,
        "location": f"{place['name']}, {place['country']}",
        "evening": evening,
        "tomorrow_morning": morning,
        "advice": _weather_advice(evening, morning),
    }
    detail = (
        f"{result['location']}: {evening['description']}, {evening['temperature_c']}°C tonight, "
        f"{evening['precipitation_probability']}% rain."
    )
    recorder.record("get_weather_context", "complete", detail, result, {"location": location})
    return result


def _window_summary(hourly: dict[str, list], start: int, end: int) -> dict[str, Any]:
    """Summarise a slice of Open-Meteo's hourly arrays (index = hours from 00:00 today)."""
    temps = hourly.get("temperature_2m") or []
    rain = hourly.get("precipitation_probability") or []
    codes = hourly.get("weather_code") or []
    window = range(start, min(end + 1, len(temps)))
    values = [temps[i] for i in window if temps[i] is not None]
    rains = [rain[i] for i in window if i < len(rain) and rain[i] is not None]
    slice_codes = [codes[i] for i in window if i < len(codes) and codes[i] is not None]
    # Most frequent code wins; ties go to the more disruptive weather.
    dominant = max(slice_codes, key=lambda code: (slice_codes.count(code), code)) if slice_codes else 0
    return {
        "temperature_c": round(sum(values) / len(values), 1) if values else None,
        "precipitation_probability": max(rains) if rains else 0,
        "description": WEATHER_CODES.get(int(dominant), "unsettled"),
    }


def _weather_advice(evening: dict[str, Any], morning: dict[str, Any]) -> str:
    notes = []
    if evening["precipitation_probability"] >= 50:
        notes.append("rain tonight, keep outdoor errands short or move them indoors")
    if morning["precipitation_probability"] >= 50:
        notes.append("wet morning commute, allow a few extra minutes")
    if evening["temperature_c"] is not None and evening["temperature_c"] <= 5:
        notes.append("cold evening, outdoor activities will feel harder")
    return "; ".join(notes) or "nothing in the forecast should change the plan"


@tool(parse_docstring=True)
def estimate_travel_time(origin: str, destination: str, mode: str = "transit") -> dict[str, Any]:
    """Estimate door-to-door travel time between two places, for free.

    Uses Open-Meteo geocoding plus a straight-line distance and an average
    city speed. It is an estimate, not a routed itinerary. If a place cannot
    be resolved, a conservative default for the mode is returned instead.

    Args:
        origin: Where the trip starts, e.g. "La Defense".
        destination: Where the trip ends, e.g. "Paris 11e".
        mode: "walk", "bike", "transit" or "drive".
    """
    recorder = agent_trace.current()
    mode = mode if mode in TRAVEL_MODES else "transit"
    start, end = geocode(origin), geocode(destination)

    if not start or not end:
        minutes = DEFAULT_TRAVEL_MINUTES[mode]
        detail = (
            f"Could not resolve both places; using a default {minutes} min {mode} estimate."
        )
        result = {
            "minutes": minutes,
            "mode": mode,
            "distance_km": None,
            "confidence": "low",
            "note": detail,
        }
    else:
        distance = haversine_km(
            start["latitude"], start["longitude"], end["latitude"], end["longitude"]
        )
        minutes = max(5, travel_minutes(distance, mode))
        detail = f"{origin} → {destination}: about {minutes} min by {mode} ({distance:.1f} km)."
        result = {
            "minutes": minutes,
            "mode": mode,
            "distance_km": round(distance, 1),
            "confidence": "medium",
            "note": "Straight-line estimate with an average city speed, not a routed trip.",
        }

    recorder.record(
        "estimate_travel_time",
        "complete",
        detail,
        result,
        {"origin": origin, "destination": destination, "mode": mode},
    )
    return result


@tool(parse_docstring=True)
def create_evening_plan(
    arrival_time: str,
    mandatory_tasks: list[str] | None = None,
    optional_tasks: list[str] | None = None,
    energy_level: str = "medium",
    preferred_bedtime: str = "23:30",
    target_sleep_hours: float = 8.0,
    tomorrow_first_event: str | None = None,
    tomorrow_event_importance: str = "medium",
    commute_minutes: int = 30,
    morning_prep_minutes: int = 45,
    wind_down_minutes: int = 20,
    extra_travel_minutes: int = 0,
) -> dict[str, Any]:
    """Build the evening plan. This tool owns every exact time in the answer.

    Call it exactly once with the normalised context. It returns the evening
    and morning schedules, bedtime, wake-up time, sleep duration, warnings,
    constraint violations and a tomorrow-readiness score. Never adjust these
    times yourself.

    Args:
        arrival_time: When the user gets home, in HH:MM.
        mandatory_tasks: Must-do items as "name" or "name|minutes".
        optional_tasks: Nice-to-have items in the same format.
        energy_level: "low", "medium" or "high".
        preferred_bedtime: The user's usual target bedtime in HH:MM.
        target_sleep_hours: Hours of sleep the user needs.
        tomorrow_first_event: Tomorrow's first commitment in HH:MM.
        tomorrow_event_importance: "low", "medium" or "high".
        commute_minutes: Door-to-door commute to that event.
        morning_prep_minutes: Getting-ready time before leaving home.
        wind_down_minutes: Screen-free wind-down before sleep.
        extra_travel_minutes: Extra evening travel, from estimate_travel_time.
    """
    recorder = agent_trace.current()
    kwargs = {
        "arrival_time": arrival_time,
        "mandatory_tasks": mandatory_tasks,
        "optional_tasks": optional_tasks,
        "energy_level": energy_level,
        "preferred_bedtime": preferred_bedtime,
        "target_sleep_minutes": int(max(5.0, min(float(target_sleep_hours), 12.0)) * 60),
        "tomorrow_first_event": tomorrow_first_event,
        "tomorrow_event_importance": tomorrow_event_importance,
        "commute_minutes": commute_minutes,
        "morning_prep_minutes": morning_prep_minutes,
        "wind_down_minutes": wind_down_minutes,
        "extra_travel_minutes": extra_travel_minutes,
    }
    plan = planner.create_evening_plan(**kwargs)

    if plan.get("status") == "error":
        recorder.record("create_evening_plan", "failed", plan["message"], plan, kwargs)
        return plan

    recorder.remember("plan", plan)
    recorder.remember("plan_kwargs", kwargs)
    detail = (
        f"Bed {plan['bedtime']}, up {plan['wake_time']}, {plan['sleep_duration']} of sleep, "
        f"readiness {plan['tomorrow_readiness_score']}/100."
    )
    recorder.record("create_evening_plan", "complete", detail, plan, kwargs)
    return plan


@tool
def compare_alternative_plans() -> dict[str, Any]:
    """Re-run the planner in sleep-first and productivity modes.

    Call this after create_evening_plan. It returns real computed times for
    each alternative so you can explain the trade-off with actual numbers.
    """
    recorder = agent_trace.current()
    kwargs = recorder.data.get("plan_kwargs")
    if not kwargs:
        detail = "No plan has been created yet; call create_evening_plan first."
        recorder.record("compare_alternative_plans", "skipped", detail, {})
        return {"error": detail}

    alternatives = planner.compare_plans(kwargs)
    recorder.remember("alternatives", alternatives)
    detail = "; ".join(
        f"{item['name']}: bed {item['bedtime']}, {item['sleep_duration']}" for item in alternatives
    )
    recorder.record(
        "compare_alternative_plans", "complete", detail, {"alternatives": alternatives}
    )
    return {"alternatives": alternatives}


AGENT_TOOLS = [
    parse_user_context,
    load_saved_places,
    check_tomorrow_pressure,
    estimate_task_priority,
    get_weather_context,
    estimate_travel_time,
    create_evening_plan,
    compare_alternative_plans,
]
