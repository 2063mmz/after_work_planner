"""
Deterministic evening scheduler for the LifeOps Agent.

Every exact time in this product is produced here. The language model decides

what to schedule and explains the trade-offs; 

this module decides when things happen, how much sleep is left, and which constraints are broken.
"""

from __future__ import annotations

import re
from typing import Any

MINUTES_PER_DAY = 1440
NIGHT_BOUNDARY = 6 * 60  # a clock time before 06:00 belongs to the next day
EARLIEST_WAKE = 5 * 60 + 30  # never move the alarm before 05:30
MIN_SLEEP_MINUTES = 300  # 5h hard floor before the plan is called impossible
DEFAULT_TASK_MINUTES = 30

ENERGY_PROFILES: dict[str, dict[str, int]] = {
    "low": {"optional_budget": 30, "decompress_minutes": 20, "extra_sleep": 30},
    "medium": {"optional_budget": 90, "decompress_minutes": 10, "extra_sleep": 0},
    "high": {"optional_budget": 150, "decompress_minutes": 0, "extra_sleep": 0},
}

IMPORTANCE_WEIGHT = {"low": 0, "medium": 1, "high": 2}

# Default durations for common evening tasks, matched on keywords.
TASK_LIBRARY: tuple[tuple[tuple[str, ...], int], ...] = (
    (("laundry", "washing machine", "washer"), 10),
    (("dishes", "dishwasher"), 10),
    (("shower", "bath"), 15),
    (("cook", "dinner", "meal", "eat"), 45),
    (("groceries", "grocery", "supermarket", "shopping"), 30),
    (("gym", "workout", "exercise", "run", "yoga", "swim"), 60),
    (("walk", "dog"), 30),
    (("study", "learn", "course", "language", "read"), 45),
    (("clean", "tidy", "vacuum"), 30),
    (("presentation", "review", "prepare", "prep", "slides", "report"), 45),
    (("email", "admin", "invoice", "paperwork"), 20),
    (("call", "phone", "family", "friend"), 20),
    (("film", "movie", "cinema"), 120),
    (("tv", "series", "episode", "netflix"), 60),
    (("meditat", "stretch", "journal"), 15),
)

# Lower sort key runs earlier in the evening. Stable, so user order is kept
# inside a bucket.
ORDER_HINTS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("laundry", "washing machine", "washer", "oven", "preheat"), 0),
    (("cook", "dinner", "meal", "eat", "groceries"), 1),
    (("dishes", "dishwasher"), 2),
    (("shower", "bath", "meditat", "stretch", "journal"), 4),
)
DEFAULT_ORDER_HINT = 3

RULES_APPLIED = (
    "Tomorrow's first event fixes the wake-up time; the evening is planned backwards from it.",
    "Mandatory tasks are scheduled first and may push bedtime later, but never past the hard limit.",
    "Optional activities are only kept while they fit the energy budget and the bedtime target.",
    "Low energy adds a decompression block on arrival and 30 minutes to the sleep target.",
    "Tasks that do not fit tonight move to the morning only if sleep stays above 5 hours.",
    "Bedtime never goes past 01:30 and the alarm never rings before 05:30.",
)


# Time helpers

def parse_time(value: str | None) -> int | None:
    """Parse a loose clock string into minutes since midnight.

    Accepts "20:15", "8 PM", "8:30pm", "20h15", "20.15" and "8".
    Returns None when the value cannot be understood.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    meridiem = None
    if "pm" in text or "p.m" in text:
        meridiem = "pm"
    elif "am" in text or "a.m" in text:
        meridiem = "am"

    match = re.search(r"(\d{1,2})\s*(?:[:h.]\s*(\d{1,2}))?", text)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2) or 0)
    if meridiem == "pm" and hours < 12:
        hours += 12
    elif meridiem == "am" and hours == 12:
        hours = 0
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        return None
    return hours * 60 + minutes


def format_time(minutes: int) -> str:
    """Render absolute minutes (which may cross midnight) as HH:MM."""
    minutes = int(minutes) % MINUTES_PER_DAY
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_duration(minutes: int) -> str:
    minutes = max(0, int(minutes))
    return f"{minutes // 60}h {minutes % 60:02d}min"


def _match_keywords(name: str, table: tuple[tuple[tuple[str, ...], int], ...], default: int) -> int:
    lowered = name.lower()
    for keywords, value in table:
        if any(keyword in lowered for keyword in keywords):
            return value
    return default


def estimate_task_minutes(name: str) -> int:
    """Best-guess duration for a task described in plain language."""
    return _match_keywords(name, TASK_LIBRARY, DEFAULT_TASK_MINUTES)


def normalize_tasks(tasks: list[str] | None, category: str) -> list[dict[str, Any]]:
    """Turn ``["cook dinner|45", "laundry"]`` into structured task dicts.

    An explicit ``|minutes`` suffix always wins over the keyword library.
    """
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in tasks or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        name, _, minutes_text = raw.partition("|")
        name = name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            minutes = int(float(minutes_text.strip()))
        except ValueError:
            minutes = estimate_task_minutes(name)
        minutes = max(5, min(minutes, 300))
        normalized.append(
            {
                "name": name,
                "minutes": minutes,
                "category": category,
                "order": _match_keywords(name, ORDER_HINTS, DEFAULT_ORDER_HINT),
            }
        )
    return normalized


# Tomorrow pressure

def _morning_anchors(
    first_event_time: str | None,
    commute_minutes: int,
    morning_prep_minutes: int,
    work_departure_time: str = "08:00",
) -> tuple[int | None, int, int]:
    """Return (first event, home departure, wake-up) in minutes after midnight."""
    event = parse_time(first_event_time)
    if event is not None:
        departure = event - commute_minutes
    else:
        departure = parse_time(work_departure_time)
        if departure is None:
            departure = 8 * 60
    wake = max(EARLIEST_WAKE, departure - morning_prep_minutes)
    return event, departure, wake


def assess_tomorrow_pressure(
    first_event_time: str | None = None,
    event_importance: str = "medium",
    commute_minutes: int = 30,
    morning_prep_minutes: int = 45,
    target_sleep_minutes: int = 480,
    work_departure_time: str = "08:00",
) -> dict[str, Any]:
    """Work out how hard tomorrow morning pushes on tonight.

    Returns the required wake-up time, the latest bedtime that still delivers
    the sleep target, and a pressure level used by the planner and the UI.
    """
    importance = event_importance if event_importance in IMPORTANCE_WEIGHT else "medium"
    commute_minutes = max(0, int(commute_minutes))
    morning_prep_minutes = max(10, int(morning_prep_minutes))
    target_sleep_minutes = max(MIN_SLEEP_MINUTES, int(target_sleep_minutes))

    event, departure, wake = _morning_anchors(
        first_event_time, commute_minutes, morning_prep_minutes, work_departure_time
    )

    # Everything below is expressed on tomorrow's clock (day + 1).
    wake_absolute = wake + MINUTES_PER_DAY
    latest_healthy_bedtime = wake_absolute - target_sleep_minutes

    score = IMPORTANCE_WEIGHT[importance]
    if wake <= 6 * 60 + 30:
        score += 2
    elif wake <= 7 * 60 + 15:
        score += 1
    level = "high" if score >= 3 else "medium" if score >= 2 else "low"

    if event is not None:
        headline = (
            f"First event at {format_time(event)} ({importance} importance). "
            f"Leave by {format_time(departure)} for a {commute_minutes} min commute."
        )
    else:
        headline = (
            f"No fixed event given; assuming a normal departure at {format_time(departure)}."
        )

    return {
        "first_event": format_time(event) if event is not None else None,
        "event_importance": importance,
        "commute_minutes": commute_minutes,
        "departure_time": format_time(departure),
        "required_wake_time": format_time(wake),
        "latest_healthy_bedtime": format_time(latest_healthy_bedtime),
        "target_sleep_minutes": target_sleep_minutes,
        "pressure_level": level,
        "summary": headline,
    }


def estimate_task_priority(
    tasks: list[str] | None = None,
    energy_level: str = "medium",
    tomorrow_pressure: str = "medium",
) -> dict[str, Any]:
    """Score tasks so the agent can argue about what to drop first.

    Higher scores are harder to drop. Tasks tied to tomorrow (preparing a
    presentation, packing a bag) score up when tomorrow's pressure is high.
    """
    energy = energy_level if energy_level in ENERGY_PROFILES else "medium"
    pressure = tomorrow_pressure if tomorrow_pressure in IMPORTANCE_WEIGHT else "medium"
    scored: list[dict[str, Any]] = []

    for task in normalize_tasks(tasks, "unscored"):
        name = task["name"].lower()
        score = 50
        reasons: list[str] = []

        if any(word in name for word in ("dinner", "cook", "eat", "meal", "medic", "shower")):
            score += 25
            reasons.append("basic daily need")
        if any(word in name for word in ("presentation", "review", "prepare", "prep", "client", "slides", "deadline")):
            score += 15 + 10 * IMPORTANCE_WEIGHT[pressure]
            reasons.append("feeds tomorrow's first event")
        if any(word in name for word in ("laundry", "dishes", "trash", "bins")):
            score += 5
            reasons.append("short chore, cheap to keep")
        if any(word in name for word in ("tv", "series", "netflix", "tiktok", "scroll", "game")):
            score -= 20
            reasons.append("leisure, first to drop")
        if task["minutes"] >= 90:
            score -= 10
            reasons.append("long block, expensive tonight")
        if energy == "low" and any(word in name for word in ("gym", "workout", "exercise", "run")):
            score -= 15
            reasons.append("high effort on a low-energy evening")

        scored.append(
            {
                "task": task["name"],
                "minutes": task["minutes"],
                "priority": max(0, min(100, score)),
                "reason": ", ".join(reasons) or "no strong signal, default priority",
            }
        )

    scored.sort(key=lambda item: item["priority"], reverse=True)
    return {
        "energy_level": energy,
        "tomorrow_pressure": pressure,
        "ranked_tasks": scored,
        "drop_first": [item["task"] for item in reversed(scored)][:2],
    }


# The planner

def _block(start: int, end: int, task: str, kind: str) -> dict[str, Any]:
    return {
        "start": format_time(start),
        "end": format_time(end),
        "task": task,
        "type": kind,
        "minutes": int(end - start),
    }


def create_evening_plan(
    arrival_time: str,
    mandatory_tasks: list[str] | None = None,
    optional_tasks: list[str] | None = None,
    energy_level: str = "medium",
    preferred_bedtime: str = "23:30",
    target_sleep_minutes: int = 480,
    tomorrow_first_event: str | None = None,
    tomorrow_event_importance: str = "medium",
    commute_minutes: int = 30,
    morning_prep_minutes: int = 45,
    wind_down_minutes: int = 20,
    extra_travel_minutes: int = 0,
    latest_bedtime: str = "01:30",
    mode: str = "balanced",
) -> dict[str, Any]:
    """Build a constraint-checked evening plan and compute every exact time.

    Args:
        arrival_time: Time the user gets home, e.g. "20:15" or "8 PM".
        mandatory_tasks: Must-do items tonight, each "name" or "name|minutes".
        optional_tasks: Nice-to-have items, same format. Dropped first.
        energy_level: "low", "medium" or "high". Drives the optional-activity
            budget, an arrival decompression block, and the sleep target.
        preferred_bedtime: The user's normal target bedtime in HH:MM.
        target_sleep_minutes: Sleep the user needs, in minutes (480 = 8h).
        tomorrow_first_event: Tomorrow's first fixed commitment in HH:MM.
        tomorrow_event_importance: "low", "medium" or "high".
        commute_minutes: Door-to-door travel to tomorrow's first event.
        morning_prep_minutes: Getting-ready time before leaving home.
        wind_down_minutes: Screen-free wind-down before sleeping.
        extra_travel_minutes: Extra evening travel (e.g. a detour to the gym).
        latest_bedtime: Hard bedtime limit, defaults to 01:30.
        mode: "balanced" (default), "sleep_first" (drop optional items and go
            to bed as early as possible) or "productivity" (keep everything and
            use the full night). Use the alternatives to compare trade-offs.
    """
    energy = energy_level if energy_level in ENERGY_PROFILES else "medium"
    profile = ENERGY_PROFILES[energy]
    mode = mode if mode in {"balanced", "sleep_first", "productivity"} else "balanced"

    arrival = parse_time(arrival_time)
    preferred = parse_time(preferred_bedtime)
    hard_limit = parse_time(latest_bedtime)
    if arrival is None or preferred is None or hard_limit is None:
        return {
            "status": "error",
            "message": (
                "Times must be readable clock values such as 20:15 or 8 PM "
                f"(got arrival={arrival_time!r}, bedtime={preferred_bedtime!r})."
            ),
        }

    try:
        target_sleep_minutes = int(target_sleep_minutes)
        commute_minutes = max(0, int(commute_minutes))
        morning_prep_minutes = max(10, int(morning_prep_minutes))
        wind_down_minutes = max(0, int(wind_down_minutes))
        extra_travel_minutes = max(0, int(extra_travel_minutes))
    except (TypeError, ValueError):
        return {"status": "error", "message": "Duration arguments must be whole minutes."}

    # The sleep target is the same in every mode so alternatives stay comparable.
    effective_sleep_target = max(
        MIN_SLEEP_MINUTES, target_sleep_minutes + profile["extra_sleep"]
    )

    tomorrow = assess_tomorrow_pressure(
        first_event_time=tomorrow_first_event,
        event_importance=tomorrow_event_importance,
        commute_minutes=commute_minutes,
        morning_prep_minutes=morning_prep_minutes,
        target_sleep_minutes=effective_sleep_target,
        work_departure_time="08:00",
    )

    # Absolute timeline: tonight uses today's clock, tomorrow adds 1440.
    if arrival < NIGHT_BOUNDARY:
        arrival += MINUTES_PER_DAY
    if preferred <= arrival:
        preferred += MINUTES_PER_DAY
    if hard_limit < NIGHT_BOUNDARY:
        hard_limit += MINUTES_PER_DAY
    _, departure, base_wake = _morning_anchors(
        tomorrow_first_event, commute_minutes, morning_prep_minutes
    )
    departure += MINUTES_PER_DAY
    base_wake += MINUTES_PER_DAY

    hard_bedtime = min(hard_limit, base_wake - MIN_SLEEP_MINUTES)
    if arrival >= hard_bedtime:
        return {
            "status": "error",
            "message": (
                f"Arrival at {format_time(arrival)} leaves no usable evening before the "
                f"{format_time(hard_bedtime)} bedtime limit."
            ),
        }

    preferred = min(preferred, hard_bedtime)
    healthy_bedtime = base_wake - effective_sleep_target
    if mode == "sleep_first":
        target_bedtime = min(healthy_bedtime, preferred)
    elif mode == "productivity":
        target_bedtime = hard_bedtime
    else:
        target_bedtime = min(preferred, healthy_bedtime)
    target_bedtime = max(min(target_bedtime, hard_bedtime), arrival)

    warnings: list[str] = []
    violations: list[str] = []
    evening: list[dict[str, Any]] = []
    cursor = arrival

    if extra_travel_minutes:
        evening.append(
            _block(cursor, cursor + extra_travel_minutes, "Evening travel", "commute")
        )
        cursor += extra_travel_minutes

    decompress = 0 if mode == "productivity" else profile["decompress_minutes"]
    if decompress:
        evening.append(
            _block(cursor, cursor + decompress, "Decompress after work", "buffer")
        )
        cursor += decompress

    hard_deadline = max(arrival, hard_bedtime - wind_down_minutes)
    optional_deadline = (
        hard_deadline if mode == "productivity" else max(arrival, target_bedtime - wind_down_minutes)
    )

    mandatory = sorted(normalize_tasks(mandatory_tasks, "mandatory"), key=lambda t: t["order"])
    optional = normalize_tasks(optional_tasks, "optional")

    carryover: list[dict[str, Any]] = []
    dropped: list[str] = []
    dropped_for_energy: list[str] = []

    for task in mandatory:
        end = cursor + task["minutes"]
        if end <= hard_deadline:
            evening.append(_block(cursor, end, task["name"], "mandatory"))
            cursor = end
        else:
            carryover.append(task)

    if mode == "sleep_first":
        optional_budget = 0
    elif mode == "productivity":
        optional_budget = sum(task["minutes"] for task in optional)
    else:
        optional_budget = profile["optional_budget"]
    spent = 0
    for task in optional:
        end = cursor + task["minutes"]
        if spent + task["minutes"] > optional_budget:
            dropped_for_energy.append(task["name"])
        elif end > optional_deadline:
            dropped.append(task["name"])
        else:
            evening.append(_block(cursor, end, task["name"], "optional"))
            cursor = end
            spent += task["minutes"]

    if wind_down_minutes:
        wind_end = min(cursor + wind_down_minutes, hard_bedtime)
        if wind_end > cursor:
            evening.append(_block(cursor, wind_end, "Wind down, screens off", "buffer"))
            cursor = wind_end

    # Sleep-first goes to bed as soon as the evening is done; the other modes
    # keep the usual bedtime and fill the gap with free time.
    fill_target = cursor if mode == "sleep_first" else min(preferred, healthy_bedtime)
    bedtime = min(max(cursor, fill_target), hard_bedtime)
    if bedtime > cursor:
        evening.append(_block(cursor, bedtime, "Free time", "buffer"))

    # ---- morning -----
    carryover_total = sum(task["minutes"] for task in carryover)
    earliest_wake = max(EARLIEST_WAKE + MINUTES_PER_DAY, bedtime + MIN_SLEEP_MINUTES)
    morning_capacity = max(0, base_wake - max(earliest_wake, base_wake - carryover_total))

    morning: list[dict[str, Any]] = []
    scheduled_carryover: list[dict[str, Any]] = []
    unscheduled: list[str] = []
    used = 0
    for task in carryover:
        if used + task["minutes"] <= morning_capacity:
            used += task["minutes"]
            scheduled_carryover.append(task)
        else:
            unscheduled.append(task["name"])

    wake = base_wake - used
    cursor = wake
    for task in scheduled_carryover:
        end = cursor + task["minutes"]
        morning.append(_block(cursor, end, f"{task['name']} (moved to the morning)", "mandatory"))
        cursor = end
    morning.append(_block(cursor, departure, "Get ready and leave", "routine"))
    if tomorrow["first_event"]:
        event_start = departure + commute_minutes
        if commute_minutes:
            morning.append(_block(departure, event_start, "Commute", "commute"))
        morning.append(_block(event_start, event_start + 60, "First event", "event"))

    sleep_minutes = wake - bedtime

    # ---- diagnostics ------
    if sleep_minutes < effective_sleep_target:
        deficit = effective_sleep_target - sleep_minutes
        warnings.append(
            f"Sleep is {format_duration(deficit)} short of the "
            f"{format_duration(effective_sleep_target)} target."
        )
    if sleep_minutes < 6 * 60:
        warnings.append(
            f"Only {format_duration(sleep_minutes)} of sleep before a "
            f"{tomorrow['pressure_level']}-pressure morning."
        )
    if dropped:
        warnings.append("No time left tonight for: " + ", ".join(dropped) + ".")
    if dropped_for_energy:
        warnings.append(
            f"Held back to protect a {energy}-energy evening: "
            + ", ".join(dropped_for_energy)
            + "."
        )
    if scheduled_carryover:
        warnings.append(
            "Moved to the morning (alarm at "
            + format_time(wake)
            + "): "
            + ", ".join(task["name"] for task in scheduled_carryover)
            + "."
        )
    for name in unscheduled:
        violations.append(f'"{name}" fits neither tonight nor tomorrow morning.')

    readiness = _readiness_score(
        sleep_minutes=sleep_minutes,
        target_sleep=effective_sleep_target,
        pressure=tomorrow["pressure_level"],
        warnings=warnings,
        violations=violations,
    )

    return {
        "status": "ok" if not violations else "constraint_violation",
        "mode": mode,
        "energy_level": energy,
        "arrival_time": format_time(arrival),
        "evening_schedule": evening,
        "morning_schedule": morning,
        "bedtime": format_time(bedtime),
        "wake_time": format_time(wake),
        "sleep_duration": format_duration(sleep_minutes),
        "sleep_minutes": sleep_minutes,
        "target_sleep_minutes": effective_sleep_target,
        "tomorrow": tomorrow,
        "morning_carryover": [task["name"] for task in scheduled_carryover],
        "dropped_optional": dropped + dropped_for_energy,
        "unscheduled": unscheduled,
        "warnings": warnings,
        "violations": violations,
        "tomorrow_readiness_score": readiness,
        "totals": {
            "mandatory_minutes": sum(
                block["minutes"] for block in evening if block["type"] == "mandatory"
            ),
            "optional_minutes": spent,
            "buffer_minutes": sum(
                block["minutes"] for block in evening if block["type"] == "buffer"
            ),
        },
        "rules_applied": list(RULES_APPLIED),
    }


def _readiness_score(
    sleep_minutes: int,
    target_sleep: int,
    pressure: str,
    warnings: list[str],
    violations: list[str],
) -> int:
    """Score how ready the user will be for tomorrow, from 0 to 100."""
    score = 100.0
    deficit = max(0, target_sleep - sleep_minutes)
    score -= min(45.0, deficit * 45 / 120)  # a two-hour deficit costs 45 points
    score -= min(30, 12 * len(violations))
    score -= min(15, 5 * len(warnings))
    if pressure == "high" and sleep_minutes < target_sleep:
        score -= 10
    if not warnings and not violations and sleep_minutes >= target_sleep:
        score += 5
    return int(max(0, min(100, round(score))))


def compare_plans(base_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the planner in each mode so trade-offs can be compared on real times."""
    labels = {
        "sleep_first": "Sleep-first plan",
        "balanced": "Balanced plan",
        "productivity": "Productivity plan",
    }
    results = []
    for mode, name in labels.items():
        plan = create_evening_plan(**{**base_kwargs, "mode": mode})
        if plan.get("status") == "error":
            continue
        results.append(
            {
                "name": name,
                "mode": mode,
                "bedtime": plan["bedtime"],
                "wake_time": plan["wake_time"],
                "sleep_duration": plan["sleep_duration"],
                "tomorrow_readiness_score": plan["tomorrow_readiness_score"],
                "kept_tonight": [
                    block["task"]
                    for block in plan["evening_schedule"]
                    if block["type"] in {"mandatory", "optional"}
                ],
                "dropped_optional": plan["dropped_optional"],
                "morning_carryover": plan["morning_carryover"],
            }
        )
    return results
