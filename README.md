# LifeOps Agent

LifeOps Agent creates an evening plan around your tasks, energy level, weather, next-day schedule, and sleep needs.

Install the dependencies from `requirements.txt`, set `GOOGLE_API_KEY`, and run `python -m uvicorn app:app --reload`.

The project uses FastAPI, LangChain, Google Gemini, and Open-Meteo, with a frontend built in HTML, CSS, and JavaScript.

`frontend/` contains the interface, while the backend handles the API and agent workflow; `planner.py`, `tools.py`, and `agent_trace.py` provide scheduling, external tools, and execution tracing.
# after_work_planner
