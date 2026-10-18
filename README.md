# AI Agent Dashboard

A lightweight FastAPI-based dashboard for agent workflow orchestration, model metadata, and task execution.

## Features

- Health and model status endpoints
- Workflow creation API with request validation
- Batch-aware local model classification stub
- Tool routing and topic explanation utilities
- Workflow orchestration with telemetry history

## Tech Stack

- Python
- FastAPI
- Pydantic
- Pytest

## Run locally

```bash
cd AI-Agent-Dashboard-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

## Evaluation

```bash
pytest -q
```
