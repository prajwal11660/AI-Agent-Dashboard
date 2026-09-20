# AI Agent Dashboard

A full-stack AI agent orchestration dashboard built with Python, FastAPI, React, PyTorch, Hugging Face Transformers, and native WebSockets. It turns a high-level request into a visible workflow instead of hiding every operation behind one model call.

The browser UI is served directly by FastAPI, so no Node build step is required.

## Overview

```text
User task
    |
    v
React dashboard <---- REST + WebSocket telemetry ----> FastAPI
    |
    v
Workflow orchestrator
    |-- Research Agent  -> ToolRegistry -> topic explainer / news search
    |-- Analysis Agent  -> local Hugging Face zero-shot classifier
    `-- Response Agent  -> concise final response
```

The workflow is decomposed into research, analysis, and response stages. The dashboard exposes agent status, tool routing, model metrics, failures, recovery, and the final response while the workflow is running.

## Features

### Multi-agent orchestration

- **Research Agent** gathers context and selects a tool.
- **Analysis Agent** classifies the task and retrieved context.
- **Response Agent** combines the available context into a user-facing result.
- Structured telemetry makes each stage observable without streaming private chain-of-thought.

### Modular tools

`ToolRegistry` keeps tools separate from orchestration logic:

- **Topic Explainer** provides verified local explanations for FastAPI, PyTorch, Transformers, WebSockets, AI agents, batching, quantization, RAG, and LangChain.
- **News Search** reads the TechCrunch RSS feed, applies lightweight keyword matching, and reports basic sentiment scores.

### Local model runtime

The analysis stage lazily loads `valhalla/distilbart-mnli-12-1` through Hugging Face Transformers and PyTorch. The runtime supports:

- CPU or CUDA device selection through `AI_DEVICE`.
- Configurable inference batching through `BATCH_SIZE`.
- Optional CPU dynamic INT8 quantization through `QUANTIZATION=true`.
- Measured latency, batch count, throughput, model, device, and quantization state.
- Deterministic fallback classification when the model cannot be downloaded or loaded.

### Failure recovery

Tool failures emit `tool_failed` and `self_correction_triggered` events. The workflow continues with fallback context where possible instead of terminating immediately.

### Real-time telemetry

The WebSocket stream includes events such as:

```text
workflow_started
task_decomposed
tool_selected
tool_started
tool_completed
tool_failed
self_correction_triggered
agent_started
agent_completed
model_inference_started
model_inference_completed
final_response_generated
workflow_completed
```

## Setup

Python 3.10+ is required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open `http://127.0.0.1:8000`. The first model-backed workflow may download the configured Hugging Face model.

## Configuration

Create a `.env` file from `.env.example`:

```dotenv
HOST=127.0.0.1
PORT=8000
AI_DEVICE=auto
MODEL_NAME=valhalla/distilbart-mnli-12-1
QUANTIZATION=false
BATCH_SIZE=4
```

`AI_DEVICE` accepts `auto`, `cpu`, or `cuda`. CUDA is selected when requested or available. Quantization is applied only to CPU inference and only when it loads successfully; CUDA remains non-quantized in this implementation.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Returns service health. |
| `GET` | `/api/model` | Returns model, device, CUDA, quantization, and batching information. |
| `POST` | `/api/workflows` | Starts a workflow. Body: `{ "task": "Analyze the latest AI innovation news" }`. |
| `GET` | `/api/workflows/{workflow_id}` | Returns workflow status, timestamps, and final result. |
| `WS` | `/ws` | Streams structured workflow events. |

## Project structure

```text
AI-Agent-Dashboard/
|-- app.py
|-- requirements.txt
|-- .env.example
|-- templates/
|   `-- index.html
|-- static/
|   `-- style.css
`-- tests/
    `-- test_app.py
```

## Testing

```powershell
pytest -q
```

The tests cover health and model endpoints, request validation, topic-explanation routing, batched inference, workflow orchestration, and external feed isolation.

## Current scope and extension points

The news tool uses RSS retrieval, keyword matching, and lightweight sentiment analysis. Embeddings, vector databases, semantic retrieval, LangChain/LCEL, persistent workflow storage, authentication, parallel agent execution, and production deployment are not implemented. They are intended extension points for a larger system.

## Design principles

- **Modularity:** tools, inference, orchestration, telemetry, and presentation remain independently extensible.
- **Observability:** execution metadata is available through REST and WebSockets.
- **Local inference:** the project demonstrates CPU/GPU-aware Transformer execution through PyTorch.
- **Safe failure handling:** external failures are represented as structured events with recovery where possible.
- **Extensibility:** additional agents, tools, retrieval systems, and model providers can be added without rewriting the workflow core.
