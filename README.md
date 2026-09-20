# AI Agent Dashboard

A small, inspectable full-stack dashboard for multi-agent workflows. The FastAPI backend exposes REST endpoints and native WebSocket telemetry; the browser UI is a React application served directly by FastAPI, so no Node build step is required.

## Implemented architecture

```text
React dashboard
  | REST + native WebSocket
FastAPI API
  |\n  | Orchestrator
  |-- Research Agent -> ToolRegistry -> topic explainer / TechCrunch feed tool
  |-- Analysis Agent -> local Hugging Face zero-shot classifier
  `-- Response Agent -> safe final response
```

A workflow is decomposed into research, analysis, and response subtasks. Tool failures emit a failure event, trigger a real retry using the submitted task as context, and allow the workflow to continue. Telemetry contains status and execution metadata only; private chain-of-thought is never streamed.

## Setup

Python 3.10+ is required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open `http://127.0.0.1:8000`. The first model-backed workflow may download the configured Hugging Face model. If download or model loading fails, the app reports `fallback: true` in inference telemetry and uses a deterministic label fallback so the dashboard remains demonstrable.

## Configuration

- `HOST`, `PORT`: FastAPI bind settings.
- `AI_DEVICE`: `auto`, `cpu`, or `cuda`. CUDA is selected only when requested or available.
- `MODEL_NAME`: Hugging Face model identifier.
- `QUANTIZATION`: set to `true` to apply PyTorch dynamic INT8 quantization to the Transformer on CPU. CUDA runs remain non-quantized because this implementation does not silently substitute an unsupported CUDA quantization path.
- `BATCH_SIZE`: maximum number of texts sent to one local inference call, default `4`.

## API

- `GET /api/health` returns service health.
- `GET /api/model` returns model name, device, CUDA availability, quantization state, and batching capability.
- `POST /api/workflows` with `{ "task": "..." }` starts a workflow and returns its ID.
- `GET /api/workflows/{workflow_id}` returns current status and final result.
- `WS /ws` streams structured events such as decomposition, agent status, tool selection, tool failure, self-correction, inference metrics, and completion.

## Inference and RAG limitations

Transformers and PyTorch are integrated behind `LocalModel`. Inference is lazy, CPU-compatible, and measures latency, batch size, batch count, device, and measured throughput. The abstraction chunks multi-item requests into configured batches. Individual dashboard workflows currently submit one analysis item, while the shared model interface is ready for concurrent or bulk analysis. CPU dynamic INT8 quantization is active only when explicitly enabled and successfully loaded; the runtime reports failures or CUDA limitations instead of claiming quantization.

The tool result is passed as explicit analysis context. A vector database, embeddings, LangChain, LCEL, and production RAG pipeline are not implemented in this compact version; they are extension points rather than claims about the current system. The current news tool uses feed retrieval plus keyword matching, not semantic vector search.

## Tests

```powershell
pytest -q
```

Tests cover health/model endpoints, Pydantic validation, topic-explanation routing, batched inference, and an end-to-end orchestration path with the external feed isolated.
