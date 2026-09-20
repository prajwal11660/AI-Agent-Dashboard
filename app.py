"""AI Agent Dashboard backend."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import feedparser
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from textblob import TextBlob

ROOT = os.path.dirname(os.path.abspath(__file__))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRequest(BaseModel):
    task: str = Field(min_length=3, max_length=2000)


class WorkflowSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    workflow_id: str
    task: str
    status: str
    result: str | None = None
    created_at: str
    updated_at: str


class TelemetryHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.history: list[dict[str, Any]] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)
        for event in self.history[-100:]:
            await websocket.send_json(event)

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def publish(self, event_type: str, workflow_id: str, **data: Any) -> None:
        event = {"timestamp": utc_now(), "type": event_type, "workflow_id": workflow_id, **data}
        self.history.append(event)
        self.history = self.history[-500:]
        disconnected: list[WebSocket] = []
        for client in self.clients:
            try:
                await client.send_json(event)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.disconnect(client)


class ToolError(RuntimeError):
    pass


class Tool:
    name = "tool"

    async def run(self, query: str) -> dict[str, Any]:
        raise NotImplementedError


class TopicExplainerTool(Tool):
    name = "topic_explainer"
    topics = {
        "rag": "Retrieval-Augmented Generation retrieves relevant documents, adds them as context, and then asks a language model to generate a grounded answer. This dashboard does not implement a RAG pipeline yet.",
        "websocket": "WebSockets keep one live connection open so the backend can stream workflow events to the dashboard as agents and tools run.",
        "transformer": "A Transformer is a neural network architecture that uses attention to process relationships between tokens in text.",
        "quantization": "Quantization reduces model weight precision to reduce memory usage. This project supports optional CPU dynamic INT8 quantization when configured.",
        "langchain": "LangChain is a framework for composing LLM calls, prompts, tools, and retrieval workflows. This dashboard does not currently use LangChain; its orchestration is implemented directly in Python.",
        "fastapi": "FastAPI is the Python web framework exposing the REST endpoints and WebSocket connection used by this dashboard.",
        "pytorch": "PyTorch provides the local model runtime, device detection, and optional CPU dynamic quantization used by this dashboard.",
        "agent": "An agent is a role in the workflow with a focused responsibility. This project uses Research, Analysis, and Response agents.",
        "batching": "Batching processes multiple model inputs together to reduce repeated inference overhead. The local model interface supports configured batch sizes.",
    }

    async def run(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        for topic, answer in self.topics.items():
            if topic in lowered:
                return {"topic": topic, "answer": answer}
        return {"topic": "general", "answer": "The topic explainer has no verified local explanation for this request."}


class NewsTool(Tool):
    name = "news_search"
    feed_url = "https://techcrunch.com/feed/"

    async def run(self, query: str) -> dict[str, Any]:
        feed = await asyncio.to_thread(feedparser.parse, self.feed_url)
        stop_words = {"find", "show", "get", "recent", "latest", "news", "about", "the", "and", "please"}
        query_terms = [word for word in re.findall(r"[a-z0-9]+", query.lower()) if word not in stop_words]
        entries = []
        for entry in feed.entries[:8]:
            text = f"{entry.get('title', '')}. {entry.get('summary', '')}"
            if not query_terms or any(word in text.lower() for word in query_terms):
                entries.append({
                    "title": entry.get("title", "Untitled"),
                    "summary": re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:240],
                    "sentiment": round(TextBlob(text).sentiment.polarity, 3),
                })
        if not entries:
            entries = [{
                "title": entry.get("title", "Untitled"),
                "summary": re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:240],
                "sentiment": 0,
            } for entry in feed.entries[:5]]
        if not entries:
            raise ToolError("The news feed returned no entries")
        return {"source": self.feed_url, "items": entries[:5]}


class ToolRegistry:
    def __init__(self) -> None:
        self.tools = {tool.name: tool for tool in (TopicExplainerTool(), NewsTool())}

    def route(self, task: str) -> Tool:
        topic_terms = ("rag", "websocket", "transformer", "quantization", "langchain", "fastapi", "pytorch", "agent", "batching")
        if any(term in task.lower() for term in topic_terms):
            return self.tools["topic_explainer"]
        return self.tools["news_search"]


class LocalModel:
    """Lazy Hugging Face inference with measured metrics and safe fallback."""

    def __init__(self) -> None:
        requested = os.getenv("AI_DEVICE", "auto").lower()
        self.device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
        self.model_name = os.getenv("MODEL_NAME", "valhalla/distilbart-mnli-12-1")
        self.quantization_requested = os.getenv("QUANTIZATION", "false").lower() == "true"
        self.max_batch_size = max(1, int(os.getenv("BATCH_SIZE", "4")))
        self.quantized = False
        self.quantization_error: str | None = None
        self._pipeline: Any = None
        self.load_error: str | None = None

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "quantized": self.quantized,
            "quantization_requested": self.quantization_requested,
            "batching": True,
            "max_batch_size": self.max_batch_size,
            "quantization_error": self.quantization_error,
            "status": "lazy",
        }

    def _load(self) -> None:
        if self._pipeline is not None or self.load_error:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            if self.quantization_requested and self.device == "cpu":
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                self._pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=-1)
                self.quantized = True
            else:
                if self.quantization_requested and self.device == "cuda":
                    self.quantization_error = "CUDA quantization is not enabled; using non-quantized CUDA inference"
                self._pipeline = pipeline("zero-shot-classification", model=self.model_name, device=0 if self.device == "cuda" else -1)
        except Exception as exc:
            self.load_error = str(exc)

    async def classify(self, texts: list[str], labels: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()
        await asyncio.to_thread(self._load)
        batches = [texts[index:index + self.max_batch_size] for index in range(0, len(texts), self.max_batch_size)]
        if self._pipeline is not None:
            results = []
            for batch in batches:
                batch_results = await asyncio.to_thread(self._pipeline, batch, candidate_labels=labels)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
        else:
            results = []
            for text in texts:
                lowered = text.lower()
                scores = {label: sum(word in lowered for word in label.split()) for label in labels}
                top = max(scores, key=scores.get)
                results.append({"labels": [top], "scores": [0.5 if scores[top] else 0.34]})
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return results, {
            "batch_size": len(texts),
            "batch_count": len(batches),
            "max_batch_size": self.max_batch_size,
            "latency_ms": latency_ms,
            "throughput_per_second": round(len(texts) / max(latency_ms / 1000, 0.001), 2),
            "device": self.device,
            "model": self.model_name,
            "quantized": self.quantized,
            "fallback": self._pipeline is None,
            "quantization_error": self.quantization_error,
        }


@dataclass
class Workflow:
    workflow_id: str
    task: str
    status: str = "queued"
    result: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


class Orchestrator:
    def __init__(self, telemetry: TelemetryHub) -> None:
        self.telemetry = telemetry
        self.tools = ToolRegistry()
        self.model = LocalModel()
        self.workflows: dict[str, Workflow] = {}

    async def event(self, event_type: str, workflow_id: str, **data: Any) -> None:
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.updated_at = utc_now()
        await self.telemetry.publish(event_type, workflow_id, **data)

    async def start(self, task: str) -> Workflow:
        workflow = Workflow(str(uuid.uuid4()), task)
        self.workflows[workflow.workflow_id] = workflow
        asyncio.create_task(self.run(workflow))
        return workflow

    async def run(self, workflow: Workflow) -> None:
        wid = workflow.workflow_id
        try:
            workflow.status = "running"
            await self.event("workflow_started", wid, task=workflow.task, status="running")
            subtasks = [
                {"id": "research", "agent": "Research Agent", "status": "pending", "task": "Gather relevant information"},
                {"id": "analysis", "agent": "Analysis Agent", "status": "pending", "task": "Analyze retrieved information"},
                {"id": "response", "agent": "Response Agent", "status": "pending", "task": "Synthesize a concise answer"},
            ]
            await self.event("task_decomposed", wid, task=workflow.task, subtasks=subtasks)
            tool = self.tools.route(workflow.task)
            await self.event("tool_selected", wid, agent="Research Agent", tool=tool.name, status="selected")
            await self.event("agent_started", wid, agent="Research Agent", status="running")
            await self.event("tool_started", wid, agent="Research Agent", tool=tool.name, status="running")
            try:
                tool_result = await tool.run(workflow.task)
                await self.event("tool_completed", wid, agent="Research Agent", tool=tool.name, status="completed", output=tool_result)
            except ToolError as exc:
                await self.event("tool_failed", wid, agent="Research Agent", tool=tool.name, status="failed", error=str(exc))
                await self.event("self_correction_triggered", wid, agent="Research Agent", reason=str(exc), retry="using task context")
                tool_result = {"source": "fallback", "answer": "No external evidence was available for this workflow."}
            await self.event("agent_completed", wid, agent="Research Agent", status="completed")
            context = str(tool_result)
            await self.event("agent_started", wid, agent="Analysis Agent", status="running")
            labels = ["investment", "risk", "innovation"]
            if tool.name == "topic_explainer":
                label = "informational"
                metrics = {"batch_size": 0, "batch_count": 0, "latency_ms": 0, "throughput_per_second": 0, "device": self.model.device, "model": self.model.model_name, "quantized": self.model.quantized, "skipped": True, "reason": "verified topic explanation did not require classification"}
                await self.event("model_inference_started", wid, agent="Analysis Agent", status="skipped", reason=metrics["reason"])
                await self.event("model_inference_completed", wid, agent="Analysis Agent", status="skipped", metrics=metrics, label=label)
            else:
                await self.event("model_inference_started", wid, agent="Analysis Agent", batch_size=1)
                results, metrics = await self.model.classify([f"Task: {workflow.task}\nContext: {context}"], labels)
                label = results[0]["labels"][0]
                await self.event("model_inference_completed", wid, agent="Analysis Agent", status="completed", metrics=metrics, label=label)
            await self.event("agent_completed", wid, agent="Analysis Agent", status="completed")
            await self.event("agent_started", wid, agent="Response Agent", status="running")
            if tool_result.get("source") == "fallback":
                answer = f"I could not retrieve external evidence for this request. {tool_result['answer']}"
            elif tool.name == "topic_explainer":
                answer = f"{tool_result['answer']} Analysis used the verified topic explainer; model classification was skipped for speed."
            elif tool.name == "news_search":
                headlines = [item.get("title", "Untitled") for item in tool_result.get("items", [])[:3]]
                evidence = " ".join(f"{index + 1}. {headline}" for index, headline in enumerate(headlines))
                answer = f"Main trend: {label}. Recent retrieved headlines: {evidence or 'No headline text was returned.'}"
            else:
                answer = "The selected tool returned no user-facing answer."
            if tool.name != "topic_explainer":
                answer += f" Research used {tool.name}; analysis ran on {metrics['device']} with a batch size of {metrics['batch_size']}."
            else:
                answer += f" Research used {tool.name}."
            workflow.result = answer
            await self.event("agent_completed", wid, agent="Response Agent", status="completed")
            await self.event("final_response_generated", wid, agent="Response Agent", status="completed", response=answer)
            workflow.status = "completed"
            await self.event("workflow_completed", wid, status="completed", result=answer)
        except Exception as exc:
            workflow.status = "failed"
            await self.event("workflow_failed", wid, status="failed", error=str(exc))


telemetry = TelemetryHub()
orchestrator = Orchestrator(telemetry)
app = FastAPI(title="AI Agent Dashboard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(os.path.join(ROOT, "templates", "index.html"))


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model")
async def model_info() -> dict[str, Any]:
    return orchestrator.model.info()


@app.post("/api/workflows", response_model=WorkflowSummary, status_code=202)
async def create_workflow(request: TaskRequest) -> Workflow:
    return await orchestrator.start(request.task)


@app.get("/api/workflows/{workflow_id}", response_model=WorkflowSummary)
async def get_workflow(workflow_id: str) -> Workflow:
    workflow = orchestrator.workflows.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await telemetry.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
