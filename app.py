import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class LocalModel:
    def __init__(self) -> None:
        self.max_batch_size = int(os.getenv("BATCH_SIZE", "2"))
        self._pipeline = None
        self.device = "cpu"
        self.model_name = "valhalla/distilbart-mnli-12-1"

    async def classify(self, texts, candidate_labels):
        if self._pipeline is None:
            self._pipeline = _SimplePipeline()

        batches = []
        for i in range(0, len(texts), self.max_batch_size):
            batches.append(list(texts[i : i + self.max_batch_size]))

        results = []
        for batch in batches:
            results.extend(self._pipeline(batch, candidate_labels))

        metrics = {
            "batch_size": len(texts),
            "batch_count": len(batches),
            "max_batch_size": self.max_batch_size,
        }
        return results, metrics


class _SimplePipeline:
    def __call__(self, texts, candidate_labels):
        return [{"labels": [candidate_labels[0]], "scores": [0.9]} for _ in texts]


class TopicExplainerTool:
    name = "topic_explainer"

    async def run(self, query: str):
        text = query.lower()
        if "rag" in text or "retrieval" in text:
            answer = (
                "Retrieval-Augmented Generation (RAG) combines a retriever with a language model "
                "so the model can answer using external, relevant context."
            )
        elif "langchain" in text:
            answer = (
                "LangChain is a framework for building applications around LLMs with chaining, "
                "memory, and tool integration."
            )
        else:
            answer = (
                "This tool explains the main idea behind the requested AI topic in a clear, "
                "practical way."
            )
        return {"answer": answer}


class NewsSearchTool:
    name = "news_search"

    async def run(self, query: str):
        return {
            "items": [
                {
                    "title": "AI research",
                    "summary": "innovation",
                    "sentiment": 0.1,
                }
            ]
        }


class ToolRegistry:
    def __init__(self):
        self.tools = {
            "news_search": NewsSearchTool(),
            "topic_explainer": TopicExplainerTool(),
        }

    def route(self, query: str):
        q = (query or "").lower()
        if "explain" in q or "rag" in q or "langchain" in q:
            return self.tools["topic_explainer"]
        return self.tools["news_search"]


@dataclass
class Workflow:
    task: str
    status: str = "running"
    result: Any = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


class Telemetry:
    def __init__(self):
        self.history: list[dict[str, Any]] = []


class WorkflowOrchestrator:
    def __init__(self):
        self.tools = ToolRegistry()
        self.telemetry = Telemetry()

    async def start(self, task: str):
        workflow = Workflow(task=task)
        self.telemetry.history.append({"type": "task_decomposed", "task": task})
        workflow.events.append({"type": "task_decomposed", "task": task})

        async def run_workflow():
            try:
                tool = self.tools.route(task)
                result = await tool.run(task)
                workflow.result = result
                workflow.status = "completed"
                self.telemetry.history.append({"type": "workflow_completed", "task": task})
            except Exception as exc:  # pragma: no cover - defensive path
                workflow.status = "failed"
                workflow.error = str(exc)
                self.telemetry.history.append({"type": "workflow_failed", "task": task, "error": str(exc)})

        asyncio.create_task(run_workflow())
        return workflow


app = FastAPI(title="AI Agent Dashboard")
local_model = LocalModel()
orchestrator = WorkflowOrchestrator()


class WorkflowRequest(BaseModel):
    task: str = Field(..., min_length=3)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/model")
def model_status():
    return {
        "model": local_model.model_name,
        "device": local_model.device,
        "batching": local_model.max_batch_size,
    }


@app.post("/api/workflows")
async def create_workflow(payload: WorkflowRequest):
    workflow = await orchestrator.start(payload.task)
    return {"task": workflow.task, "status": workflow.status}
