import asyncio

from fastapi.testclient import TestClient

from app import LocalModel, TopicExplainerTool, ToolRegistry, app, orchestrator


def test_health_and_model_endpoints():
    with TestClient(app) as client:
        assert client.get('/api/health').json() == {'status': 'ok'}
        model = client.get('/api/model').json()
        assert {'model', 'device', 'batching'} <= model.keys()


def test_task_validation():
    with TestClient(app) as client:
        response = client.post('/api/workflows', json={'task': 'x'})
        assert response.status_code == 422


def test_tool_routing_and_topic_explainer():
    registry = ToolRegistry()
    assert registry.route('explain RAG').name == 'topic_explainer'
    assert registry.route('explain LangChain').name == 'topic_explainer'
    result = asyncio.run(TopicExplainerTool().run('explain RAG'))
    assert 'Retrieval-Augmented Generation' in result['answer']


def test_local_model_batches_inputs(monkeypatch):
    monkeypatch.setenv('BATCH_SIZE', '2')
    model = LocalModel()
    calls = []

    class FakePipeline:
        def __call__(self, texts, candidate_labels):
            calls.append(list(texts))
            return [{'labels': [candidate_labels[0]], 'scores': [0.9]} for _ in texts]

    model._pipeline = FakePipeline()
    results, metrics = asyncio.run(model.classify(['one', 'two', 'three', 'four', 'five'], ['innovation']))

    assert len(results) == 5
    assert [len(batch) for batch in calls] == [2, 2, 1]
    assert metrics['batch_size'] == 5
    assert metrics['batch_count'] == 3
    assert metrics['max_batch_size'] == 2


def test_workflow_orchestration_without_external_feed(monkeypatch):
    async def fake_news(query):
        return {'items': [{'title': 'AI research', 'summary': 'innovation', 'sentiment': 0.1}]}

    monkeypatch.setattr(orchestrator.tools.tools['news_search'], 'run', fake_news)

    async def run_workflow():
        workflow = await orchestrator.start('Summarize AI innovation')
        for _ in range(600):
            if workflow.status in {'completed', 'failed'}:
                return workflow
            await asyncio.sleep(0.01)
        return workflow

    workflow = asyncio.run(run_workflow())
    assert workflow.status == 'completed'
    assert any(event['type'] == 'task_decomposed' for event in orchestrator.telemetry.history)
