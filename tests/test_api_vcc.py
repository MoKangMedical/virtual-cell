"""REST API + VCC Pipeline HTTP integration tests."""

from fastapi.testclient import TestClient

from virtual_cell import __version__
from virtual_cell.api import app


client = TestClient(app)


def test_catalog_endpoints_return_platform_metadata():
    """Core catalog endpoints should expose the advertised platform metadata."""
    health = client.get("/health")
    assert health.status_code == 200
    health_payload = health.json()
    assert health_payload["status"] == "healthy"
    assert health_payload["version"] == __version__
    assert health_payload["models"] == 15
    assert health_payload["datasets"] == 26

    models = client.get("/models")
    assert models.status_code == 200
    models_payload = models.json()
    assert len(models_payload) == 15
    assert any(model["key"] == "scgpt" for model in models_payload)

    datasets = client.get("/datasets")
    assert datasets.status_code == 200
    datasets_payload = datasets.json()
    assert len(datasets_payload) == 26
    assert any(dataset["key"] == "kang2018" for dataset in datasets_payload)

    tasks = client.get("/tasks")
    assert tasks.status_code == 200
    task_names = {task["name"] for task in tasks.json()}
    assert task_names == {"cell_annotation", "perturbation", "integration", "grn"}

    stats = client.get("/api/v1/stats")
    assert stats.status_code == 200
    stats_payload = stats.json()
    assert stats_payload["platform"] == "VirtualCell"
    assert stats_payload["version"] == __version__
    assert stats_payload["n_models"] == 15
    assert stats_payload["n_datasets"] == 26
    assert stats_payload["n_generators"] == 2


def test_generation_benchmark_and_prediction_endpoints():
    """Generation, benchmarking, and prediction endpoints should return structured results."""
    generate = client.post(
        "/generate",
        json={
            "task": "perturbation",
            "dataset": "kang2018",
            "n_architectures": 2,
            "mode": "mock",
        },
    )
    assert generate.status_code == 200
    generate_payload = generate.json()
    assert len(generate_payload["architectures"]) == 2
    assert isinstance(generate_payload["task_analysis"], dict)
    assert isinstance(generate_payload["design_history"], list)

    benchmark = client.post(
        "/benchmark",
        json={
            "models": ["scgpt"],
            "datasets": ["kang2018"],
            "tasks": ["perturbation"],
            "max_cells": 12,
        },
    )
    assert benchmark.status_code == 200
    benchmark_payload = benchmark.json()
    assert benchmark_payload["n_results"] == 1
    assert len(benchmark_payload["leaderboard"]) == 1
    assert benchmark_payload["leaderboard"][0]["task"] == "perturbation"

    predict = client.post(
        "/predict",
        json={
            "model": "scgpt",
            "task": "cell_annotation",
            "dataset": "zheng68k",
            "n_cells": 10,
        },
    )
    assert predict.status_code == 200
    predict_payload = predict.json()
    assert predict_payload["model"] == "scGPT"
    assert predict_payload["task"] == "cell_annotation"
    assert predict_payload["n_predictions"] == 10
    assert predict_payload["metadata"]["mode"] == "mock"


def test_leaderboard_endpoints_return_rows_and_task_filters():
    """Leaderboard endpoints should return persisted rows and runtime task slices."""
    leaderboard = client.get("/leaderboard", params={"top_n": 5})
    assert leaderboard.status_code == 200
    leaderboard_payload = leaderboard.json()
    assert leaderboard_payload["task"] == "all"
    assert len(leaderboard_payload["leaderboard"]) == 5
    first_entry = leaderboard_payload["leaderboard"][0]
    assert {"model", "dataset", "task", "primary_score", "all_metrics"} <= first_entry.keys()

    filtered = client.get("/leaderboard", params={"task": "perturbation", "top_n": 3})
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert filtered_payload["task"] == "perturbation"
    assert len(filtered_payload["leaderboard"]) == 3
    assert all(entry["task"] == "perturbation" for entry in filtered_payload["leaderboard"])

    by_task = client.get("/api/v1/leaderboard/perturbation", params={"top_n": 4})
    assert by_task.status_code == 200
    by_task_payload = by_task.json()
    assert by_task_payload["task"] == "perturbation"
    assert 0 < len(by_task_payload["leaderboard"]) <= 4
    assert by_task_payload["total_entries"] >= len(by_task_payload["leaderboard"])
    assert all(entry["task"] == "perturbation" for entry in by_task_payload["leaderboard"])


def test_model_info_pipeline_compare_and_generator_endpoints():
    """Model detail and higher-level orchestration endpoints should stay callable via HTTP."""
    info = client.get("/info/scgpt")
    assert info.status_code == 200
    info_payload = info.json()
    assert info_payload["key"] == "scgpt"
    assert info_payload["name"] == "scGPT"

    detail = client.get("/api/v1/models/scgpt/detail")
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["key"] == "scgpt"
    assert detail_payload["name"] == "scGPT"
    assert "paper" in detail_payload
    assert "supported_tasks" in detail_payload

    pipeline = client.post(
        "/api/v1/pipeline/run",
        json={
            "task": "perturbation",
            "dataset": "kang2018",
            "n_architectures": 1,
            "max_cells": 12,
            "mode": "mock",
        },
    )
    assert pipeline.status_code == 200
    pipeline_payload = pipeline.json()
    assert pipeline_payload["status"] == "completed"
    assert pipeline_payload["task"] == "perturbation"
    assert pipeline_payload["dataset"] == "kang2018"
    assert pipeline_payload["n_architectures"] == 1
    assert len(pipeline_payload["leaderboard"]) == 1

    compare = client.post(
        "/api/v1/compare",
        json={
            "model1": "scgpt",
            "model2": "geneformer",
            "datasets": ["kang2018"],
            "tasks": ["perturbation"],
            "max_cells": 12,
        },
    )
    assert compare.status_code == 200
    compare_payload = compare.json()
    assert compare_payload["model1"]["name"] == "scgpt"
    assert compare_payload["model2"]["name"] == "geneformer"
    assert compare_payload["winner"] in {"scgpt", "geneformer", "tie"}
    assert compare_payload["tasks_compared"] == ["perturbation"]
    assert compare_payload["datasets_compared"] == ["kang2018"]

    generators = client.get("/api/v1/generators")
    assert generators.status_code == 200
    generators_payload = generators.json()
    assert generators_payload["total"] == 2
    assert {item["name"] for item in generators_payload["generators"]} == {
        "CellForge",
        "CellForgeFull",
    }


def test_invalid_requests_return_expected_http_errors():
    """Invalid model/task requests should surface clear client errors."""
    missing_model = client.get("/info/not-a-model")
    assert missing_model.status_code == 404

    missing_detail = client.get("/api/v1/models/not-a-model/detail")
    assert missing_detail.status_code == 404

    invalid_task = client.get("/api/v1/leaderboard/not-a-task")
    assert invalid_task.status_code == 400

    bad_compare = client.post(
        "/api/v1/compare",
        json={
            "model1": "not-a-model",
            "model2": "geneformer",
            "datasets": ["kang2018"],
            "tasks": ["perturbation"],
        },
    )
    assert bad_compare.status_code == 404

    bad_pipeline = client.post(
        "/api/v1/pipeline/run",
        json={
            "task": "not-a-task",
            "dataset": "kang2018",
            "n_architectures": 1,
            "max_cells": 10,
            "mode": "mock",
        },
    )
    assert bad_pipeline.status_code == 400
