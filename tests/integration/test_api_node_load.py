"""GET /api/node/load: active replies, loaded models, threads and free memory for routing."""

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel


def test_idle_node_reports_nothing_active(client: TestClient) -> None:
    body = client.get("/api/node/load").json()

    assert body["active"] == 0 and body["models"] == []
    assert body["threads"] >= 1 and body["node_kind"] == "compute"
    assert body["free_memory_bytes"] is None or body["free_memory_bytes"] > 0


def test_a_loaded_model_is_listed_with_its_size(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    client.post("/api/generate", json={"model": "tiny-mistral3:latest"})  # load only

    models = client.get("/api/node/load").json()["models"]

    assert [(m["name"], m["active"]) for m in models] == [("tiny-mistral3:latest", 0)]
    assert models[0]["size_bytes"] > 0


def test_memory_and_pins_are_reported(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    assert (
        client.put(
            "/api/runtime-settings", json={"pinned_models": ["tiny-mistral3:latest"]}
        ).status_code
        == 200
    )
    body = client.get("/api/node/load").json()
    assert body["pins"] == [
        {
            "tag": "tiny-mistral3:latest",
            "memory": "auto",
            "state": "loading",
            "attempt": 0,
            "error": None,
        }
    ]
    assert body["max_loaded_models"] >= 1 and body["gpu"] is None
    assert body["total_memory_bytes"] is None or body["total_memory_bytes"] > 0


def test_a_pin_list_the_node_cannot_take_is_refused(client: TestClient) -> None:
    response = client.put("/api/runtime-settings", json={"pinned_models": ["not-installed:latest"]})
    assert response.status_code == 409 and "not installed" in response.json()["detail"]


def test_each_model_reports_when_its_keep_alive_unloads_it(
    client: TestClient, tiny_mistral3_model: InstalledModel
) -> None:
    client.post("/api/generate", json={"model": "tiny-mistral3:latest", "keep_alive": 600})
    model = client.get("/api/node/load").json()["models"][0]
    assert 590 <= model["expires_in_seconds"] <= 600 and model["pinned"] is False
