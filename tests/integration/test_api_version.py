from fastapi.testclient import TestClient

from app.config import MATRICXON_VERSION


def test_returns_the_version_string(client: TestClient) -> None:
    response = client.get("/api/version")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == MATRICXON_VERSION
    # The git commit of a checkout, or None for a copy without .git.
    assert body["commit"] is None or isinstance(body["commit"], str)
