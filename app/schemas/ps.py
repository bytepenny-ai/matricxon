from pydantic import BaseModel


class PsEntry(BaseModel):
    name: str
    size: int
    size_vram: int
    expires_at: str
    pinned: bool = False  # kept loaded (app.models.pins); expires_at is then far away


class PsResponse(BaseModel):
    models: list[PsEntry]
