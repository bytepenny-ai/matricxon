from collections.abc import Callable

from fastapi import APIRouter, Depends, Response

from app.cluster.provisioning import EnvFile
from app.config import settings
from app.consensus.runtime import ConsensusRuntime
from app.dependencies import (
    get_consensus_runtime,
    get_env_file,
    get_model_catalog,
    get_model_manager,
    get_model_pins,
)
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.models.pins import ModelPins, split_pin
from app.routers.pull_router import require_pull_lease
from app.schemas.delete import DeleteRequest
from app.schemas.runtime_settings import RuntimeSettingsUpdate
from app.server.admin_auth import require_admin
from app.server.node_kind import require_compute
from app.server.runtime_settings import RuntimeSettingsApplier

router = APIRouter()


class DeleteRequestHandler:
    """Unloads the model first (if loaded) before touching its files: an open `mmap` can't always be
    cleanly deleted while a `ModelWorker` thread holds tensors backed by it."""

    def __init__(
        self,
        catalog: ModelCatalog,
        model_manager: ModelManager,
        unpin: Callable[[str], None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._model_manager = model_manager
        self._unpin = unpin

    def handle(self, request: DeleteRequest) -> None:
        self._catalog.check_deletable(request.model)
        self._model_manager.unload(request.model)
        self._catalog.delete(request.model)
        if self._unpin is not None:
            self._unpin(request.model)  # its pin would only fail at every start


class PinRemover:
    """Drops a deleted model's pin on this node, in memory and .env (as PUT runtime-settings)."""

    def __init__(self, applier: RuntimeSettingsApplier, pins: ModelPins) -> None:
        self._applier = applier
        self._pins = pins

    def __call__(self, tag: str) -> None:
        kept = [entry for entry in self._pins.tags() if split_pin(entry)[0] != tag]
        if len(kept) < len(self._pins.tags()):
            self._applier.apply(RuntimeSettingsUpdate(pinned_models=kept))


@router.delete(
    "/api/delete",
    dependencies=[Depends(require_admin), Depends(require_compute)],
    status_code=200,
)
def delete_model(
    request: DeleteRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    model_manager: ModelManager = Depends(get_model_manager),
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
    pins: ModelPins = Depends(get_model_pins),
    env_file: EnvFile = Depends(get_env_file),
) -> Response:
    # The models folder is shared: removing a model changes it like pulling one, so only the
    # pull-lease owner does it (which also needs a quorum).
    require_pull_lease(runtime)
    unpin = PinRemover(RuntimeSettingsApplier(model_manager, settings, pins, env_file), pins)
    DeleteRequestHandler(catalog, model_manager, unpin).handle(request)
    return Response(status_code=200)
