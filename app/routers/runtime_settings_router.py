from fastapi import APIRouter, Depends, HTTPException

from app.cluster.provisioning import EnvFile
from app.config import settings
from app.dependencies import get_env_file, get_model_manager, get_model_pins
from app.models.manager import ModelManager
from app.models.pins import ModelPins, PinLimitError
from app.schemas.runtime_settings import RuntimeSettingsResult, RuntimeSettingsUpdate
from app.server.admin_auth import require_admin
from app.server.runtime_settings import RuntimeSettingsApplier

router = APIRouter()


@router.put(
    "/api/runtime-settings",
    dependencies=[Depends(require_admin)],
    response_model=RuntimeSettingsResult,
)
def put_runtime_settings(
    change: RuntimeSettingsUpdate,
    manager: ModelManager = Depends(get_model_manager),
    pins: ModelPins = Depends(get_model_pins),
    env_file: EnvFile = Depends(get_env_file),
) -> RuntimeSettingsResult:
    """Changes settings without a restart; signed like the other admin routes. A pin list this node
    can't take (too many chat models, not installed) is refused with 409 and nothing changes."""
    try:
        return RuntimeSettingsApplier(manager, settings, pins, env_file).apply(change)
    except PinLimitError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
