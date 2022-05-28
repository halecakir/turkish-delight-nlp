
from fastapi import APIRouter

from turkishdelightnlp.common.heartbeat import HeartbeatResult

router = APIRouter()


@router.get("/healthz", response_model=HeartbeatResult, name="healthz")
def get_hearbeat() -> HeartbeatResult:
    heartbeat = HeartbeatResult(is_alive=True)
    return heartbeat
