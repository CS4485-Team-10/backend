from fastapi import APIRouter
from supabase import create_client

from app.core.config import settings

router = APIRouter()


@router.get("/videos")
def list_videos():
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        return {"error": "Supabase env vars not set"}
    supabase = create_client(
        settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
    )
    res = supabase.table("videos").select("*").execute()
    return {"ok": True, "data": res.data, "count": len(res.data)}
