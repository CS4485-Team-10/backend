from fastapi import APIRouter
from supabase import create_client

from app.core.config import settings

router = APIRouter()


@router.get("/supabase/ping")
def supabase_ping():
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        return {"ok": False, "error": "Supabase env vars not set"}

    try:
        supabase = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
        )
        # Simple query: use a table you have (e.g. videos).
        # Create it in Supabase if missing.
        res = supabase.table("videos").select("video_id").limit(1).execute()
        return {"ok": True, "data": res.data}
    except Exception as e:
        return {"ok": False, "error": str(e)}
