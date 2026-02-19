from fastapi import APIRouter

from app.api.v1.endpoints import health, supabase_ping, videos

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(supabase_ping.router, tags=["supabase"])
api_router.include_router(videos.router, tags=["videos"])
