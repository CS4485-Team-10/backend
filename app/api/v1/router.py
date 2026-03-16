from fastapi import APIRouter

from app.api.v1.endpoints import (
    channels,
    health,
    ingest,
    insights,
    overview,
    supabase_ping,
    transcripts,
    videos,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(supabase_ping.router, tags=["supabase"])
api_router.include_router(overview.router, tags=["overview"])
api_router.include_router(videos.router, tags=["videos"])
api_router.include_router(channels.router, tags=["channels"])
api_router.include_router(insights.router, tags=["insights"])
api_router.include_router(transcripts.router, tags=["transcripts"])
api_router.include_router(ingest.router, tags=["ingest"])
