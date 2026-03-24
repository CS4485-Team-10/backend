from urllib.parse import quote, urlparse, urlunparse

from pydantic import field_validator
from pydantic_settings import BaseSettings


def _encode_password_in_url(url: str) -> str:
    """URL-encode password to avoid special chars (e.g. !) breaking psycopg2."""
    if not url:
        return url
    try:
        p = urlparse(url)
        if not p.password:
            return url
        encoded = quote(p.password, safe="")
        if encoded == p.password:
            return url
        netloc = f"{p.username}:{encoded}@{p.hostname}"
        if p.port:
            netloc += f":{p.port}"
        return urlunparse((p.scheme, netloc, p.path or "/", p.params or "", p.query or "", p.fragment or ""))
    except Exception:
        return url


class Settings(BaseSettings):
    ENV: str = "dev"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    DATABASE_URL: str = ""

    YOUTUBE_API_KEY: str = ""
    YOUTUBE_DATA_API_KEY: str = ""
    YOUTUBE_SEARCH_QUERY: str = ""

    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""

    @property
    def youtube_api_key(self) -> str:
        return self.YOUTUBE_API_KEY or self.YOUTUBE_DATA_API_KEY

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def encode_database_url_password(cls, v: str) -> str:
        return _encode_password_in_url(v)

    class Config:
        env_file = ".env"


settings = Settings()
