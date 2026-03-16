from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "dev"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    DATABASE_URL: str = ""

    YOUTUBE_DATA_API_KEY: str = ""
    YOUTUBE_SEARCH_QUERY: str = ""

    # LLM-API Provisioning
    LLM_PROVIDER: str = ""
    LLM_MODEL: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
