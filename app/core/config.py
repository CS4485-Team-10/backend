from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: str = "dev"
    PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:3000"

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    YOUTUBE_API_KEY: str = ""
    # DS notebook uses YOUTUBE_DATA_API_KEY; either key can be used for ingestion
    YOUTUBE_DATA_API_KEY: str = ""

    @property
    def youtube_api_key(self) -> str:
        return self.YOUTUBE_API_KEY or self.YOUTUBE_DATA_API_KEY

    class Config:
        env_file = ".env"


settings = Settings()
