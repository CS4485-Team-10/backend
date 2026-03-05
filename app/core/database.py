from sqlmodel import Session, create_engine

from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, echo=(settings.ENV == "dev"))


def get_session():
    with Session(engine) as session:
        yield session
