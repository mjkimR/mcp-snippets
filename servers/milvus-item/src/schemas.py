import uuid
from datetime import datetime
from typing import Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ValidationError

from src.logger import logger


class BoardCreate(BaseModel):
    category: str
    title: str
    contents: str
    owner: str


class BoardRead(BaseModel):
    uuid: uuid.UUID
    category: str
    title: str
    contents: str
    owner: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_dict(cls, entity):
        try:
            return cls(
                uuid=uuid.UUID(entity.get('uuid')),
                category=entity.get('category'),
                title=entity.get('title'),
                contents=entity.get('contents'),
                owner=entity.get('owner'),
                created_at=datetime.fromtimestamp(entity.get('created_at')),
                updated_at=datetime.fromtimestamp(entity.get('updated_at'))
            )
        except (TypeError, ValueError) as e:
            logger.error(e, exc_info=True)
            raise ValueError(f"Failed to create BoardRead from dict. Check entity data: {e}")


class BoardData(BaseModel):
    uuid: str
    category: str
    title: str
    contents: str
    owner: str
    created_at: int
    updated_at: int
    embedding: Optional[list[float]] = None

    @classmethod
    def from_create(cls, board_create: BoardCreate, embed_model: Embeddings):
        try:
            obj = cls(
                uuid=str(uuid.uuid4()),
                category=board_create.category,
                title=board_create.title,
                contents=board_create.contents,
                owner=board_create.owner,
                created_at=cls.datetime_to_timestamp(datetime.now()),
                updated_at=cls.datetime_to_timestamp(datetime.now()),
            )
            obj.embedding = cls.embedding_content(obj, embed_model)
            return obj
        except ValidationError as e:
            logger.error(e, exc_info=True)
            raise ValueError(f"Failed to create BoardData from BoardCreate. Invalid input: {e}")
        except Exception as e:
            logger.error(e, exc_info=True)
            raise Exception(f"Unexpected error creating BoardData: {e}")

    @staticmethod
    def embedding_content(obj, embed_model):
        try:
            return embed_model.embed_query(
                f"category: {obj.category} || title: {obj.title} || contents: {obj.contents}")
        except Exception as e:
            logger.error(e, exc_info=True)
            raise Exception(f"Failed to embed content: {e}")

    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        try:
            return int(dt.timestamp())
        except AttributeError as e:
            logger.error(e, exc_info=True)
            raise ValueError(f"Invalid datetime object provided: {e}")
