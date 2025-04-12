import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, AsyncIterator

from langchain_core.embeddings import Embeddings
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus.exceptions import MilvusException

from logger import logger, trace_error


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
            trace_error()
            raise ValueError(f"Failed to create BoardRead from dict. Check entity data.")


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
            trace_error()
            raise ValueError(f"Failed to create BoardData from BoardCreate. Invalid input.")
        except Exception as e:
            trace_error()
            raise Exception(f"Unexpected error creating BoardData.")

    @staticmethod
    def embedding_content(obj, embed_model):
        try:
            return embed_model.embed_query(
                f"category: {obj.category} || title: {obj.title} || contents: {obj.contents}")
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to embed content.")

    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        try:
            return int(dt.timestamp())
        except AttributeError as e:
            trace_error()
            raise ValueError(f"Invalid datetime object provided.")


class LocalEmbedding(Embeddings):
    def __init__(
            self,
            base_url="http://localhost:1234/v1",
            model="text-embedding-multilingual-e5-large-instruct",
            api_key="lm-studio"
    ):
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to initialize OpenAI client. Check configuration.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            texts = list(map(lambda text: text.replace("\n", " "), texts))
            datas = self.client.embeddings.create(input=texts, model=self.model).data
            return list(map(lambda data: data.embedding, datas))
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to embed documents.")

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.embed_documents([text])[0]
        except IndexError:
            trace_error()
            raise ValueError("No embedding returned for the query.")
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to embed query.")


class MilvusClient:
    def __init__(
            self,
            embedding_model: Embeddings,
            host="localhost", port="19530",
            vector_dim: int | None = None,
            collection_name: str = "board_collection",
            collection_description: str = "Board collection",
    ):
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection = None
        try:
            self.connect()
            self.init_collection()
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to initialize Milvus client.")

    def connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
        except Exception as e:
            trace_error()
            raise Exception(f"Failed to connect to Milvus server. Check host and port.")

    def release(self):
        try:
            if self.collection:
                self.collection.release()
        except Exception as e:
            raise Exception(f"Error releasing Milvus collection: {e}")

    def init_collection(self):
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                return

            # 필드 정의
            fields = [
                FieldSchema(name="uuid", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="contents", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="owner", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="updated_at", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
            ]

            schema = CollectionSchema(fields, self.collection_description)
            self.collection = Collection(self.collection_name, schema)

            # 인덱스 생성
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)

            # 필터링을 위한 인덱스
            self.collection.create_index("category", index_name="idx_category")
            self.collection.create_index("created_at", index_name="idx_created_at")
            self.collection.create_index("updated_at", index_name="idx_updated_at")
        except MilvusException as e:
            trace_error()
            raise Exception(f"Failed to initialize or create Milvus collection.")
        except Exception as e:
            trace_error()
            raise Exception(f"Unexpected error during Milvus collection initialization.")

    def create_board(self, board: BoardCreate) -> BoardRead:
        try:
            self.collection.load()
            board_data = BoardData.from_create(board, self.embedding_model)
            data = [board_data.model_dump()]
            self.collection.insert(data=data)
            self.collection.flush()
            return BoardRead.from_dict(data[0])
        except MilvusException as e:
            trace_error()
            raise Exception(f"Failed to create board in Milvus.")
        except Exception as e:
            trace_error()
            raise Exception(f"Unexpected error creating board.")

    def search_boards(
            self,
            query: str,
            category: Optional[str] = None,
            created_after: Optional[datetime] = None,
            created_before: Optional[datetime] = None,
            updated_after: Optional[datetime] = None,
            updated_before: Optional[datetime] = None,
            limit: int = 10
    ) -> list[BoardRead]:
        try:
            self.collection.load()

            # 필터 구성
            expr_parts = []
            if category:
                expr_parts.append(f'category == "{category}"')

            if created_after:
                created_after_ts = int(created_after.timestamp())
                expr_parts.append(f'created_at >= {created_after_ts}')

            if created_before:
                created_before_ts = int(created_before.timestamp())
                expr_parts.append(f'created_at <= {created_before_ts}')

            if updated_after:
                updated_after_ts = int(updated_after.timestamp())
                expr_parts.append(f'updated_at >= {updated_after_ts}')

            if updated_before:
                updated_before_ts = int(updated_before.timestamp())
                expr_parts.append(f'updated_at <= {updated_before_ts}')

            expr = " and ".join(expr_parts) if expr_parts else None
            output_fields = ["uuid", "category", "title", "contents", "owner", "created_at", "updated_at"]
            results = []

            query_embedding = self.embedding_model.embed_query(query)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            search_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields
            )

            for hits in search_results:
                for hit in hits:
                    entity = hit.entity
                    results.append(BoardRead.from_dict(entity))

            return results
        except MilvusException as e:
            trace_error()
            raise Exception(f"Failed to search boards in Milvus.")
        except Exception as e:
            trace_error()
            raise Exception(f"Unexpected error during board search.")


@dataclass
class AppContext:
    embed: Embeddings
    vdb: MilvusClient

    @staticmethod
    def from_context(ctx: Context) -> "AppContext":
        """Get AppContext from FastMCP context"""
        return ctx.request_context.lifespan_context


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    client = None
    try:
        embed_model = LocalEmbedding()
        client = MilvusClient(embed_model, vector_dim=1024)
        yield AppContext(embed=embed_model, vdb=client)
    except Exception as e:
        trace_error()
        raise Exception(f"Error during application lifespan: {e}")
    finally:
        client.release()


mcp = FastMCP("Board Manager", lifespan=app_lifespan)


@mcp.tool()
def create_board(
        ctx: Context,
        category: str,
        title: str,
        contents: str,
        owner: str = "admin"
):
    """Create a new board"""
    try:
        return AppContext.from_context(ctx).vdb.create_board(BoardCreate(
            category=category,
            title=title,
            contents=contents,
            owner=owner
        ))
    except ValidationError as e:
        trace_error()
        raise Exception(f"Invalid input for creating board.")
    except Exception as e:
        trace_error()
        raise Exception(f"Failed to create board.")


@mcp.tool()
def search_board(
        ctx: Context,
        query: str,
        category: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        updated_after: Optional[datetime] = None,
        updated_before: Optional[datetime] = None,
        limit: int = 10
):
    """Search boards"""
    try:
        result = AppContext.from_context(ctx).vdb.search_boards(
            query=query,
            category=category,
            created_after=created_after,
            created_before=created_before,
            updated_after=updated_after,
            updated_before=updated_before,
            limit=limit
        )
        return result
    except Exception as e:
        trace_error()
        raise Exception(f"Failed to search boards.")
