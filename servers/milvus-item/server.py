import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, AsyncIterator

from langchain_core.embeddings import Embeddings
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from pydantic import ValidationError
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus.exceptions import MilvusException

from src.embed_model import LocalEmbedding
from src.exceptions import NonRetryableError, ExternalServiceUnavailableError
from src.logger import logger
from src.schemas import BoardCreate, BoardRead, BoardData


class MilvusClient:
    def __init__(
            self,
            embedding_model: Embeddings,
            host="localhost", port="19530",
            collection_name: str = "board_collection",
            collection_description: str = "Board collection",
    ):
        self.host = host
        self.port = port
        self.embedding_model = embedding_model
        self.vector_dim = self._get_vector_dimension()
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection = None
        try:
            self.connect()
            self.init_collection()
        except Exception as e:
            logger.error(e, exc_info=True)
            raise Exception(f"Failed to initialize Milvus client.")

    def _get_vector_dimension(self):
        if hasattr(self.embedding_model, "dimensions"):
            return self.embedding_model.dimensions
        else:
            return len(self.embedding_model.embed_query("test"))

    def connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise ExternalServiceUnavailableError(f"Failed to connect to Milvus server. Check host and port.")

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
            logger.error(e, exc_info=True)
            raise Exception(f"Failed to initialize or create Milvus collection.")
        except Exception as e:
            logger.error(e, exc_info=True)
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
            logger.error(e, exc_info=True)
            raise Exception(f"Failed to create board in Milvus.")
        except Exception as e:
            logger.error(e, exc_info=True)
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
            logger.error(e, exc_info=True)
            raise Exception(f"Failed to search boards in Milvus.")
        except Exception as e:
            logger.error(e, exc_info=True)
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
        embed_model = LocalEmbedding(
            model="text-embedding-multilingual-e5-large-instruct",
            dimensions=1024,
        )
        client = MilvusClient(embed_model)
        yield AppContext(embed=embed_model, vdb=client)
    except Exception as e:
        logger.error(e, exc_info=True)
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
    except NonRetryableError as e:
        error_message = f"[NON-RETRYABLE] {str(e)}"
        logger.error(error_message, exc_info=True)
        raise Exception(error_message)
    except ValidationError as e:
        logger.error(e, exc_info=True)
        raise Exception(f"Invalid input for creating board.")
    except Exception as e:
        logger.error(e, exc_info=True)
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
    except NonRetryableError as e:
        error_message = f"[NON-RETRYABLE] {str(e)}"
        logger.error(error_message, exc_info=True)
        raise Exception(error_message)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise Exception(f"Failed to search boards.")
