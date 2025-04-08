import uuid
from datetime import datetime
from typing import Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pydantic import BaseModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


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
        return cls(
            uuid=uuid.UUID(entity.get('uuid')),
            category=entity.get('category'),
            title=entity.get('title'),
            contents=entity.get('contents'),
            owner=entity.get('owner'),
            created_at=datetime.fromtimestamp(entity.get('created_at')),
            updated_at=datetime.fromtimestamp(entity.get('updated_at'))
        )


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

    @staticmethod
    def embedding_content(obj, embed_model):
        return embed_model.embed_query(f"category: {obj.category} || title: {obj.title} || contents: {obj.contents}")

    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> int:
        return int(dt.timestamp())


class MyEmbeddings(Embeddings):
    def __init__(
            self,
            base_url="http://localhost:1234/v1",
            model="text-embedding-multilingual-e5-large-instruct",
            api_key="lm-studio"
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = list(map(lambda text: text.replace("\n", " "), texts))
        datas = self.client.embeddings.create(input=texts, model=self.model).data
        return list(map(lambda data: data.embedding, datas))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


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
        self.connect()

        self.collection = None
        self.init_collection()

    def connect(self):
        connections.connect("default", host=self.host, port=self.port)

    def init_collection(self):
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

    def create_board(self, board: BoardCreate):
        self.collection.load()
        data = [BoardData.from_create(board, self.embedding_model).model_dump()]
        self.collection.insert(data=data)
        self.collection.flush()
        return board

    def search_boards(
            self,
            query: str,
            category: Optional[str] = None,
            created_after: Optional[datetime] = None,
            created_before: Optional[datetime] = None,
            updated_after: Optional[datetime] = None,
            updated_before: Optional[datetime] = None,
            limit: int = 10
    ):
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


if __name__ == "__main__":
    # 예시 사용법
    milvus_client = MilvusClient(MyEmbeddings(), port="47309", vector_dim=1024)
