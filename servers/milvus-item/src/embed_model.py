from typing import Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from logger import trace_error


class LocalEmbedding(Embeddings):
    def __init__(
            self,
            base_url: str = "http://localhost:1234/v1",
            model: str = "text-embedding-multilingual-e5-large-instruct",
            api_key: str = "lm-studio",
            dimensions: Optional[int] = 1024,
    ):
        try:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model
            self.dimensions = dimensions
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