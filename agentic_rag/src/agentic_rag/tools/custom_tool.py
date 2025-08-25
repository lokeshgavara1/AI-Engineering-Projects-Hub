import os
import requests
from typing import Type, List
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")


class FireCrawlWebSearchTool(BaseModel):
    api_key: str = Field(..., description="FireCrawl API key")

    def __init__(self, api_key: str = None, **data):
        # If api_key is not passed, try from environment variable
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")

        if not api_key:
            raise ValueError("❌ FireCrawl API key not provided. "
                             "Set FIRECRAWL_API_KEY in your environment.")

        super().__init__(api_key=api_key, **data)

    def search(self, query: str):
        url = f"https://api.firecrawl.dev/search?q={query}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the uploaded document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """Initialize with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")  # ✅ In-memory Qdrant

        # ✅ Use lightweight sentence transformer for embeddings
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # ✅ Create collection in Qdrant
        self.client.recreate_collection(
            collection_name="demo_collection",
            vectors_config=qmodels.VectorParams(
                size=self.embedding_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

        # Process document and insert into vector DB
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> List[str]:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",  # just for chunking
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )
        chunks = chunker.chunk(raw_text)
        return [chunk.text for chunk in chunks]

    def _process_document(self):
        """Process document: chunk text, create embeddings, and insert into Qdrant."""
        raw_text = self._extract_text()
        docs = self._create_chunks(raw_text)

        # Create embeddings
        vectors = self.embedder.encode(docs, convert_to_numpy=True)

        # Insert into Qdrant
        self.client.upsert(
            collection_name="demo_collection",
            points=[
                qmodels.PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"text": docs[i], "source": os.path.basename(self.file_path)},
                )
                for i in range(len(docs))
            ],
        )

    def _run(self, query: str) -> str:
        """Search Qdrant collection with query string."""
        query_vector = self.embedder.encode(query).tolist()

        search_result = self.client.search(
            collection_name="demo_collection",
            query_vector=query_vector,
            limit=3,
        )

        docs = [hit.payload["text"] for hit in search_result]
        if not docs:
            return "No relevant results found."

        separator = "\n---\n"
        return separator.join(docs)


# ✅ Test Script
def test_tools():
    # 🔹 FireCrawl test
    try:
        firecrawl = FireCrawlWebSearchTool()
        print("🔥 FireCrawl search:", firecrawl.search("AI news"))
    except ValueError as e:
        print(e)

    # 🔹 Document search test
    pdf_path = "C:/Users/lokes/Documents/PROJECTS/ai-engineering-hub-main/agentic_rag/knowledge/dspy.pdf"
    searcher = DocumentSearchTool(file_path=pdf_path)
    result = searcher._run("What is the purpose of DSpy?")
    print("🔍 Document Search Results:\n", result)


if __name__ == "__main__":
    test_tools()
