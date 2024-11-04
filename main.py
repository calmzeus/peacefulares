from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters.base import Language
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# see https://github.com/lo-b/heavenlyhades.git
PROJECT_PATH = "/home/bram/projects/heavenlyhades/java/simple-api/"
QDRANT_COLLECTION_NAME = "simple-java-api"
# see https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/
VOYAGE_MODEL_NAME = "voyage-code-2"
EMBEDDING_BATCH_SIZE = 1


class LoaderContext:
    """
    Loader context which is setup using a particular strategy.

    After setup it knows which files to include for indexing.
    """

    def __init__(self, strategy: LoadStrategy) -> None:
        """
        Create a new context with the specified strategy.
        """

        print("LoaderContext:", type(strategy).__name__, "is set.")
        self._strategy = strategy

    @property
    def strategy(self) -> LoadStrategy:
        """
        Get the strategy set for this loader.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: LoadStrategy) -> None:
        """
        Set the loader context's strategy of how to load files.
        """

        self._strategy = strategy

    def determine_glob_pattern(self) -> str:
        """
        Determine the glob pattern for the specified context.
        """

        print("LoaderContext: determine glob ...")
        result = self._strategy.determine_glob()
        print("LoaderContext:", result)
        return result

    def filetypes_of_intereset(self) -> list[str]:
        print("LoaderContext: getting filetypes of intereset ... ")
        filetypes = self._strategy.determine_filetypes()
        print(
            "LoaderContext: intereseted in", *[f"'{ft}'" for ft in filetypes], "files"
        )
        return filetypes


class LoadStrategy(ABC):
    """
    LoadStrategy interface which defines operations to get information about
    files of interest for a particular programming language (+ framework).
    """

    @abstractmethod
    def determine_glob(self) -> str:
        """Return the glob pattern used when relevant files are loaded."""
        pass

    @abstractmethod
    def determine_filetypes(self) -> list[str]:
        """Return relevant file types for this load strategy."""
        pass


class JavaLoadStrategy(LoadStrategy):
    """Java-based load strategy."""

    def __init__(self):
        """Initialise a new Java-based load strategy."""

    # NOTE: KISS for now
    def determine_glob(self) -> str:
        return "**/src/main/**/[!.]*"

    def determine_filetypes(self) -> list[str]:
        return [".java", ".properties"]


def create_loader(
    language: Language, loader_context: LoaderContext, path: str
) -> GenericLoader:
    """Create new loader using a specific context."""

    return GenericLoader.from_filesystem(
        path,
        glob=loader_context.determine_glob_pattern(),
        suffixes=loader_context.filetypes_of_intereset(),
        # WARNING: Loading docs using the 'parser' below gives deprecation
        # warning. This is because treesitter package had to be downgraded
        # to a version in [0.21, 0.22) for loading to work.
        parser=LanguageParser(language),  # type: ignore[arg-type]
    )


def create_qdrant_store(
    collection_name: str, client: QdrantClient, embeddings: Embeddings
) -> VectorStore:
    """
    Create a Qdrant vector store using the client and embeddings. The
    collection is created if it does not exist yet.
    """

    sample_text = "69-420"  # sample text to determine embedding size
    embedding_size = len(embeddings.embed_query(sample_text))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=embeddings
    )


def main():
    assert load_dotenv(), ".env file exists and contains at least one variable"
    print("Hello from peacefulares!")
    print("Client: set strategy")
    loader_context = LoaderContext(JavaLoadStrategy())
    print("Client: create loader")
    loader = create_loader(Language.JAVA, loader_context, PROJECT_PATH)
    docs = loader.load()
    print("Client: loaded", len(docs), "documents with the following paths:")
    for doc in docs:
        print(" ", doc.metadata["source"])

    embeddings = VoyageAIEmbeddings(
        model=VOYAGE_MODEL_NAME, batch_size=EMBEDDING_BATCH_SIZE
    )

    # keep data in-memeroy; gets lost when the client is destroyed
    client = QdrantClient(":memory:")
    vector_store = create_qdrant_store(QDRANT_COLLECTION_NAME, client, embeddings)

    if (
        client.collection_exists(QDRANT_COLLECTION_NAME)
        and client.get_collection(QDRANT_COLLECTION_NAME).points_count == 0
    ):
        print("Client: adding all documents to collection ...")
        uuids = [str(uuid4()) for _ in range(len(docs))]
        v_uuids = vector_store.add_documents(documents=docs, ids=uuids)
        print(v_uuids)


if __name__ == "__main__":
    main()
