from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser

PROJECT_PATH = "/home/bram/projects/heavenlyhades/java/simple-api/"


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


def create_loader(path: str, loader_context: LoaderContext) -> GenericLoader:
    """Create new loader using a specific context."""

    return GenericLoader.from_filesystem(
        path,
        glob=loader_context.determine_glob_pattern(),
        suffixes=loader_context.filetypes_of_intereset(),
        # WARNING: Loading docs using the 'parser' below gives deprecation
        # warning. This is because treesitter package had to be downgraded
        # to a version in [0.21, 0.21) for loading to work.
        parser=LanguageParser(language="java"),
    )


def main():
    print("Hello from peacefulares!")
    print("Client: set strategy")
    loader_context = LoaderContext(JavaLoadStrategy())
    print("Client: create loader")
    loader = create_loader(PROJECT_PATH, loader_context)
    docs = loader.load()
    print("Client: loaded", len(docs), "documents with the following paths:")
    for doc in docs:
        print(" ", doc.metadata["source"])


if __name__ == "__main__":
    main()
