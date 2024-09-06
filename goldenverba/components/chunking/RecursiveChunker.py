import contextlib

with contextlib.suppress(Exception):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from goldenverba.components.chunk import Chunk
from goldenverba.components.interfaces import Chunker
from goldenverba.components.document import Document
from goldenverba.components.types import InputConfig
from goldenverba.components.interfaces import Embedding


class RecursiveChunker(Chunker):
    """
    RecursiveChunker for Verba using LangChain.
    """

    def __init__(self):
        super().__init__()
        self.name = "Recursive"
        self.requires_library = ["langchain_text_splitters"]
        self.description = "Split CSV documents by newline characters"
        self.config = {
            "Seperators": InputConfig(
                type="multi",
                value="\n",
                description="Separator to split the text (newline for CSV)",
                values=["\n"],
            ),
        }
        # Remove "Chunk Size" and "Overlap" from config

    async def chunk(
        self,
        config: dict,
        documents: list[Document],
        embedder: Embedding | None = None,
        embedder_config: dict | None = None,
    ) -> list[Document]:
        seperator = config["Seperators"].value

        for document in documents:
            if len(document.chunks) > 0:
                continue

            chunks = document.content.split(seperator)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Skip empty lines
                    document.chunks.append(
                        Chunk(
                            content=chunk.strip(),
                            chunk_id=i,
                            start_i=None,
                            end_i=None,
                            content_without_overlap=chunk.strip(),
                        )
                    )

        return documents