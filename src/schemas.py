from pydantic import BaseModel


class Block(BaseModel):
    vertices: list[tuple[float, float]] = []
    test: str


class Page(BaseModel):
    blocks: list[Block]


class ParsedDocument(BaseModel):
    pages: list[Page]
