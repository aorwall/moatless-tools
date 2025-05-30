from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class BlockType(str, Enum):
    text = "text"
    list = "list"
    table = "table"


class TextVariant(str, Enum):
    default = "default"
    heading = "heading"
    subheading = "subheading"


class ListVariant(str, Enum):
    unordered = "unordered"
    ordered = "ordered"


class BaseBlock(BaseModel):
    type: BlockType
    id: str


class TextBlock(BaseBlock):
    type: BlockType = BlockType.text
    content: str
    variant: Optional[TextVariant] = TextVariant.default


class ListBlock(BaseBlock):
    type: BlockType = BlockType.list
    items: list[str]
    variant: Optional[ListVariant] = ListVariant.unordered


class TableBlock(BaseBlock):
    type: BlockType = BlockType.table
    headers: list[str]
    rows: list[list[Union[str, int]]]


ContentBlock = Union[TextBlock, ListBlock, TableBlock]


class ContentSection(BaseModel):
    id: str
    title: str
    blocks: list[ContentBlock]
    sections: Optional[list["ContentSection"]] = None


class ContentStructure(BaseModel):
    title: str | None = None
    sections: list[ContentSection]
