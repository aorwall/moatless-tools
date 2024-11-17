from hashlib import sha256

from llama_index.core.schema import TextNode


class CodeNode(TextNode):
    # Skip start and end line in metadata to try to lower the number of changes and triggers of new embeddings.
    @property
    def hash(self):
        metadata = self.metadata.copy()
        metadata.pop("start_line", None)
        metadata.pop("end_line", None)
        metadata.pop("tokens", None)
        cleaned_text = self._clean_text(self.text)
        doc_identity = cleaned_text + str(metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    def _clean_text(self, text):
        """
        Remove all whitespace and convert to lowercase to reduce the number of changes in hashes.
        """
        return "".join(text.split()).lower()
