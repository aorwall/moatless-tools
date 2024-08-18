import logging
from typing import List

from llama_index.embeddings.voyageai import VoyageEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt
from voyageai.error import InvalidRequestError

logger = logging.getLogger(__name__)


class VoyageEmbeddingWithRetry(VoyageEmbedding):
    @retry(
        wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6)
    )
    def _get_embedding(self, texts: List[str], input_type: str) -> List[List[float]]:
        try:
            return self._client.embed(
                texts,
                model=self.model_name,
                input_type=input_type,
                truncation=self.truncation,
            ).embeddings
        except InvalidRequestError as e:
            if "Please lower the number of tokens in the batch" in str(e):
                if len(texts) < 10:
                    raise  # If batch size is already less than 10 we expect batchs to be abnormaly large and raise the error

                mid = len(texts) // 2
                first_half = texts[:mid]
                second_half = texts[mid:]

                logger.info(
                    f"Splitting batch of {len(texts)} texts into two halves of {len(first_half)} and {len(second_half)} texts."
                )

                embeddings_first = self._get_embedding(first_half, input_type)
                embeddings_second = self._get_embedding(second_half, input_type)

                return embeddings_first + embeddings_second
            raise
