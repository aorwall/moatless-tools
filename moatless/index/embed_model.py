import os


def get_embed_model(model_name: str) -> "BaseEmbedding":
    if model_name.startswith("voyage"):
        try:
            from llama_index.embeddings.voyageai import VoyageEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-voyageai is not installed. Please install it using `pip install llama-index-embeddings-voyageai`"
            ) from e

        if "VOYAGE_API_KEY" not in os.environ:
            raise ValueError("VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key.")

        return VoyageEmbedding(
            model_name=model_name,
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
            truncation=True,
            embed_batch_size=60,
        )
    else:
        # Assumes OpenAI otherwise
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError as e:
            raise ImportError(
                "llama-index-embeddings-openai is not installed. Please install it using `pip install llama-index-embeddings-openai`"
            ) from e

        return OpenAIEmbedding(model_name=model_name)
