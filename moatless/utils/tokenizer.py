import os

_voyage_clients = {}
_tiktoken_encoders = {}
_hf_tokenizers = {}


def count_tokens(content: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in `content` based on the model name.

    1) If `model.startswith("voyage")`, use VoyageAI client.
    2) Otherwise, try to use `tiktoken`. If tiktoken raises KeyError
       (unrecognized model name), fallback to Hugging Face tokenizer.
    """
    if model.startswith("voyage"):
        if model not in _voyage_clients:
            # Lazy-import VoyageAI & create a client
            try:
                import voyageai
            except ImportError as e:
                raise ImportError(
                    "`voyageai` package not found, please run `pip install voyageai`"
                ) from e

            _voyage_clients[model] = voyageai.Client()

        # VoyageAI expects a list of texts
        return _voyage_clients[model].count_tokens([content])

    try:
        import tiktoken

        # If we don't already have a tiktoken encoder for this model,
        # try to fetch it. This will raise KeyError if the model name
        # is unknown (e.g. a Hugging Face model).
        if model not in _tiktoken_encoders:
            # Temporarily set TIKTOKEN_CACHE_DIR if not present
            should_revert = False
            if "TIKTOKEN_CACHE_DIR" not in os.environ:
                should_revert = True
                os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "_static",
                    "tiktoken_cache"
                )

            _tiktoken_encoders[model] = tiktoken.encoding_for_model(model)

            # Clean up TIKTOKEN_CACHE_DIR if we set it
            if should_revert:
                del os.environ["TIKTOKEN_CACHE_DIR"]

        # Now we can encode
        encoder = _tiktoken_encoders[model]
        return len(encoder.encode(content, allowed_special="all"))

    except ImportError as e:
        # tiktoken isn't installed at all
        raise ImportError(
            "`tiktoken` package not found, please run `pip install tiktoken`"
        ) from e

    except KeyError:
        """
        tiktoken raised KeyError, meaning it does not recognize this `model`
        as a valid OpenAI model. We will fallback to a Hugging Face tokenizer.
        """

        if model not in _hf_tokenizers:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "Hugging Face `transformers` not found. "
                    "Please install via `pip install transformers`."
                ) from e

            _hf_tokenizers[model] = AutoTokenizer.from_pretrained(model)

        hf_tokenizer = _hf_tokenizers[model]
        # For HF, simply return the length of the encoded IDs
        return len(hf_tokenizer.encode(content))
