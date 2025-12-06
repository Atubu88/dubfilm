import logging
import time

from openai import AsyncOpenAI

from config import OPENAI_API_KEY, OPENAI_SEARCH_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Найди актуальную информацию в интернете и выдай точный, проверяемый ответ с фактами, "
    "адресами, телефонами и ссылками, если они есть. Ничего не выдумывай."
)
_MAX_RESPONSE_LENGTH = 3500

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def run_search(query: str) -> str:
    """Execute a search query using OpenAI chat completions."""
    start_time = time.monotonic()
    logger.info("Running search for query: %s", query)

    response = await _client.chat.completions.create(
        model=OPENAI_SEARCH_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    result = (response.choices[0].message.content or "").strip()

    duration = time.monotonic() - start_time
    logger.info("Search query finished in %.2f seconds", duration)

    if len(result) > _MAX_RESPONSE_LENGTH:
        result = result[:_MAX_RESPONSE_LENGTH]

    return result
