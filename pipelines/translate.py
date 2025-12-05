from ai.service import AIService


async def run_translation(text: str, source_language: str, target_language: str, ai_service: AIService) -> str:
    return await ai_service.translate_text(
        text=text,
        source_language=source_language,
        target_language=target_language,
    )
