from ai.service import AIService


async def run_summary(original_text: str, translated_text: str, target_language: str, ai_service: AIService) -> str:
    return await ai_service.summarize_text(
        original_text=original_text,
        translated_text=translated_text,
        target_language=target_language,
    )
