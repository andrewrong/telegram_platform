"""Ingredient analyzer using LLM."""

import logging

from telegramify_markdown import convert

from src.exceptions import LLMServiceError
from src.services.llm import LLMService
from src.services.ocr import OCREngine, OCRService

logger = logging.getLogger(__name__)


class IngredientAnalyzer:
    """Analyzer for food ingredient lists.

    This service analyzes ingredient lists from text or images using
    LLM to provide health assessments and detailed explanations.

    Attributes:
        llm_service: LLM service for analysis.
    """

    def __init__(self, llm_service: LLMService, ocr_engine: OCREngine = "glm") -> None:
        """Initialize ingredient analyzer.

        Args:
            llm_service: LLM service instance.
            ocr_engine: OCR engine used for ingredient images.
        """
        self.llm_service = llm_service
        self.ocr_engine = ocr_engine
        logger.info("IngredientAnalyzer initialized")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for ingredient analysis.

        Returns:
            System prompt string.
        """
        return """你是一位专业的食品成分分析师。
你的任务是分析食品配料表，并提供详细的健康评估和产品推荐。

请按以下格式输出分析报告：

## 配料表分析

### 1. 整体评估
- 健康等级：⭐⭐⭐⭐⭐ (1-5星)
- 简短总结：一句话概括该食品的健康程度

### 2. 成分分类
- **主要成分**：列出主要原料
- **添加剂**：列出食品添加剂（如有）
- **过敏原**：标注可能的过敏原（如有）

### 3. 重点关注成分
对以下类型的成分进行特别说明：
- 高糖成分
- 高盐/钠成分
- 反式脂肪酸
- 人工色素
- 防腐剂
- 其他值得注意的成分

### 4. 健康建议
- 适合人群
- 注意事项
- 食用建议

### 5. 品牌与产品推荐
基于该产品的成分分析，推荐更好的选择：

**🏆 无污点品牌（安全可靠）**
- **优先推荐欧盟、美国、日本、澳洲等严格监管地区的品牌**
- 历史上无重大食品安全丑闻，长期质量稳定
- 奶源/原料来自监管严格地区，抗生素、激素标准严苛
- ❌ **明确排除：**
  - 曾因三聚氰胺、黄曲霉毒素、抗生素超标等严重问题被处罚的品牌
  - 缺乏第三方国际认证（如欧盟有机认证、USDA Organic）的国内品牌

**⭐ 口碑品牌（行业认可）**
- 在严格监管市场中长期保持好评的国际品牌
- 有第三方国际检测报告（欧盟EFSA、美国FDA等）
- 质量控制体系完善（自有牧场、全程可追溯、第三方审计）

**🌿 健康替代（更优选择）**
- 成分更简单、无添加的海外品牌产品
- 有机认证（EU Organic、USDA Organic、JAS）产品
- 草饲、A2蛋白等高品质奶源产品

**重要提醒：**
- **优先选择海外品牌：**欧盟、日本、澳洲等地的食品安全监管更为严格
- **国内品牌需格外谨慎：**仅推荐有国际认证、长期无质量丑闻的极少数品牌
- **不推荐：**缺乏透明检测报告、依赖营销而非质量的国内大品牌
- 品牌知名度不等于质量，大型商业品牌也可能存在严重质量问题

请使用中文回答，保持客观、专业、易懂。推荐的品牌应基于实际质量记录和监管严格程度。"""

    def _build_user_prompt(self, ingredient_text: str) -> str:
        """Build user prompt from ingredient text.

        Args:
            ingredient_text: Raw ingredient text.

        Returns:
            Formatted user prompt.
        """
        return f"请分析以下食品配料表：\n\n{ingredient_text}\n\n请提供详细的分析报告。"

    @staticmethod
    def is_plausible_ingredient_text(text: str) -> bool:
        """Return whether OCR text is plausible input for ingredient analysis."""
        meaningful = [character for character in text if character.isalnum()]
        if not meaningful or not any(character.isalpha() for character in meaningful):
            return False

        if len(meaningful) >= 8:
            most_common_count = max(meaningful.count(character) for character in set(meaningful))
            if most_common_count / len(meaningful) >= 0.8:
                return False
        return True

    async def analyze_from_text(self, ingredient_text: str) -> str:
        """Analyze ingredient list from text.

        Args:
            ingredient_text: Ingredient list as text.

        Returns:
            Markdown formatted analysis report.
        """
        text = ingredient_text.strip()
        if not text:
            return "❌ 无法分析：配料表为空"
        if not self.is_plausible_ingredient_text(text):
            logger.warning("Rejected implausible ingredient OCR text: %d chars", len(text))
            return (
                "❌ 无法分析：未识别到有效配料文字。"
                "请靠近包装拍摄，让配料表占据画面主要区域，或裁剪后重试。"
            )

        logger.info(f"Analyzing ingredient text: {len(text)} chars")

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(text)},
        ]

        try:
            result = await self.llm_service.chat(messages, temperature=0.3)
            logger.info("Ingredient analysis completed successfully")
            return result
        except LLMServiceError as e:
            logger.error(f"LLM analysis failed: {e}")
            return f"❌ 分析失败: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error in analysis: {e}")
            return f"❌ 分析过程中出现错误: {e}"

    def format_for_telegram(self, markdown_text: str) -> tuple[str, list]:
        """Convert markdown text to Telegram format.

        Uses telegramify-markdown to convert standard Markdown to Telegram
        plain text with MessageEntity objects.

        Args:
            markdown_text: Raw markdown text from LLM.

        Returns:
            Tuple of (plain_text, entities) ready for Telegram API.
        """
        # Handle error messages (no conversion needed)
        if markdown_text.startswith("❌"):
            return markdown_text, []

        try:
            text, entities = convert(markdown_text)
            # Convert entities to dict format for python-telegram-bot
            entities_dict = [e.to_dict() for e in entities]
            return text, entities_dict
        except Exception as e:
            logger.error(f"Failed to format markdown for Telegram: {e}")
            # Fallback to plain text
            return markdown_text, []

    async def analyze_from_image(self, image_data: bytes, ocr_service: OCRService) -> str:
        """Analyze ingredient list from image.

        This method first extracts text from the image using OCR,
        then analyzes the extracted text.

        Args:
            image_data: Raw image bytes.
            ocr_service: OCR service for text extraction.

        Returns:
            Markdown formatted analysis report.
        """
        logger.info(f"Extracting text from image: {len(image_data)} bytes")

        try:
            ocr_result = await ocr_service.process_image(
                image_data,
                engine=self.ocr_engine,
            )

            if not ocr_result.success:
                error_msg = ocr_result.error or "OCR failed"
                logger.error(f"OCR failed: {error_msg}")
                return f"❌ 图片识别失败: {error_msg}"

            extracted_text = ocr_result.text.strip()
            if not extracted_text:
                logger.warning("OCR returned empty text")
                return "❌ 无法从图片中识别出文字，请确保图片清晰且包含配料表"

            logger.info(f"OCR extracted {len(extracted_text)} chars, analyzing...")
            return await self.analyze_from_text(extracted_text)

        except Exception as e:
            logger.exception(f"Error processing image: {e}")
            return f"❌ 图片处理失败: {e}"
