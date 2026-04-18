//! Transform layer for converting between API formats.

pub mod anthropic_filter;
pub mod gemini_format;
pub mod image_cleaning;
pub mod openai_format;

pub use anthropic_filter::filter_non_anthropic_blocks;
pub use gemini_format::{
    build_tool_id_to_name_map, convert_anthropic_content_to_gemini,
    convert_anthropic_message_to_gemini, GeminiContent, GeminiConversionOptions, GeminiPart,
};
pub use image_cleaning::maybe_remove_image_blocks;
pub use openai_format::{
    consolidate_reasoning_details, convert_to_openai_messages, sanitize_gemini_messages,
    ConvertToOpenAiMessagesOptions, ReasoningDetail,
};
