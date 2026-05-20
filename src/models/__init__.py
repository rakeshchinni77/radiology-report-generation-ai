"""Model components for OpenI multimodal radiology report generation."""

from src.models.encoder import (
	DEFAULT_CLIP_VISION_MODEL,
	EncoderConfig,
	EncoderPreview,
	OpenIClipVisionEncoder,
	build_clip_vision_encoder,
	count_trainable_parameters,
	preview_encoder_embeddings,
	validate_encoder_frozen,
)
from src.models.decoder import (
	DEFAULT_GPT2_MODEL,
	DEFAULT_TRAINABLE_TOP_BLOCKS,
	DecoderConfig,
	DecoderPreview,
	OpenIGPT2Decoder,
	build_gpt2_decoder,
	preview_decoder_output_shapes,
	validate_decoder_frozen,
)
from src.models.attention import (
	DEFAULT_DROPOUT,
	DEFAULT_MODEL_DIM,
	DEFAULT_NUM_HEADS,
	AttentionConfig,
	AttentionPreview,
	OpenICrossAttentionFusion,
	build_cross_attention_fusion,
	preview_cross_attention_shapes,
	validate_attention_trainability,
)

__all__ = [
	"DEFAULT_CLIP_VISION_MODEL",
	"DEFAULT_GPT2_MODEL",
	"DEFAULT_MODEL_DIM",
	"DEFAULT_NUM_HEADS",
	"DEFAULT_DROPOUT",
	"DEFAULT_TRAINABLE_TOP_BLOCKS",
	"EncoderConfig",
	"EncoderPreview",
	"DecoderConfig",
	"DecoderPreview",
	"AttentionConfig",
	"AttentionPreview",
	"OpenIClipVisionEncoder",
	"OpenIGPT2Decoder",
	"OpenICrossAttentionFusion",
	"build_clip_vision_encoder",
	"build_gpt2_decoder",
	"build_cross_attention_fusion",
	"count_trainable_parameters",
	"preview_encoder_embeddings",
	"preview_decoder_output_shapes",
	"preview_cross_attention_shapes",
	"validate_encoder_frozen",
	"validate_decoder_frozen",
	"validate_attention_trainability",
]

from src.models.multimodal_model import (
	MultimodalConfig,
	MultimodalReportGenerator,
	build_multimodal_model,
)

__all__ += [
	"MultimodalConfig",
	"MultimodalReportGenerator",
	"build_multimodal_model",
]
