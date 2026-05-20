"""Data preprocessing helpers for the OpenI dataset.

The package stays lightweight on import. Heavy modules such as the tokenizer,
dataset, transforms, and dataloader are loaded lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module

from src.data.parser import (
	ParsedRecord,
	build_image_index,
	clean_text,
	extract_image_references,
	extract_report_text,
	iter_xml_files,
	parse_xml_file,
	resolve_image_reference,
)

_LAZY_EXPORTS = {
	# Dataset / tokenizer helpers
	"DEFAULT_MAX_LENGTH": ("src.data.tokenizer", "DEFAULT_MAX_LENGTH"),
	"SPECIAL_TOKENS": ("src.data.tokenizer", "SPECIAL_TOKENS"),
	"TokenizerStats": ("src.data.tokenizer", "TokenizerStats"),
	"build_tokenizer": ("src.data.tokenizer", "build_tokenizer"),
	"encode_report": ("src.data.tokenizer", "encode_report"),
	"iter_reports": ("src.data.tokenizer", "iter_reports"),
	"load_or_build_tokenizer": ("src.data.tokenizer", "load_or_build_tokenizer"),
	"load_tokenizer": ("src.data.tokenizer", "load_tokenizer"),
	"prepare_tokenizer": ("src.data.tokenizer", "prepare_tokenizer"),
	"validate_special_tokens": ("src.data.tokenizer", "validate_special_tokens"),
	# Multimodal dataset helpers
	"OpenIRadiologyDataset": ("src.data.dataset", "OpenIRadiologyDataset"),
	"load_openi_dataset": ("src.data.dataset", "load_openi_dataset"),
	"preview_dataset_image": ("src.data.dataset", "preview_dataset_image"),
	# Image transform helpers
	"CLIP_IMAGE_SIZE": ("src.data.transforms", "CLIP_IMAGE_SIZE"),
	"CLIP_MEAN": ("src.data.transforms", "CLIP_MEAN"),
	"CLIP_STD": ("src.data.transforms", "CLIP_STD"),
	"TransformPreview": ("src.data.transforms", "TransformPreview"),
	"build_clip_image_transform": ("src.data.transforms", "build_clip_image_transform"),
	"build_inference_transforms": ("src.data.transforms", "build_inference_transforms"),
	"build_training_transforms": ("src.data.transforms", "build_training_transforms"),
	"build_validation_transforms": ("src.data.transforms", "build_validation_transforms"),
	"preview_transform": ("src.data.transforms", "preview_transform"),
	"safe_load_image_tensor": ("src.data.transforms", "safe_load_image_tensor"),
	# DataLoader helpers
	"DataLoaderConfig": ("src.data.dataloader", "DataLoaderConfig"),
	"build_dataloader": ("src.data.dataloader", "build_dataloader"),
	"build_test_dataloader": ("src.data.dataloader", "build_test_dataloader"),
	"build_train_dataloader": ("src.data.dataloader", "build_train_dataloader"),
	"build_validation_dataloader": ("src.data.dataloader", "build_validation_dataloader"),
	"create_openi_dataloaders": ("src.data.dataloader", "create_openi_dataloaders"),
	"log_dataloader_config": ("src.data.dataloader", "log_dataloader_config"),
	"preview_batch_shapes": ("src.data.dataloader", "preview_batch_shapes"),
	"preview_dataloader_batch": ("src.data.dataloader", "preview_dataloader_batch"),
}

__all__ = sorted({
	"ParsedRecord",
	"build_image_index",
	"clean_text",
	extract_image_references.__name__,
	extract_report_text.__name__,
	"iter_xml_files",
	"parse_xml_file",
	"resolve_image_reference",
    * _LAZY_EXPORTS.keys(),
})


def __getattr__(name: str):
	if name not in _LAZY_EXPORTS:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module_name, attribute_name = _LAZY_EXPORTS[name]
	module = import_module(module_name)
	value = getattr(module, attribute_name)
	globals()[name] = value
	return value
