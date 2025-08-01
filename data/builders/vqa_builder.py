from data.builders.base_dataset_builder import BaseDatasetBuilder

from common.registry import registry
from data.datasets.coco_vqa import *
from data.datasets.gqa_datasets import *
from data.datasets.temporal_vqa import *
from data.datasets.visualcomet import *
from data.datasets.blink import *
from pathlib import Path
import warnings
import os

@registry.register_builder("blink_visual_similarity")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_Visual_Similarity.yaml"
    }

@registry.register_builder("blink_visual_correspondance")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_visual_correspondance.yaml"
    }

@registry.register_builder("blink_spatial_relation")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_Spatial_Relation.yaml"
    }

@registry.register_builder("blink_semantic_correspondance")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_semantic_correspondance.yaml"
    }

@registry.register_builder("blink_relative_depth")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_Relative_Depth.yaml"
    }

@registry.register_builder("blink_object_localization")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_object_localization.yaml"
    }

@registry.register_builder("blink_multiview_reasoning")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_multiview_reasoning.yaml"
    }

@registry.register_builder("blink_jigsaw")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_Jigsaw.yaml"
    }

@registry.register_builder("blink_counting")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_counting.yaml"
    }

@registry.register_builder("blink_functional_correspondance")
class BLINKBuilder(BaseDatasetBuilder):
    train_dataset_cls = BLINKDataset_Raw
    eval_dataset_cls = BLINKEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/blink_functional_correspondance.yaml"
    }

@registry.register_builder("visualcomet")
class VisualCOMETBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisualCOMETDataset_Raw
    eval_dataset_cls = VisualCOMETDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/visualcomet.yaml"
    }

@registry.register_builder("coco_vqa_raw")
class COCOVQABuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/defaults_vqa_raw.yaml",
        "eval": "configs/datasets/vqa/eval_vqa_raw.yaml",
    }

@registry.register_builder("gqa_raw")
class GQABuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = GQADataset_Raw
    eval_dataset_cls = GQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults_gqa_raw.yaml",
        "eval": "configs/datasets/gqa/defaults_gqa_raw.yaml"
    }

@registry.register_builder("gqa_ood")
class GQAOODBuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = GQADataset_Raw
    eval_dataset_cls = GQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/gqa-ood.yaml",
        "eval": "configs/datasets/gqa/gqa-ood.yaml"
    }


@registry.register_builder("coco_vqa_cp")
class COCOVQACPBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/vqa_cp.yaml"
    }


@registry.register_builder("coco_vqa_rephrasings")
class COCOVQA_Rephrasings_Builder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/vqa_rephrasings.yaml"
    }

@registry.register_builder("coco_vqa_ce")
class COCOVQACEBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/vqa_ce.yaml"
    }

@registry.register_builder("coco_cv-vqa")
class COCOCVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/cv-vqa.yaml"
    }

@registry.register_builder("coco_iv-vqa")
class COCOIVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/iv-vqa.yaml"
    }

@registry.register_builder("coco_advqa")
class COCOADVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/advqa.yaml"
    }

@registry.register_builder("textvqa")
class TextVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/textvqa.yaml"
    }

@registry.register_builder("vizwiz")
class VizWizBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/vizwiz.yaml"
    }

@registry.register_builder("coco_okvqa")
class COCOOKVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "configs/datasets/vqa/ok-vqa.yaml"
    }

@registry.register_builder("temporal_vqa")
class TemporalVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = TemporalVQADataset_Raw
    eval_dataset_cls = TemporalVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa/temporal_vqa.yaml"
    }