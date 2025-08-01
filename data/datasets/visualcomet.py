import os
import json
import random
from PIL import Image
import torch
from data.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from data.datasets.base_dataset import BaseDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict({
            "file": ann["img_fn"],
            "question": ann["question"],
            "question_id": ann["question_id"],
            "answers": "; ".join(ann["answer"]),
            "image": sample["image"],
        })

class VisualCOMETDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            answer_weight[answer] = answer_weight.get(answer, 0) + 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class VisualCOMETDataset_Raw(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.jsonl_paths = ann_paths
        self.full_jsonl = []

        if isinstance(self.jsonl_paths, list):
            for path in self.jsonl_paths:
                with open(path, "r") as f:
                    self.full_jsonl.extend(json.load(f))
        else:
            with open(self.jsonl_paths, "r") as f:
                self.full_jsonl = json.load(f)

        question_templates = {
            "event": "what is happening here?",
            "before": "what happened before?",
            "after": "what will happen after?",
            "intent": "what is the intent?"
        }

        bbox_guide = (
            "We have drawn bounding boxes around each person in the image. "
            "Please refer to each person by their number: red is person 1, orange is 2, yellow is 3, green is 4, and blue is 5. "
            "When answering the question, use these numbers (e.g., '1' instead of 'the red person')."
        )

        self.flattened_qa = []
        self.valid_image_paths = []

        for item in self.full_jsonl:
            img_fn = item["img_fn"]
            full_img_path = os.path.join(vis_root.replace('visualcomet', 'visualcomet-bbox'), img_fn)
            if not os.path.exists(full_img_path):
                continue

            for key in ["event", "before", "after", "intent"]:
                if key in item and item[key]:
                    question_base = question_templates[key]
                    question = f"{bbox_guide} {question_base}"
                    answers = item[key] if isinstance(item[key], list) else [item[key]]

                    self.flattened_qa.append({
                        "img_fn": img_fn,
                        "image_path": full_img_path,
                        "question_type": key,
                        "question": question,
                        "answers": answers
                    })

    def __len__(self):
        return len(self.flattened_qa)

    def __getitem__(self, index):
        qa_entry = self.flattened_qa[index]
        image_path = qa_entry["image_path"]

        try:
            image_raw = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return self.__getitem__((index + 1) % len(self))

        question = self.text_processor(qa_entry["question"])

        answer_weight = {}
        for answer in qa_entry["answers"]:
            answer_weight[answer] = answer_weight.get(answer, 0) + 1 / len(qa_entry["answers"])

        return {
            "image_raw": image_raw,
            "text_input_raw": qa_entry["question"],
            "question_id": question,
            "answers": list(answer_weight.keys()),
            "weights": list(answer_weight.values()),
        }

