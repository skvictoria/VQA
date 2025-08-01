import os
import json
import random
import torch
import pandas as pd

from PIL import Image

from data.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from data.datasets.base_dataset import BaseDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class GQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }
    

class GQA_Raw(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            if any(ext in ann_path for ext in ['csv', 'tsv']):
                df = pd.read_csv(ann_path)
                self.annotation.extend(df.to_dict(orient="records"))
                
            elif 'jsonl' in ann_path:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f])

            else:
                with open(ann_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                       self.annotation.extend([{"question_id": k, **v} if isinstance(v, dict) else {"question_id": k, "data": v} for k, v in loaded.items()])
                       self.annotation = [{**ann, "image": ann["imageId"] + ".jpg"} for ann in self.annotation]


        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        answer_list, weight_list = [], []
        image_raw_list, question_raw_list, multiple_choice_answer_list = [], [], []

        num_answers = []

        for sample in samples:
            image_raw_list.append(sample["image_raw"])
            question_raw_list.append(sample["text_input_raw"])

            multiple_choice_answer_list.append(sample["multiple_choice_answer"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image_raw": image_raw_list,
            "text_input_raw": question_raw_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
            "multiple_choice_answer": multiple_choice_answer_list,
        }
    

class GQADataset_Raw(GQA_Raw, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print(ann)

        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")

        answers = [ann["answer"]]
        weights = [1]

        # select the most frequent multiple_choice_answer in the list - ann["answer"]
        multiple_choice_answer = ann["answer"]

        return {
            "answers": answers,
            "multiple_choice_answer": multiple_choice_answer,
            "weights": weights,
            "image_raw": image_raw,
            "text_input_raw": ann["question"],
        }


class GQAInstructDataset(GQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = random.choice(data["answers"])
        return data

    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data


class GQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        if "answer" in ann:
            # answer is a string
            answer = ann["answer"]
        else:
            answer = None

        return {
            "image": image,
            "text_input": question,
            "answer": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
    

class GQAEvalDataset_Raw(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        loaded = json.load(open(ann_paths[0]))
        self.annotation = []
        if isinstance(loaded, list):
            self.annotation.extend(loaded)
        elif isinstance(loaded, dict):
            self.annotation.extend([{"question_id": k, **v} if isinstance(v, dict) else {"question_id": k, "data": v} for k, v in loaded.items()])
            # change the key 'imageId' to 'iamge' and add ".jpg" after the value of 'image'
            self.annotation = [{**ann, "image": ann["imageId"] + ".jpg"} for ann in self.annotation]

        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")
        multiple_choice_answer = ann["answer"]

        answers = [ann["answer"]]
        weights = [1]

        return {
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "image_raw": image_raw,
            "image_path": image_path,
            "text_input_raw": ann["question"],
            "multiple_choice_answer": multiple_choice_answer,
            "answers": answers,
            "weights": weights,
        }

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        answer_list, weight_list = [], []
        image_raw_list, question_raw_list, multiple_choice_answer_list = [], [], []
        num_answers = []

        question_id_list, instance_id_list = [], []

        for sample in samples:
            image_raw_list.append(sample["image_raw"])
            question_raw_list.append(sample["text_input_raw"])

            multiple_choice_answer_list.append(sample["multiple_choice_answer"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

            question_id_list.append(sample["question_id"])
            instance_id_list.append(sample["instance_id"])

        return {
            "image_raw": image_raw_list,
            "text_input_raw": question_raw_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
            "multiple_choice_answer": multiple_choice_answer_list,
            "question_id": question_id_list,
            "instance_id": instance_id_list,
        }