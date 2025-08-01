import torch
from collections import OrderedDict
from datasets import load_dataset
from data.datasets.base_dataset import BaseDataset
from PIL import Image
from io import BytesIO

def concat_images_horizontally_with_margin_PIL(images, margin=10):
    """
    Concatenates a list of PIL images horizontally with a margin, returns a new PIL image (not saved to file).
    """
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))

    x_offset = 0
    for image in images:
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin
    return new_image

class __DisplMixin:
    def displ_item(self, index):
        sample = self.__getitem__(index)
        return OrderedDict(
            {
                "question_id": sample["question_id"],
                "question": sample["text_input_raw"],
                "answers": "; ".join(sample["answers"]),
                "choices": "; ".join(sample["multiple_choices"]),
            }
        )

class BLINKDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="val"):
        vis_root = vis_root.split('/')[-1]
        self.dataset = load_dataset("BLINK-Benchmark/BLINK", vis_root, split=split)
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ann = self.dataset[index]

        image_list = []
        for i in range(1, 5):
            img = ann.get(f"image_{i}", None)
            if img is not None:
                image_list.append(img)
        
        if image_list:
            concat_image = concat_images_horizontally_with_margin_PIL(image_list)
            image_tensor = concat_image
            #image_tensor = self.vis_processor(concat_image)
        else:
            image_tensor = None

        # Combine prompt and question
        question = f"{ann.get('prompt', '').strip()}\n{ann['question'].strip()}".strip()
        question_proc = self.text_processor(question)

        # Answer
        answer = ann["answer"]
        answers = [answer] if answer != "hidden" else []
        weights = [1.0] if answer != "hidden" else []

        return {
            "image": image_tensor,
            "text_input": question_proc,
            "answers": answers,
            "weights": weights,
            "question_id": ann["idx"],
            "multiple_choices": ann["choices"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        return {
            "image": [s["image"] for s in samples],
            "text_input": [s["text_input"] for s in samples],
            "answer": [a for s in samples for a in s["answers"]],
            "weight": [w for s in samples for w in s["weights"]],
            "n_answers": torch.LongTensor([len(s["answers"]) for s in samples]),
            "question_id": [s["question_id"] for s in samples],
            "multiple_choices": [s["multiple_choices"] for s in samples],
        }


class BLINKDataset_Raw(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="val"):
        # Note: vis_root means subtask
        vis_root = vis_root.split('/')[-1]
        self.dataset = load_dataset("BLINK-Benchmark/BLINK", vis_root, split=split)
        self.text_processor = text_processor
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ann = self.dataset[index]

        image_list = []
        for i in range(1, 5):
            img = ann.get(f"image_{i}", None)
            if img is not None:
                image_list.append(img)
        
        if image_list:
            concat_image = concat_images_horizontally_with_margin_PIL(image_list)
            image_tensor = concat_image
            #image_tensor = self.vis_processor(concat_image)
        else:
            image_tensor = None
        question = f"{ann.get('prompt', '').strip()}\n{ann['question'].strip()}".strip()
        answer = ann["answer"]
        answers = [answer] if answer != "hidden" else []
        weights = [1.0] if answer != "hidden" else []

        return {
            "image_raw": image_tensor,
            "text_input_raw": question,
            "question_id": ann["idx"],
            "answers": answers,
            "weights": weights,
            "multiple_choices": ann["choices"],
        }

    def collater(self, samples):
        samples = [s for s in samples if s is not None]
        if not samples:
            return None

        return {
            "image_raw": [s["image_raw"] for s in samples],
            "text_input_raw": [s["text_input_raw"] for s in samples],
            "answer": [a for s in samples for a in s["answers"]],
            "weight": [w for s in samples for w in s["weights"]],
            "n_answers": torch.LongTensor([len(s["answers"]) for s in samples]),
            "question_id": [s["question_id"] for s in samples],
            "multiple_choices": [s["multiple_choices"] for s in samples],
        }


class BLINKInstructDataset(BLINKDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data is not None and data["answers"]:
            data["text_output"] = data["answers"][0]
        else:
            data["text_output"] = ""
        return data

    def collater(self, samples):
        data = super().collater(samples)
        data["text_output"] = data["answer"]
        return data


class BLINKEvalDataset(BLINKDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="test"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, split=split)
        self._add_instance_ids()

    def _add_instance_ids(self):
        for i, example in enumerate(self.dataset):
            self.dataset[i]["instance_id"] = i

    def __getitem__(self, index):
        data = super().__getitem__(index)
        data["instance_id"] = self.dataset[index]["instance_id"]
        return data


class BLINKEvalDataset_Raw(BLINKDataset_Raw):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split="test"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, split=split)
        self._add_instance_ids()

    def _add_instance_ids(self):
        self.dataset = self.dataset.map(lambda example, idx: {"instance_id": idx}, with_indices=True)


    def __getitem__(self, index):
        data = super().__getitem__(index)
        data["instance_id"] = self.dataset[index]["instance_id"]
        return data