import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from common.registry import registry
from models.base_model import BaseModel
from common.utils import get_abs_path, is_url, download_cached_file
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import contextlib
import copy
from tasks.vqa_task_utils import QAOutput

@registry.register_model("qwenvl")
class QwenVL(BaseModel):
    """
    QwenVL model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "Qwen/Qwen2-VL-2B-Instruct": "configs/models/qwen/qwen2_vl_2b_instruct.yaml",
    }

    def __init__(
        self,
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        dtype=torch.bfloat16,
        apply_lemmatizer=False,
        return_action=False,
    ):
        super().__init__()

        self.model_id = model_id
        print("model_id", model_id)
        self.dtype = dtype
        self.return_action = return_action

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="eager", ###########
            low_cpu_mem_usage=True,
        )
        self.config = self.model.config

        # print(self.config)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples, **kwargs):
        image = samples["image_raw"]

        if isinstance(samples["text_input_raw"], str):
            samples["text_input_raw"] = [samples["text_input_raw"]]
        # if prompt:
        #     text_input = [prompt.format(question) for question in samples["text_input_raw"]]
        # else:
        #     text_input = samples["text_input_raw"]

        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": s},
                ],
            } for s, img in zip(text_input, image)
        ]
        text_input = [self.processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True) for conv in conversation]

        model_inputs = self.processor(text=text_input, images=image, return_tensors="pt", padding="longest").to(self.dtype).to(self.device)
        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        # print("loss: ", loss)
    
        return {"loss": loss}
    
    def predict_answers(
            self, 
            samples,
            num_beams=5,
            inference_method="generate",
            max_len=10,
            min_len=1,
            num_ans_candidates=128,
            answer_list=None,
            prompt="",
            length_penalty=-1,
            unnorm_key="bridge_orig",
            **kwargs
        ):
        image = samples["image_raw"]

        if isinstance(samples["text_input_raw"], str):
            samples["text_input_raw"] = [samples["text_input_raw"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input_raw"]]
        else:
            text_input = samples["text_input_raw"]

        conversation = [
            {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": s},
                ],
            } for s, img in zip(text_input, image)
        ]
        text_input = [self.processor.apply_chat_template([conv], tokenize=False, add_generation_prompt=True) for conv in conversation]

        model_inputs = self.processor(text=text_input, images=image, return_tensors="pt", padding="longest").to(self.dtype).to(self.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            # When the model generates a response, it appends the generated tokens to this input sequence.
            outputs = outputs[:, input_len:]
            output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        # print("output_text", output_text)

        if ("return_dict" in kwargs) and kwargs["return_dict"]:
            return QAOutput(answer=output_text)
        else:
            return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        model_id = cfg.get("model_id", "Qwen/Qwen2-VL-2B-Instruct")
        dtype = cfg.get("dtype", torch.bfloat16)

        model = cls(
            model_id=model_id,
            dtype=dtype,
        )

        load_finetuned = cfg.get("load_finetuned", False)

        # LoRA
        use_lora = int(cfg.get("use_lora", 0))
        lora_alpha = int(cfg.get("lora_alpha", 16))
        lora_rank = int(cfg.get("lora_rank", 4))
        target_modules = cfg.get("target_modules", "q_proj k_proj v_proj o_proj").split()

        if use_lora == 1:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules,
            )
            
            logging.info(lora_config)
            
            model = get_peft_model(model, lora_config)
            logging.info(model.print_trainable_parameters())

        # Linear Probe
        linear_probe = int(cfg.get("linear_probe", 0))
        if linear_probe == 1:
            assert use_lora == 0, "Linear probe and LoRA cannot be used together"
            # only tune "lm_head" layer
            for name, module in model.named_modules():
                if "lm_head" not in name:
                    for param in module.parameters():
                        param.requires_grad_(False)
                        print(f"{name} requires_grad: {param.requires_grad}")
                else:
                    for param in module.parameters():
                        param.requires_grad_(True)
                        print(f"{name} requires_grad: {param.requires_grad}")
            logging.info("Linear probe: only tune 'lm_head' layer")
        
        # WiSE
        wise = int(cfg.get("wise", 0))
        if wise == 1:
            assert load_finetuned, "WiSE requires load_finetuned=True"
            w0 = {key: value.to('cpu') for key, value in model.state_dict().items()}
            w0 = copy.deepcopy(w0)
        
        if load_finetuned:
            model.load_checkpoint_from_config(cfg)

        finetune_lp_path = cfg.get("finetuned_lp", None)
        if finetune_lp_path is not None:
            model.load_checkpoint(finetune_lp_path)
            logging.info("load linear probe checkpoint from %s" % finetune_lp_path)
        
        if wise == 1:
            w1 = {key: value.to('cpu') for key, value in model.state_dict().items()}
            # alpha * w0 + (1 - alpha) * w1
            alpha = 0.5
            wise = {key: (alpha * w0[key] + (1 - alpha) * w1[key]).to(model.device) for key in w1.keys()}
            model.load_state_dict(wise)
            logging.info("WiSE: load finetuned model and apply WiSE")

        # print("Final Model before runner", model)

        return model
    
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device, weights_only=True)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device, weights_only=True)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            start_key = "base_model.model.model."
            if key.startswith(start_key):
                state_dict[key[len(start_key):]] = state_dict.pop(key)
        
        # Load the current state_dict of the model
        current_state_dict = self.model.state_dict()

        # Update the current state_dict with the new parameters
        for key in state_dict.keys():
            assert key in current_state_dict, f"key {key} not in current_state_dict"
            current_state_dict[key] = state_dict[key]

        # Load the updated state_dict back into the model
        self.model.load_state_dict(current_state_dict)
        logging.info("load pretrained checkpoint from %s" % url_or_filename)

    def load_checkpoint(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device, weights_only=True)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device, weights_only=True)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        for key in list(state_dict.keys()):
            start_key = "base_model.model.model."
            start_key_2 = "model."
            if key.startswith(start_key):
                state_dict[key[len(start_key):]] = state_dict.pop(key)
            elif key.startswith(start_key_2):
                state_dict[key[len(start_key_2):]] = state_dict.pop(key)
        
        # Load the current state_dict of the model
        current_state_dict = self.model.state_dict()

        # Update the current state_dict with the new parameters
        for key in state_dict.keys():
            assert key in current_state_dict, f"key {key} not in current_state_dict"
            current_state_dict[key] = state_dict[key]

        # Load the updated state_dict back into the model
        self.model.load_state_dict(current_state_dict)
        logging.info("load checkpoint from %s" % url_or_filename)


if __name__ == "__main__":
    # Example usage:
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    device = "cuda:0"
    dtype = torch.bfloat16

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # print("image", np.array(image))

    samples = {
        "image_raw": [image],
        "text_input_raw": ["What is the color of the car"],
        "multiple_choice_answer": ["red"],
    }
    with torch.inference_mode():
        model = QwenVL(model_id=model_id, dtype=dtype).to(device)
        # loss = model(samples)
        # print("loss", loss)
        text = model.predict_answers(samples, prompt='Question: {} Short answer:')
        print(text)

