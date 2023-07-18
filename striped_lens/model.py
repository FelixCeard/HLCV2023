import os
from pathlib import Path
from typing import Any, List, Optional

import huggingface_hub
import open_clip
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    CLIPModel,
    CLIPProcessor,
)


def flatten(l):
    return [item for sublist in l for item in sublist]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lens(nn.Module):
    def __init__(
        self,
        clip_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        attributes_weights: str = "zw_attributes_laion_ViT_H_14_2B_descriptors_text_davinci_003_full.pt",
        tags_weights: str = "zw_tags_laion_ViT_H_14_2B_vocab_lens.pt",
        vocab_attributes: str = "llm-lens/descriptors-text-davinci-003",
        vocab_tags: str = "llm-lens/vocab_tags",
        split_attributes: str = "full",
        split_tags: str = "train",
        device: torch.device = device,
    ):
        super().__init__()
        # Load Base models
        self.device = device
        self.clip_name = clip_name
        if self.clip_name is not None:
            self.clip_model = self.load_clip_model(self.clip_name, self.device)
            # Load weights
            huggingface_hub.hf_hub_download(
                repo_id="llm-lens/attributes",
                filename=attributes_weights,
                local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            )
            huggingface_hub.hf_hub_download(
                repo_id="llm-lens/tags",
                filename=tags_weights,
                local_dir=str(Path(Path(__file__).resolve().parent) / "weights"),
            )

            self.attributes_weights = torch.load(
                str(
                    Path(Path(__file__).resolve().parent)
                    / f"weights/{attributes_weights}"
                ),
                map_location=self.device,
            ).float()
            self.tags_weights = torch.load(
                str(Path(Path(__file__).resolve().parent) / f"weights/{tags_weights}"),
                map_location=self.device,
            ).float()
            # Load Vocabularies
            self.vocab_tags = load_dataset(vocab_tags, split=split_tags)[
                "prompt_descriptions"
            ]
            self.vocab_attributes = flatten(
                load_dataset(vocab_attributes, split=split_attributes)[
                    "prompt_descriptions"
                ]
            )

    def load_clip_model(self, model_name: str, device: torch.device):
        if "openai" in model_name:
            model = CLIPModel.from_pretrained(model_name).to(device)

        elif "laion" in model_name:
            model = open_clip.create_model_and_transforms(model_name)[0].to(device)
        return model

    def __call__(
        self,
        samples: dict,
        num_tags: int = 5,
        num_attributes: int = 5,
        contrastive_th: float = 0.2,
        num_beams: int = 5,  # For beam search
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
        return_tags: bool = True,
        return_attributes: bool = True,
        return_global_caption: bool = True,
        return_intensive_captions: bool = True,
        return_complete_prompt: bool = True,
        **kwargs,
    ):
        samples = self.forward_tags(
            samples, num_tags=num_tags, contrastive_th=contrastive_th
        )
        samples = self.forward_attributes(
            samples, num_attributes=num_attributes, contrastive_th=contrastive_th
        )

        return samples

    def forward_tags(
        self, samples: dict, num_tags: int = 5, contrastive_th: float = 0.2
    ):
        # Get Image Features
        tags = []
        try:
            image_features = self.clip_model.encode_image(
                samples["clip_image"].to(self.device)
            )
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = image_features @ self.tags_weights
        top_scores, top_indexes = text_scores.float().cpu().topk(k=num_tags, dim=-1)
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.vocab_tags[index] for index in filter_indexes]
            else:
                top_k_tags = []
            tags.append(top_k_tags)
        samples[f"tags"] = tags
        return samples

    def forward_attributes(
        self, samples: dict, num_attributes: int = 5, contrastive_th: float = 0.2
    ):
        # Get Image Features
        attributes = []
        try:
            image_features = self.clip_model.encode_image(
                samples["clip_image"].to(self.device)
            )
        except:
            image_features = self.clip_model.get_image_features(
                pixel_values=samples["clip_image"]
            )
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = image_features @ self.attributes_weights
        top_scores, top_indexes = (
            text_scores.float().cpu().topk(k=num_attributes, dim=-1)
        )
        for scores, indexes in zip(top_scores, top_indexes):
            filter_indexes = indexes[scores >= contrastive_th]
            if len(filter_indexes) > 0:
                top_k_tags = [self.vocab_attributes[index] for index in filter_indexes]
            else:
                top_k_tags = []
            attributes.append(top_k_tags)
        samples[f"attributes"] = attributes
        return samples


class LensProcessor:
    def __init__(
        self,
        clip_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ):
        self.clip_processor = self.load_clip_transform(clip_name)

    def load_clip_transform(self, model_name: str):
        if "openai" in model_name:
            return CLIPProcessor.from_pretrained(model_name)

        elif "laion" in model_name:
            return open_clip.create_model_and_transforms(model_name)[2]

    def __call__(self, images: Any, questions: str):
        try:
            clip_image = torch.stack([self.clip_processor(image) for image in images])
        except:
            clip_image = self.clip_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]

        return {
            "clip_image": clip_image,
            "questions": questions,
        }

