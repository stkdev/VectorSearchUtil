from transformers import AutoImageProcessor, EfficientNetModel
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature

from typing import Union, List
import ftfy, html, re, io

import torch
from PIL import Image
import open_clip

import pandas as pd

from vsu.base.VectorSearchBase import VectorSearchBase


class VSU_Image_CLIP(VectorSearchBase):
    """
    https://github.com/mlfoundations/open_clip
    """
    def __init__(self, save_name=None, echo=False):
        super(VSU_Image_CLIP, self).__init__(save_name, echo=echo)

    # override
    def init_model(self):
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.vec_size = self.model.token_embedding.embedding_dim

    # override
    def do_zeroshot(self):
        if self.zeroshot_vec is None:
            return

        image_features = torch.tensor(self.data["vector"])
        text_features = torch.tensor(self.zeroshot_vec)

        scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        pred = []
        for s in scores:
            idx = s.tolist().index(max(s))
            pred.append(self.zeroshot_labels[idx])

        self.data["zeroshot_pred"] = pred
        self.data["zeroshot_one_score"] = [float(s[0]) for s in scores]
        return scores, pred

    def do_zeroshot_detail(self, zeroshot_dict):
        if zeroshot_dict is None:
            return

        df_scores = None

        for d in zeroshot_dict:
            arr = zeroshot_dict[d]
            image_features = torch.tensor(self.data["vector"])
            text_features = torch.tensor(self._trans_vec_sub_func(arr))

            scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = pd.DataFrame(scores, columns=[f"{d}_{ar}" for ar in arr])

            if df_scores is None:
                df_scores = scores
            else:
                df_scores = pd.concat([df_scores, scores], axis=1)

        return df_scores

    # override
    def _trans_vec_main_func(self, ar):
        imgs = [Image.open(p) if type(p)==str else p for p in ar]
        images = [self.preprocess(img).unsqueeze(0) for img in imgs]
        image_features = [self.model.encode_image(image) for image in images]
        image_features = [vec / vec.norm(dim=-1, keepdim=True) for vec in image_features]
        image_features = [vec.tolist()[0] for vec in image_features]

        return image_features

    # override
    def _trans_vec_sub_func(self, ar):
        text = self.tokenizer(ar)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        v = text_features.tolist()
        return v



class VSU_Image_JP_CLIP(VectorSearchBase):
    """
    https://huggingface.co/stabilityai/japanese-stable-clip-vit-l-16
    """
    def __init__(self, save_name=None, echo=False):
        super(VSU_Image_JP_CLIP, self).__init__(save_name, echo=echo)

    def basic_clean(self, text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def whitespace_clean(self, text):
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def tokenize(self,
                 tokenizer,
                 texts: Union[str, List[str]],
                 max_seq_len: int = 77,
                 ):
        """
        This is a function that have the original clip's code has.
        https://github.com/openai/CLIP/blob/main/clip/clip.py#L195
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [self.whitespace_clean(self.basic_clean(text)) for text in texts]

        inputs = tokenizer(
            texts,
            max_length=max_seq_len - 1,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )
        # add bos token at first place
        input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
        attention_mask = [[1] + am for am in inputs["attention_mask"]]
        position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

        return BatchFeature(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
            }
        )

    # override
    def init_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "stabilityai/japanese-stable-clip-vit-l-16"
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, cache_dir='model', token=self.hf_token).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='model', token=self.hf_token)
        self.processor = AutoImageProcessor.from_pretrained(model_path, cache_dir='model', token=self.hf_token)

        # self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.vec_size = self.model.vision_embed_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # override
    def do_zeroshot(self):
        if self.zeroshot_vec is None:
            return

        image_features = torch.tensor(self.data["vector"])
        text_features = torch.tensor(self.zeroshot_vec)

        scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        pred = []
        for s in scores:
            idx = s.tolist().index(max(s))
            pred.append(self.zeroshot_labels[idx])

        self.data["zeroshot_pred"] = pred
        self.data["zeroshot_one_score"] = [float(s[0]) for s in scores]
        return scores, pred


    # override
    def _trans_vec_main_func(self, ar):
        imgs = [Image.open(p) if type(p)==str else p for p in ar]
        images = [self.processor(img, return_tensors="pt").to(self.device) for img in imgs]
        image_features = [self.model.get_image_features(**image) for image in images]
        # image_features = [vec / vec.norm(dim=-1, keepdim=True) for vec in image_features]
        # image_features = [vec.tolist()[0] for vec in image_features]

        return image_features

    # override
    def _trans_vec_sub_func(self, ar):
        text = self.tokenize(tokenizer=self.tokenizer, texts=ar).to(self.device)
        text_features = self.model.get_text_features(**text)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

        v = text_features.tolist()
        return v



class VSU_Image_EfficientNet(VectorSearchBase):
    def __init__(self, save_name=None, echo=False):
        super(VSU_Image_EfficientNet, self).__init__(save_name, echo=echo)

    # override
    def init_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
        self.model = EfficientNetModel.from_pretrained("google/efficientnet-b0")
        self.vec_size = self.model.config.hidden_dim

    # override
    def do_zeroshot(self):
        return

    # override
    def _trans_vec_main_func(self, ar):
        imgs = [Image.open(p) if type(p)==str else p for p in ar]

        images = []
        for img in imgs:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(self.image_processor(img, return_tensors="pt"))

        image_features = []

        with torch.no_grad():
            for inputs in images:
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                image_features.append(last_hidden_states.mean(dim=[0, 2, 3]).tolist())

        # image_features if type(image_features) == type([]) else
        return image_features

    # override
    # def _trans_vec_sub_func(self, ar):
    #   text = self.tokenizer(ar)
    #   text_features = self.model.encode_text(text)
    #   text_features /= text_features.norm(dim=-1, keepdim=True)

    #   v = text_features.tolist()
    #   return v
