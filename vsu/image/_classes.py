from transformers import AutoImageProcessor, EfficientNetModel

import torch
from PIL import Image
import open_clip

from vsu.base.VectorSearchBase import VectorSearchBase


class VSU_Image_CLIP(VectorSearchBase):
    def __init__(self, db_name=':memory:'):
        super(VSU_Image_CLIP, self).__init__(db_name)

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
        return scores, pred

    # override
    def _trans_vec_main_func(self, ar):
        imgs = [Image.open(p) for p in ar]
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


class VSU_Image_EfficientNet(VectorSearchBase):
    def __init__(self, db_name=':memory:'):
        super(VSU_Image_EfficientNet, self).__init__(db_name)

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
        imgs = [Image.open(p) for p in ar]

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
