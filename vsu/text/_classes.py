import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np

from vsu.base.VectorSearchBase import VectorSearchBase


class VSU_Text_E5(VectorSearchBase):
    def __init__(self, save_name=None, echo=False):
        super(VSU_Text_E5, self).__init__(save_name, echo=echo)

    # override
    def init_model(self):
        model_name = 'intfloat/multilingual-e5-small'

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.vec_size = self.model.embeddings.word_embeddings.embedding_dim

    def __average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # override
    def do_zeroshot(self):
        if self.zeroshot_vec is None:
            return

        v = F.normalize(Tensor(self.data["vector"]), p=2, dim=1)

        z = np.array(F.normalize(Tensor(self.zeroshot_vec), p=2, dim=1))
        scores = (v @ z.T) * 100

        pred = []
        for s in scores:
            idx = s.tolist().index(max(s))
            pred.append(self.zeroshot_labels[idx])

        self.data["zeroshot_pred"] = pred
        return scores, pred

    # override
    def _trans_vec_main_func(self, ar):
        prefix = self.config.get('query_prefix', '')

        ar = [a for a in ar]
        batch_dict = self.tokenizer(ar, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        features = embeddings.tolist()

        return features

    # override
    def _trans_vec_sub_func(self, ar):
        return self._trans_vec_main_func(ar)

    # override
    def query(self, q, k=5):
        q = 'query: '+q
        return super().query(q, k)
