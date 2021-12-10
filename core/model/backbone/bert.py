import torch
import torch.nn as nn

from transformers import AutoModel

class Bert(nn.Module):

    def __init__(self, bert_model):
        super(Bert, self).__init__()
        self.bert_model = bert_model
        self.model = AutoModel.from_pretrained(self.bert_model)

    def forward_single_batch(self, input_ids, attention_mask, segment_ids):
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return self.model.pooler(last_hidden_state)

    def forward(self, x):
        one_to_one_augmented = len(x[0]) == 6
        if one_to_one_augmented:
            input_ids, attention_mask, segment_ids, aug_input_ids, aug_attention_mask, aug_segment_ids = zip(*x)
            orig_feat = self.forward_single_batch(input_ids, attention_mask, segment_ids)
            aug_feat = self.forward_single_batch(aug_input_ids, aug_attention_mask, aug_segment_ids)
            return torch.mul(torch.add(orig_feat, aug_feat), 0.5)

        input_ids, attention_mask, segment_ids = zip(*x)
        return self.forward_single_batch(input_ids, attention_mask, segment_ids)
