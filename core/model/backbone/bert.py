import torch
import torch.nn as nn

from transformers import AutoModel

class Bert(nn.Module):

    def __init__(self, bert_model):
        super(Bert, self).__init__()
        self.bert_model = bert_model
        self.model = AutoModel.from_pretrained(self.bert_model)

    def forward(self, x):
        input_ids, attention_mask, segment_ids = zip(*x)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)

        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return self.model.pooler(last_hidden_state)
