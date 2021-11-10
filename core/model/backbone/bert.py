import torch
import torch.nn as nn

from transformers import AutoModel

class Bert(nn.Module):

    def __init__(self, num_labels, bert_model):
        super(Bert, self).__init__()
        self.num_labels = num_labels
        self.bert_model = bert_model
        self.model = AutoModel.from_pretrained(self.bert_model, num_labels=self.num_labels)

    def forward(self, x):
        input_ids, attention_mask, segment_ids = zip(*x)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        # segment_ids = torch.stack(segment_ids)
        return self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']#, token_type_ids=segment_ids)
