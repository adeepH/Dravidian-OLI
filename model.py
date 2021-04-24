import torch
import torch.nn as nn
from transformers import AutoModel


class OffensiveModel(nn.Module):

    def __init__(self, model_name, num_classes, requires_grad=False):
        super(OffensiveModel, self).__init__()
        self.model_name = model_name
        self.base_model = AutoModel.from_pretrained(model_name)
        self.num_classes = num_classes
        self.requires_grad = requires_grad
        self.relu = nn.ReLU
        self.drop = nn.Dropout(p=0.2)

        for name, params in self.base_model.named_parameters():
            if "emb" in name:
                params.requires_grad = True
            else:
                params.requires_grad = self.requires_grad

        self.fc1 = nn.Linear(in_features=self.base_model.config.hidden_size, out_features=self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):

        model_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = self.avg_pool(model_output[0].permute(0, 2, 1)).squeeze(dim=1)
        cls_embedding = self.drop(cls_embedding)
        x = self.fc1(cls_embedding)
        return x


if __name__ == "__main__":
    num_class = 6
    model_name = 'xlm-roberta-base'

    Model = OffensiveModel(model_name, num_class)

    for name, params in Model.named_parameters():
        if "emb" in name:
            print(name)
            print(Model.parameters())