from utils import *

import torch
from torch.utils.data import DataLoader, Dataset
import tqdm
import time  # 添加time模块导入
# Import mnist dataset
from torchvision import datasets, transforms
import torchvision.models as models
from transformers import BertModel, BertConfig
import torchvision.transforms as transforms



class MNIST_Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 配置BERT模型参数
        self.config = BertConfig(
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=785,
            type_vocab_size=1,
            vocab_size=256
        )
        
        self.bert = BertModel(self.config)
        
        # 修改图像预处理层
        self.pixel_to_patch = torch.nn.Sequential(
            torch.nn.Conv2d(1, 768, kernel_size=1),  # [batch_size, 768, 28, 28]
            torch.nn.Flatten(2),  # [batch_size, 768, 784]
        )
        
        # 单独的LayerNorm层，用于序列维度
        self.layer_norm = torch.nn.LayerNorm(768)
        
        # 分类头
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 将图像转换为序列 [batch_size, 768, 784]
        x = self.pixel_to_patch(x)
        
        # 转置为BERT期望的输入格式 [batch_size, 784, 768]
        x = x.transpose(1, 2)
        
        # 对每个位置应用LayerNorm
        x = self.layer_norm(x)
        
        # 添加[CLS]标记的位置嵌入
        cls_tokens = torch.zeros(batch_size, 1, 768).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, 785).to(x.device)
        
        # 通过BERT
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        
        # 使用[CLS]标记的输出进行分类
        cls_output = outputs.last_hidden_state[:, 0]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits

    