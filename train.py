import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from utils import train_epoch, eval_model, epoch_time
from loss import WeightAdjustingLoss, FocalLoss
from dataset import OffensiveDataset, create_data_loader
from model import OffensiveModel
from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_name = 'bert-base-multilingual-cased'
plt.style.use("ggplot")
BATCH_SIZE = 32
MAX_LEN = 128
EPOCHS = 4
history = defaultdict(list)
best_accuracy = 0
LOAD_MODEL = False

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
test_data_loader = create_data_loader(test_df, tokenizers, MAX_LEN, BATCH_SIZE, shuffle=False)
val_data_loader = create_data_loader(val_df, tokenizers, MAX_LEN, BATCH_SIZE, shuffle=False)

model = OffensiveModel()
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss = WeightAdjustingLoss().to(device)

for epoch in range(EPOCHS):
    start_time = time.time()
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss,
        optimizer,
        device,
        scheduler,
        6217
    )
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss,
        device,
        777
    )
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch::{epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Loss {train_loss} accuracy {train_acc}')
    print(f'Val Loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

 if val_acc > best_accuracy:
    torch.save(model.state_dict(),'bert-base-multilingual-cased.bin')
    best_accuracy = val_acc