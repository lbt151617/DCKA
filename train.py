from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import logging
from torch.utils.data import DataLoader, Dataset
from models import BertForTCMClassification
from safetensors.torch import load_file
import numpy as np
import math
from tqdm import tqdm
import torch
import os
import pickle as pkl
from load_data import process_data
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


num_epoch = 10
# 加载训练集、测试集和验证集
datapath = 'data/data.pkl'
if os.path.exists(datapath):
    with open(datapath, 'rb') as f:
        train_data, test_data, dev_data = pkl.load(f)
    f.close()
else:
    train_data, test_data, dev_data = process_data()
    with open(datapath, 'wb') as f:
        pkl.dump([train_data, test_data, dev_data], f)
    f.close()
# train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=32)
# dev_dataloader = DataLoader(dev_data, batch_size=32)
config = AutoConfig.from_pretrained("zy-bert", num_labels=148)
model = BertForTCMClassification(config)
# checkpoint_path = 'cnn+rnn+crossattention/checkpoint-43180/model.safetensors'
# state_dict = load_file(checkpoint_path)
# model.load_state_dict(state_dict)
training_args = TrainingArguments(
    output_dir='./crossattention',
    eval_strategy='epoch',
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=200,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    learning_rate=2e-5,
    save_total_limit=3,
    weight_decay=0.01,
    logging_steps=200,  # 每隔多少步记录一次日志
    log_level='info',
    logging_dir='./logs'
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 计算各指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

# 开始训练
trainer.train()
test_results = trainer.evaluate(eval_dataset=dev_data)
print(test_results)