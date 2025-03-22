#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, ROC_AUC, Loss, Recall, Precision
from torchinfo import summary
from ignite.handlers import EarlyStopping

import random
import numpy as np
torch.use_deterministic_algorithms(True)
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


# # Load Dataset From Disk

# In[2]:


def load_data_from_file(data_filie, size):
    df = pd.read_csv(data_filie)
    if size > 0:
        df = df[1: size]
    raw_text = df['text'].tolist()
    print(df['label'].value_counts())
    raw_text_labels = [1 if sentiment == 'design' else 0 for sentiment in df['label'].tolist()]
    return raw_text, raw_text_labels

texts, labels = load_data_from_file('./combined_raw.csv', 0)


# In[3]:


class TextClassificationDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', 
                                  max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}



# In[4]:


bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 8

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained(bert_model_name)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer,  max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# ## Construct the Classifier

# In[5]:


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, 1) # chose 1 since this is binary classification
        
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    


# In[6]:


if torch.backends.mps.is_available():
    device = torch.device("cpu") 
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ## Define Training and Evaluation Functions

# In[7]:


learning_rate = 2e-5


model = BertClassifier(bert_model_name, num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)


# In[8]:


# model summary
data_iter = iter(train_dataloader)
sample_batch = next(data_iter)  # Get first batch of data

if isinstance(sample_batch, dict):
    input_ids = sample_batch['input_ids']
    attention_mask = sample_batch['attention_mask']
elif isinstance(sample_batch, (list, tuple)):  
    input_ids, attention_mask, _ = sample_batch 
else:
    raise TypeError("Unexpected data format from DataLoader")

summary(model, input_data=(input_ids, attention_mask), 
        col_names=["input_size", "output_size", "num_params"], device=device)


# In[9]:


def train_step(engine, batch):
    model.train()
    
    optimizer.zero_grad()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    labels = batch['label'].to(device, dtype=torch.float32)
    
    labels = labels.unsqueeze(1)
  
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    loss = nn.BCEWithLogitsLoss()(outputs, labels)
    loss.backward()
    
    optimizer.step()
    
    return loss.item()

def evaluation_step(engine, batch):
    model.eval()
    
    with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            prediction_probability = torch.sigmoid(logits)
            predictions = (prediction_probability >= 0.5).float()
            
            actual_labels = labels
        
    return logits, prediction_probability, predictions,  actual_labels
        


    
trainer = Engine(train_step)
evaluator  = Engine(evaluation_step)

@trainer.on(Events.EPOCH_COMPLETED(every=1))
def run_validation():
    evaluator.run(val_dataloader)
    


@trainer.on(Events.EPOCH_COMPLETED(every=1))
def log_validation():
    metrics = evaluator.state.metrics
    
    auc_scores.append(metrics['roc_auc'])
    precision.append(metrics['precision'])
    recall.append(metrics['recall'])
    accuracy_scores.append(metrics['accuracy'])
    bce_losses.append(metrics['bce_loss'])
    f1_score.append(metrics['f1_score'])
    
    
    epochs.append(trainer.state.epoch)
    
    
    results = {
        'epoch': epochs,
        'auc_score': auc_scores,
        'accuracy_score': accuracy_scores,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    
    results_df = pd.DataFrame(data=results)
    results_df.to_csv(f"./results/saved_metrics/{bert_model_name}.csv", index=False)
    
    print(f"Epoch: {trainer.state.epoch}, Accuracy: {metrics['accuracy']},  ROC_AUC: {metrics['roc_auc']} BCE_LOSS: {metrics['bce_loss']}")


Accuracy(output_transform= lambda x: (x[2], x[3])).attach(evaluator, "accuracy")
ROC_AUC(output_transform=lambda x: (x[1], x[3])).attach(evaluator, "roc_auc")

recall = Recall(output_transform=lambda x: (x[2], x[3]), average=False)
recall.attach(evaluator, 'recall')

precision = Precision(output_transform=lambda x: (x[2], x[3]), average=False)
precision.attach(evaluator, "precision")

f1_score = (precision * recall * 2 / (precision + recall))
f1_score.attach(evaluator, 'f1_score')


# ## Early Stopping

# In[10]:


loss_criterion  = torch.nn.BCEWithLogitsLoss()
loss_metric = Loss(loss_criterion, output_transform=lambda x: (x[0],x[2]))

loss_metric.attach(evaluator, 'bce_loss')
def score_function(engine):
    val_loss = engine.state.metrics['bce_loss']
    return -val_loss
    
early_stopping_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)


# In[11]:


epochs = []
auc_scores = []
accuracy_scores = []
bce_losses = [] 
precision = []
recall = []
f1_score =[]

trainer.run(train_dataloader, 20)


# ## Save Model

# In[ ]:


torch.save(model.state_dict(), f"./results/saved_models/{bert_model_name}.pt")
print("training complemented")


# ## Results Plotting

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(epochs, auc_scores, 'r--', epochs, accuracy_scores, 'b--')
plt.legend(['AUC Score', 'Validation Accuracy',])


# 

# In[ ]:




