# %%
import pandas as pd
import numpy as np

import os
cwd = os.getcwd()
kaggle = cwd == "/kaggle/working"

pretrain = pd.read_parquet(("/kaggle/input/latsis-experiments/" if kaggle else "") + "german_datasets.parquet")
# train = pd.read_parquet(("/kaggle/input/latsis-experiments/" if kaggle else "") + "train.parquet")
# test = pd.read_parquet(("/kaggle/input/latsis-experiments/" if kaggle else "") + "test.parquet")

train = pd.read_csv(("/kaggle/input/latsis-experiments/" if kaggle else "") + "train.csv")
dev = pd.read_csv(("/kaggle/input/latsis-experiments/" if kaggle else "") + "dev.csv")

#convert to string
pretrain["text"] = pretrain["text"].astype(str)
train["text"] = train["text"].astype(str)
dev["text"] = dev["text"].astype(str)

#merge train and dev
full = pd.concat([train, dev])

#remove backslashes in the text
full["text"] = full["text"].str.replace("\\", "").replace("\"\"\"\"", "\"")

#split train and test
from sklearn.model_selection import train_test_split
train, test = train_test_split(full, test_size=0.2, random_state=42)

#keep only the first 1000 rows
# pretrain = pretrain[:1000]
# train = train[:100]
# test = test[:2500]

# %%
# for row in train.itertuples():
#     text = row.text
#     label = row.label
#     print(label)
#     print(text)

# %%
train["label"].sum()/len(train)

# %%
import torch
from torch import nn
from transformers import TrainingArguments, Trainer, AutoTokenizer, XLMRobertaTokenizerFast, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, TrainerCallback, TrainerControl, TrainingArguments


# model_name = 'xlm-roberta-base'
# model_name = "microsoft/mdeberta-v3-base"
model_name = 'aari1995/German_Semantic_STS_V2'
# model_name = "deepset/gelectra-base"
# model_name = "PM-AI/sts_paraphrase_xlm-roberta-base_de-en"
# model_name = "intfloat/multilingual-e5-large"
# model_name = "deutsche-telekom/gbert-large-paraphrase-euclidean"
# model_name = "ZurichNLP/swissbert"
# model_name = "xlm-roberta-large"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if model_name == "ZurichNLP/swissbert":
    model.set_default_language("de_CH")


config = model.config
tokenizer.model_max_length = config.max_position_embeddings


print("model parameters:" + str(sum(p.numel() for p in model.parameters())))

# %%
from torch.utils.data import Dataset
import torch
import numpy as np

max_length = 128

def encode_texts(tokenizer, texts):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    return torch.tensor(input_ids), torch.tensor(attention_masks)

pretrain_x, pretrain_attention_mask = encode_texts(tokenizer, pretrain['text'])
pretrain_y = torch.tensor(np.array(pretrain['label'].tolist()), dtype=torch.float32)

train_x, train_attention_mask = encode_texts(tokenizer, train['text'])
train_y = torch.tensor(np.array(train['label'].tolist()), dtype=torch.float32)

test_x, test_attention_mask = encode_texts(tokenizer, test['text'])
test_y = torch.tensor(np.array(test['label'].tolist()), dtype=torch.float32)

warmup_x, warmup_attention_mask = encode_texts(tokenizer, train['text'][:1000])
warmup_y = torch.tensor(np.array(train['label'].tolist())[:1000], dtype=torch.float32)

#full official dataset for submission
full_x, full_attention_mask = encode_texts(tokenizer, full['text'])
full_y = torch.tensor(np.array(full['label'].tolist()), dtype=torch.float32)

#mix of train and pretrain
mixed_x = torch.cat((train_x, pretrain_x))
mixed_attention_mask = torch.cat((train_attention_mask, pretrain_attention_mask))
mixed_y = torch.cat((train_y, pretrain_y))


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.label[idx],
        }

pretrain_dataset = CustomDataset(pretrain_x, pretrain_attention_mask, pretrain_y)
train_dataset = CustomDataset(train_x, train_attention_mask, train_y)
val_dataset = CustomDataset(test_x, test_attention_mask, test_y)
warmup_dataset = CustomDataset(warmup_x, warmup_attention_mask, warmup_y)
full_dataset = CustomDataset(full_x, full_attention_mask, full_y)
mixed_dataset = CustomDataset(mixed_x, mixed_attention_mask, mixed_y)

print(full_x.shape)


# %%
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

model = model.cuda()
optimizer = None



criterion = BCEWithLogitsLoss()

# Training function
def train(model, lr_per_epoch, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    for epoch in range(len(lr_per_epoch)):
        model.train()
        lr = lr_per_epoch[epoch]
        optimizer.param_groups[0]['lr'] = lr
        train_loss = 0.0

        # Training loop with tqdm
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)  # Remove the last dimension

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        if val_dataset is None:
            print(f"Train Loss: {avg_train_loss}")
            continue
        # Validation loop
        model.eval()
        val_loss = 0.0
        all_predictions_raw = []
        all_labels = []

        # Validation loop with tqdm
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            inputs = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['label'].cuda()

            with torch.no_grad():
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits.squeeze(-1)
                val_loss += criterion(logits, labels).item()
                pred = F.sigmoid(logits)
                
                all_predictions_raw.append(pred.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        

        all_predictions_raw = torch.cat(all_predictions_raw)
        all_labels = torch.cat(all_labels)
        accuracy = accuracy_score(all_labels.numpy(), all_predictions_raw.numpy() >= 0.5)
        f1 = f1_score(all_labels.numpy(), all_predictions_raw.numpy() >= 0.5, average='macro')
        print(f"Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Accuracy: {accuracy}, f1: {f1}")



        

        # print(f"First predictions:")
        # i = 0
        # for y_pred, y in zip(all_predictions_raw, all_labels):
        #     print(f"y_pred: {y_pred.item()}, y: {y}")
        #     i += 1
        #     if i > 5:
        #         break

        print("\n")

# Train the model

# for param in model.parameters():
#     param.requires_grad = False
# for param in model.classifier.parameters():
#     param.requires_grad = True
# for i in range(-3, 0):
#     for param in model.roberta.encoder.layer[i].parameters():
#         param.requires_grad = True
# optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-5)

# train(model, [1e-9, 1e-5, 1e-5, 1e-5], mixed_dataset, val_dataset)



for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
for i in range(-3, 0):
    # for param in model.roberta.encoder.layer[i].parameters():
    for param in model.bert.encoder.layer[i].parameters():
    # for param in model.deberta.encoder.layer[i].parameters():
    # for param in model.electra.encoder.layer[i].parameters():
        param.requires_grad = True
optimizer = AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-5)

train(model, [1e-9], warmup_dataset, val_dataset)
train(model, [1e-5], pretrain_dataset, val_dataset)
train(model, [1e-5, 1e-5, 1e-5, 1e-5, 2e-6, 1e-6], train_dataset, val_dataset)

# train(model, [1e-9], warmup_dataset, val_dataset)
# train(model, [1e-5, 1e-5, 1e-5, 1e-5, 2e-6, 1e-6], full_dataset, None)



# %%
# Save the model
# model_name = model_name.replace("/", "_")
# if not os.path.exists('ensemble'):
#     os.makedirs('ensemble')
# if not os.path.exists('ensemble/' + model_name):
#     os.makedirs('ensemble/' + model_name)
# torch.save(model, f'/kaggle/working/ensemble/{model_name}/model.pt' if kaggle else f'{model_name}.pt')
# torch.save(tokenizer, f'/kaggle/working/ensemble/{model_name}/tokenizer.pt' if kaggle else f'{model_name}_tokenizer.pt')
# print("saved")

# %%
#delete /kaggle/working/ensemble/ZurichNLP
# os.system("rm -rf /kaggle/working/model.pt")


