import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

# Environmental variables -- these should be stored in secrets or .env file
MODEL_NAME = "distilbert/distilbert-base-uncased"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
OUTPUT_DIR = "./" # Path to save model in Cloud Storage

# Initialize distributed training
#torch.distributed.init_process_group(backend="gloo") # use 'nccl' for GPUs
local_rank =  0 #torch.distributed.get_rank()
#world_size = torch.distributed.get_world_size()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the SST-2 dataset
dataset = load_dataset("sst2")

# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create data loaders with distributed sampler
#train_sampler = DistributedSampler(tokenized_dataset["train"], num_replicas=world_size, rank=local_rank)
train_loader = DataLoader(
    tokenized_dataset["train"], 
    #sampler=train_sampler,
    batch_size=BATCH_SIZE, 
    shuffle=True
)

# Load model and wrap with distributeddataparallel
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
#model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"] #.cuda()
        attention_mask = batch["attention_mask"] #.cuda()
        labels = batch["labels"] #.cuda()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask) #, labels=labels)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print(f"Epoch: {epoch+1}/{EPOCHS}, Batch: {idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        if idx==50:
            break
    if epoch > 0:
        break

# Save the trained model
if local_rank == 0: # Only save on the main process
    model.save_pretrained(OUTPUT_DIR) #model.module.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)