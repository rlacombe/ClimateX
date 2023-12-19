import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, DataCollatorWithPadding, EvalPrediction
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from transformers import TrainingArguments, Trainer
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()

# Define your training parameters
batch_size = 256 # Adjust to GPU memory (works on T4)
max_length = 512 # ClimateX train set max length:  
train_size = 0.85 
learning_rate = 1e-4
epochs = 1
eval_steps = 5 # Can reduce after hyperparams search

# Load your dataset into a pandas DataFrame
df = pd.read_csv('data/ipcc_statements_dataset.tsv', sep='\t')

# Define your labels and corresponding class names
class_names = ['low', 'medium', 'high', 'very high']

# Define the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(class_names),
    output_attentions=False,
    output_hidden_states=False,
)

# Freeze the BERT layers
for param in model.bert.parameters():
    param.requires_grad = False

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        statement = str(self.data.loc[index, 'statement'])
        score = self.data.loc[index, 'score']

        inputs = self.tokenizer(
            statement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(score, dtype=torch.long)
        }

# Filter the DataFrame to get the training set
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Create the custom dataset
dataset = CustomDataset(train_df, tokenizer, max_length)
test_dataset = CustomDataset(test_df, tokenizer, max_length)

# Split the dataset into training and validation sets
train_size = int(train_size * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a custom data collator that handles batch-level truncation
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest',  # Pad to the length of the longest sequence in the batch
)

# Define metrics to compute
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./bert_classifier',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='steps',
    eval_steps=eval_steps,
    logging_dir='./logs',
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=True,
    push_to_hub=False,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Define a custom evaluation function
def evaluate_accuracy(loader):
    model.eval()
    all_preds = []
    all_labels = []

    device = torch.device("cuda")

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Training loop with accuracy evaluation at each epoch
trainer.train()

# Final eval on val set
val_accuracy = evaluate_accuracy(val_loader)
print(f'Val Set Accuracy: {val_accuracy * 100:.2f}%')

# Final eval on test set
test_accuracy = evaluate_accuracy(test_loader)
print(f'Test Set Accuracy: {test_accuracy * 100:.2f}%')

# Save the fine-tuned model
model.save_pretrained(f"./bert_classifier_{epochs}ep_lr{learning_rate}")