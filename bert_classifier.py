import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from transformers import TrainingArguments, Trainer
import numpy as np

# Load your dataset into a pandas DataFrame
# Replace 'your_data.csv' with your actual dataset file
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

# Define your training parameters
batch_size = 32
train_size = 0.9
learning_rate = 3e-4
epochs = 10

# Filter the DataFrame to get the training set
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

# Compute the maximum sequence length based on the training set
max_length = min(train_df['statement'].apply(lambda x: len(tokenizer.encode(x))).max(), 512)

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

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./bert_classifier',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    logging_dir='./logs',
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=None,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Define a custom evaluation function
def evaluate_accuracy(loader):
    model.eval()
    all_preds = []
    all_labels = []

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
for epoch in range(epochs):
    trainer.train()

    # Evaluate accuracy on the validation set
    val_accuracy = evaluate_accuracy(val_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy * 100:.2f}%')

# Final eval on test set
test_accuracy = evaluate_accuracy(test_loader)
print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy * 100:.2f}%')


# Save the fine-tuned model
model.save_pretrained('./bert_fine_tuned')

# You can now use the fine-tuned model for inference or further evaluation.
