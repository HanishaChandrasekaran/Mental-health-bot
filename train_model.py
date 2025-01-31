import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Load the dataset
df = pd.read_csv("mental_health_conversation.csv")

# Ensure the columns are strings
df['conversation'] = df['conversation'].astype(str)
df['label'] = df['label'].astype(str)

# Check the data
print(f"Dataset columns: {df.columns}")
print(f"First few rows: \n{df.head()}")

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Label encoding (if needed)
label_map = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_map)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['conversation'], df['label'], test_size=0.2)

# Tokenize the texts
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

train_encodings = tokenize_function(train_texts.tolist())
val_encodings = tokenize_function(val_texts.tolist())

# Create a custom Dataset class
class MentalHealthDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and validation datasets
train_dataset = MentalHealthDataset(train_encodings, train_labels.tolist())
val_dataset = MentalHealthDataset(val_encodings, val_labels.tolist())

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results_dir',              # output directory
    num_train_epochs=3,                  # number of training epochs
    per_device_train_batch_size=8,       # batch size for training
    per_device_eval_batch_size=8,        # batch size for evaluation
    warmup_steps=500,                    # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # strength of weight decay
    logging_dir='./logs',                # directory for storing logs
    logging_steps=10,                    # log every 10 steps
    evaluation_strategy="epoch",         # evaluate after each epoch
    save_strategy="epoch",               # save after each epoch
)

# Trainer initialization
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
)

# Start training
trainer.train()
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("Model and tokenizer saved successfully!")