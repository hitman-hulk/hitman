from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate


raw_datasets = load_dataset("glue", "mrpc")
# print(raw_dataset)
raw_train_dataset = raw_datasets["train"]
# print(raw_train_dataset[0])
# print(raw_train_dataset.features)
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentence_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentence_2 = tokenizer(raw_datasets["train"]["sentence2"])
inputs = tokenizer("This is the first sentence.", "This is the second sentence.")
# print(inputs)
tokenizer.convert_ids_to_tokens(inputs["input_ids"])
tokenized_dataset = tokenizer(raw_datasets["train"]["sentence1"], raw_datasets["train"]["sentence2"],
    						  padding=True, truncation=True)

def tokenize_function(example):
	return tokenizer(example["sentence1"], example["sentence2"], truncation = True)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("mrpc")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(model, training_args, train_dataset=tokenized_datasets["train"], 
	eval_dataset=tokenized_datasets["validation"], data_collator=data_collator, tokenizer=tokenizer,
    compute_metrics=compute_metrics)

trainer.train()
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
preds = np.argmax(predictions.predictions, axis=-1)
metric = evaluate.load("glue", "mrpc")
print(metric.compute(predictions=preds, references=predictions.label_ids))