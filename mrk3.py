from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import torch


# classifier = pipeline("sentiment-analysis")
# print(classifier(["I've been waiting for a course my whole life.",
# 				  "I hate this so much!"]))

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = ["I've been waiting for a course my whole life.",
 			  "I hate this so much!"]
inputs = tokenizer(raw_inputs, padding = True, truncation = True, return_tensors = "pt")
# print(inputs)

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)
predictions = torch.nn.functional.softmax(outputs.logits, dim =- 1)
print(predictions)
print(model.config.id2label)