from transformers import pipeline


classifier = pipeline("sentiment-analysis")
print(classifier(["i have been waiting for this course for my whole life.", "i hate this a lot."]))

classifier = pipeline("zero-shot-classification")
print(classifier("this is the course about transformers library", candidate_labels = ["education", "politics", "business"]))

generator = pipeline("text-generation", model="distilgpt2")
print(generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2))