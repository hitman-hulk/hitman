from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

raw_datasets = load_dataset("/home/akugyo/trial/")
raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
# print(raw_datasets)

# for key in raw_datasets["train"][0]:
#     print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

# outputs = tokenizer(
#     raw_datasets["train"][:2]["content"],
#     truncation=True,
#     max_length=context_length,
#     return_overflowing_tokens=True,
#     return_length=True,
# )

# print(f"Input IDs length: {len(outputs['input_ids'])}")
# print(f"Input chunk lengths: {(outputs['length'])}")
# print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
print(tokenized_datasets)