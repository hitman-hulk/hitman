from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset
import torch


transcriber = pipeline(task="automatic-speech-recognition")
print(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))
transcriber = pipeline(model="openai/whisper-large-v2")
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

transcriber = pipeline(model="openai/whisper-large-v2", my_parameter=1)
out = transcriber(...)  # This will use `my_parameter=1`.
out = transcriber(..., my_parameter=2)  # This will override and use `my_parameter=2`.
out = transcriber(...)  # This will go back to using `my_parameter=1`.
transcriber = pipeline(model="openai/whisper-large-v2", device=0)
transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")

transcriber = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = transcriber(audio_filenames)
transcriber = pipeline(model="openai/whisper-large-v2", return_timestamps=True)
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
transcriber = pipeline(model="openai/whisper-large-v2", chunk_length_s=30, return_timestamps=True)
transcriber("https://huggingface.co/datasets/sanchit-gandhi/librispeech_long/resolve/main/audio.wav")

def data():
	for i in range(1000):
		yield f"My Example {i}"

pipe = pipeline(model = "gpt2", device = 0)
generated_characters = 0 
for out in pipe(data()):
	generated_characters += len(out[0]["generated_text"])

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
pipe = pipeline(model="facebook/opt-1.3b", device_map="auto", model_kwargs={"load_in_8bit": True})
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)
print(output)