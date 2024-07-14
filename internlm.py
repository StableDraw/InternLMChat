import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "weights"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)

# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.float16, trust_remote_code = True).cuda()
model = model.eval()

prompt = "diving"
segments_duration = 8

response, history = model.chat(tokenizer, f"I'm creating a clip from a neural network that generates video. The theme of my video is \"f{prompt}\". The clip consists of f{segments_duration} second segments. Please write prompts for the neural network that would describe each segment of this clip. Describe each segment separately. Write each prompt from the next line and no additional information", history=[])
print(response)

# Hello! How can I help you today?
response, history = model.chat(tokenizer, "Пожалуйста, сгенерируй про", history = history)
print(response)