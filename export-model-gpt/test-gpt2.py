from transformers import AutoConfig, AutoTokenizer
import torch
import os
import onnx
import onnxruntime 

cache_dir = os.path.join(".", "cache_models")

from packaging import version
from onnxruntime import __version__ as ort_version

if version.parse(ort_version) >= version.parse("1.12.0"):
    from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
else:
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
    raise RuntimeError("Please install onnxruntime 1.12.0 or later to run this notebook")

from transformers import GPT2LMHeadModel

model_name_or_path = "gpt2"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

print(model.config)

num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer

EXAMPLE_Text = ["best hotel in bay area", "here is an example of gpt2 model"]

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_example_inputs(prompt_text=EXAMPLE_Text):
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict["input_ids"], dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], dtype=torch.int32)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(position_ids < 0, 0)
    position_ids = position_ids.to(torch.int32)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids, attention_mask, position_ids, empty_past

config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

torch_model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
torch_model.eval().to(device)

input_ids, attention_mask, position_ids, empty_past = get_example_inputs()
print("input_ids", input_ids)
print("attention_mask", attention_mask)
print("position_ids", position_ids)

with torch.no_grad():
    torch_output = torch_model(
        input_ids, past_key_values=empty_past, attention_mask=attention_mask, position_ids=position_ids
    )

tokenizer = get_tokenizer(model_name_or_path, cache_dir)
for generated_sequence_idx, generated_sequence in enumerate(torch_output):
 print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
 #generated_sequence = generated_sequence.tolist()
 text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
 print(text)

print("==================Torch output======================================")
print(torch_output)
print("========================================================")

print("Testing model by loading from onnx model via onnxruntime")
print("========================================================")

import onnxruntime
import numpy

onnx_model_path = "gpt2.onnx"
input_ids, attention_mask, position_ids, empty_past = get_example_inputs()

# converting to numpy
session = onnxruntime.InferenceSession(onnx_model_path)
ort_inputs = {
    "input_ids": numpy.ascontiguousarray(input_ids.cpu().numpy()),
    "attention_mask": numpy.ascontiguousarray(attention_mask.cpu().numpy()),
    "position_ids": numpy.ascontiguousarray(position_ids.cpu().numpy()),
}

for i, past_i in enumerate(empty_past):
    ort_inputs[f"past_{i}"] = numpy.ascontiguousarray(past_i.cpu().numpy())
ort_outputs = session.run(None, ort_inputs)

logits_masked_diff = (torch_output[0] - ort_outputs[0]) * attention_mask.unsqueeze(2)
max_logits_diff = logits_masked_diff.abs().max()
print("max logits diff (ignored padding)", max_logits_diff)

print("=================Onnx Runtime output ===================")
print(ort_outputs)
print("========================================================")
