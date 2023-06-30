from packaging import version
from onnxruntime import __version__ as ort_version
from transformers import AutoConfig
import torch

import os

# Create a cache directory to store pretrained model.
cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if version.parse(ort_version) >= version.parse("1.12.0"):
    from onnxruntime.transformers.models.gpt2.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
else:
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel

    raise RuntimeError("Please install onnxruntime 1.12.0 or later to run this notebook")

model_name_or_path = "gpt2"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

print(model.config)

num_attention_heads = model.config.n_head
hidden_size = model.config.n_embd
num_layer = model.config.n_layer

onnx_model_path = "gpt2.onnx"

