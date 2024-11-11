
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from huggingface_hub import HfFolder

HfFolder.save_token("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")

#huggingface_token = os.getenv("HUGGINGFACE_TOKEN")  
'''processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", use_auth_token=huggingface_token
)
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", use_auth_token=huggingface_token
)'''

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision", device_map="balanced")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision", device_map="balanced")
model.tie_weights()
