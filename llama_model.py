
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
from huggingface_hub import HfFolder

HfFolder.save_token("Enter Your own key mate")

#huggingface_token = os.getenv("HUGGINGFACE_TOKEN")  
'''processor = AutoProcessor.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", use_auth_token=huggingface_token
)
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision", use_auth_token=huggingface_token
)'''

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision", device_map="cuda")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision", device_map="cuda")
#model.tie_weights()
