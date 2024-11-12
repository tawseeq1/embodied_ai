import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoProcessor, GenerationConfig

molmo_model_id = 'allenai/Molmo-7B-O-0924'
molmo_processor = AutoProcessor.from_pretrained(
    molmo_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # float16 for faster computation on GPU
    device_map='balanced'
)
molmo_model = AutoModelForCausalLM.from_pretrained(
    molmo_model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='balanced'
)
