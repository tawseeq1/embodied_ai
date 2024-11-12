import os
import torch
import pandas as pd
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
from llama_model import model, processor

print(processor)
print(processor.model_max_length)
print(processor.image_size) 

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
print(model)
#model.to("cuda")
#model_save_path = "llama_model.pth"
#torch.save(model, model_save_path)

image_folder = "downloaded_images"

csv_file = "generated_prompts.csv"
df = pd.read_csv(csv_file)

def process_row(image_path, prompt, max_tokens=600):
    try:
        # if not os.path.exists(image_path): #displayed some error previously
        #     raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        #print(f"Image dimensions: {image.size}") prints succesfully
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        #print(f"Image input shape: {inputs.pixel_values.shape}")
        
        if inputs.pixel_values.shape[0] == 0:
            raise ValueError(f"Image token count for {image_path} : 0")
        
        output = model.generate(**inputs, max_new_tokens=max_tokens)
        decoded_output = processor.decode(output[0], skip_special_tokens=True)

        return decoded_output
    except Exception as e:
        return f"Error: {str(e)}"

df["Image Path"] = df.index.map(lambda idx: os.path.join(image_folder, f"image{idx + 1}.jpg"))
df["output"] = df.apply(lambda row: process_row(row["Image Path"], row["Prompt"], max_tokens=600), axis=1)

output_csv_file = "llama_results.csv"
df.to_csv(output_csv_file, index=False)

print(f"Processing complete. Results saved to {output_csv_file}.")
