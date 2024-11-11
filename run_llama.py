import pandas as pd
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from llama_model import model, processor
from joblib import Parallel, delayed

csv_file = "generated_prompts.csv"
df = pd.read_csv(csv_file).head(10)

def process_row(image_url, prompt, max_tokens= 600):

    try:


        image = Image.open(requests.get(image_url, stream=True).raw)

      
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

     
        output = model.generate(**inputs, max_new_tokens=max_tokens)
        decoded_output = processor.decode(output[0], skip_special_tokens=True)


        return decoded_output
    except Exception as e:
        return f"Error: {str(e)}"

#for idx, row in df.iterrows():
#    process_row(row["Image URL"], row["Prompt"], max_tokens=600)


df["output"] = df.apply(lambda row: process_row(row["Image URL"], row["Prompt"], max_tokens=600), axis=1)
#def process_row_parallel(row)
#    return process_row(row["Image URL"], row["Prompt"], max_tokens= 600)

#df["output"] = Parallel(n_jobs=8)(delayed(process_row_parallel)(row) for _, row in df.iterrows())



output_csv_file = "llama_results.csv"
df.to_csv(output_csv_file, index=False)

print(f"Processing complete. Results saved to {output_csv_file}.")
