import csv
import requests
from PIL import Image
import torch
import pandas as pd

from images import image_urls
from models import molmo_processor, molmo_model

device_count = torch.cuda.device_count()
molmo_model.share_memory()
input_csv = pd.read_csv('generated_prompts.csv')
output_csv_path = "molmo_output_with_responses.csv"

def process_row(args):
   
    row, gpu_id = args
    image_url, prompt = row[0], row[1]
    
    try:
        device = torch.device(f"cuda:{gpu_id}")
        model = molmo_model.to(device)

 
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = molmo_processor.process(images=[image], text=prompt)

        
        for k, v in inputs.items():
            if k == 'input_ids':
                inputs[k] = v.to(torch.long)
            elif v.dtype != torch.float16:
                inputs[k] = v.half()
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=molmo_processor.tokenizer
        )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = molmo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return row + [response]
    
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return row + ["Error processing this image"]

def main():
    # Read the input CSV
    with open(input_csv_path, "r") as infile:
        reader = list(csv.reader(infile))
        header = reader[0]
        rows = reader[1:]

    # Prepare data for multiprocessing
    gpu_allocation = [(row, i % device_count) for i, row in enumerate(rows)]
    
    # Use multiprocessing with GPUs
    set_start_method('spawn', force=True)  # Needed for multiprocessing
    with Pool(processes=device_count) as pool:
        results = pool.map(process_row, gpu_allocation)
    
    # Write results to the output CSV
    with open(output_csv_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header + ["Model_Response"])
        writer.writerows(results)

    print(f"Responses have been written to {output_csv_path}")

if __name__ == "__main__":
    main()


