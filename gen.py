import google.generativeai as genai
import csv
import requests
from images import image_urls  # Importing the list of image URLs
import os

# Configure the Gemini API
GOOGLE_API_KEY = 'AIzaSyACMoxbEHpwusW1q7pagB20xeQGotIvvA0'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Output CSV file
output_file = 'generated_prompts.csv'

# Prompt for the model
prompt_input = ('Assume that you are researching embodied AI; your goal is to evaluate the embodied AI question-answering capabilities of a multimodal large language model. To do this, create a prompt that would require the LLM to reason about a real-world scenario as a robot. Analyze the image and design only a scenario (in scenario, also mention the type of robot to be assumed) and related questions for this image that simulate this situation, avoiding any hints or additional information about object locations or functionalities. The questions should require the LLM to infer details from the image itself, focusing on physical actions for embodied question answering and should also ask about giving step by step approach for each question. (Please, no remarks at the end that you normally do after each response, because I want to directly copy the response into another LLM)')

# Open the CSV file for writing
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Image URL", "Prompt"])

    for image_url in image_urls:
        # Download and save the image locally
        image_filename = image_url.split("/")[-1]
        response = requests.get(image_url, stream=True)

        if response.status_code == 200:
            with open(image_filename, 'wb') as img_file:
                img_file.write(response.content)

            try:
                # Upload the image to Gemini
                uploaded_file = genai.upload_file(
                    path=image_filename,
                    display_name=image_filename,
                    mime_type="image/jpeg"
                )
                print(f"Uploaded file: {uploaded_file.display_name}")

                # Generate prompt for the image
                response = model.generate_content([uploaded_file, prompt_input])
                prompt_text = response.text.strip()

                # Write to CSV
                writer.writerow([image_url, prompt_text])

            except Exception as e:
                print(f"Error processing {image_url}: {e}")
        else:
            print(f"Failed to download image from {image_url}")

        # Cleanup local image file
        if os.path.exists(image_filename):
            os.remove(image_filename)

