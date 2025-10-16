# Disable Triton for compatibility with Windows
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.pop("TORCH_COMPILE_BACKEND", None)
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE", None)

import time
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from synthetic_data_v2.c0_Configuration.config_paths import ( 
   SYSTEM_PROMPT_TEMPLATE_PATH,
   SUMMARY_OUTPUT_FILE,
   FINAL_JSON_INPUT_LLM
)

# Constants
MAX_USERS = 1  # Limit the number of users for testing
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your model name

def load_prompt_template():
    """
    Load the system prompt template from the specified path.
    """
    with open(SYSTEM_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(system_prompt, input_json_file):
    """
    Build the final prompt by combining the system prompt with user data.
    """
    # Load the input JSON file
    with open(input_json_file, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    
    # Construct the user-specific part of the prompt
    user_prompts = []
    for user in user_data[:MAX_USERS]:  # Limit to MAX_USERS for testing
        user_prompt = (
            f"User ID: {user['user_id']}\n"
            f"Check-ins: {user['user_metadata']}\n"
            f"Cluster Metadata: {user['cluster_metadata']}\n"
            f"Probable Profile Tag: {user['Probable_user_profile_tag']}\n"
        )
        user_prompts.append(user_prompt)
    
    # Combine all user prompts
    combined_user_prompts = "\n\n".join(user_prompts)
    
    # Final prompt
    final_prompt = f"{system_prompt}\n\n{combined_user_prompts}\n\nGenerate summaries for all users."
    return final_prompt

def main():
    """
    Main function to generate user summaries using a language model.
    """
    # Load the system prompt template
    system_prompt = load_prompt_template()
    input_file = FINAL_JSON_INPUT_LLM
    final_prompt = build_prompt(system_prompt, input_file)
    
        # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")
    
    device = "cuda"  # Force GPU usage
    print(f"Using device: {device}")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, force_download=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, force_download=True).to(device)
    print(f"Model is running on: {device}")
    
    # Generate summaries for all users in one batch
    start_time = time.time()
    inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True).to(device)
    input_len = tokenizer(final_prompt, return_tensors="pt")["input_ids"].shape[-1]
    print(f"Input token length: {input_len}")
    if input_len > 4096:
        print("Warning: Input length exceeds 4096 tokens, which may lead to truncation.")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,  # Adjust based on token limit
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    end_time = time.time()
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Time taken for batch processing: {end_time - start_time:.2f} seconds")
    print(f"Generated Summaries:\n{output_text}")
    # save the file here - SUMMARY_OUTPUT_FILE
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    main()