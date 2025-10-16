# # Disable Triton for compatibility with Windows
# import os
# os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
# os.environ.pop("TORCH_COMPILE_BACKEND", None)
# os.environ.pop("TORCH_LOGS", None)
# os.environ.pop("TORCH_COMPILE", None)

# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# import time
# import json
# import csv
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from c0_Configuration.s00_config_paths import ( 
#    SYSTEM_PROMPT_TEMPLATE_PATH,
#    SUMMARY_OUTPUT_FILE,
#    FINAL_JSON_INPUT_LLM
# )

# # Constants
# MAX_USERS = 1  # Limit the number of users for testing
# # MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with your model name
# # MODEL_NAME = "facebook/bart-large-cnn"
# MODEL_NAME = "openai/gpt-oss-20b"

# MAX_INPUT_LENGTH = 2048

# def load_prompt_template():
#     """
#     Load the system prompt template from the specified path.
#     """
#     with open(SYSTEM_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# def build_prompt(system_prompt, input_json_file):
#     """
#     Build the final prompt by combining the system prompt with user data.
#     """
#     # Load the input JSON file
#     with open(input_json_file, "r", encoding="utf-8") as f:
#         user_data = json.load(f)
    
#     # Construct the user-specific part of the prompt
#     user_prompts = []
#     for user in user_data[:MAX_USERS]:  # Limit to MAX_USERS for testing
#         user_prompt = (
#             f"User ID: {user['user_id']}\n"
#             f"Check-ins: {user['user_metadata']}\n"
#             f"Cluster Metadata: {user['cluster_metadata']}\n"
#             f"Probable Profile Tag: {user['Probable_user_profile_tag']}\n"
#         )
#         user_prompts.append(user_prompt)
    
#     # Combine all user prompts
#     combined_user_prompts = "\n\n".join(user_prompts)
    
#     # Final prompt
#     final_prompt = f"{system_prompt}\n\n{combined_user_prompts}\n\nGenerate summaries for all users."
#     return final_prompt

# def main():
#     """
#     Main function to generate user summaries using a language model.
#     """
#     # Load the system prompt template
#     system_prompt = load_prompt_template()
#     input_file = FINAL_JSON_INPUT_LLM
#     final_prompt = build_prompt(system_prompt, input_file)
    
#         # Force GPU usage
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")
    
#     device = "cuda"  # Force GPU usage
#     print(f"Using device: {device}")

#     # Load the model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # if force_download is True, the code will force download the model files again. 
#     # the transformer library will not be caching the model files 
#     model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
#     print(model.config.max_position_embeddings)
#     print(f"Model is running on: {device}")
    
#     # Generate summaries for all users in one batch
#     start_time = time.time()
#     inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
#     input_len = inputs["input_ids"].shape[-1]
#     print(f"Input token length: {input_len}")
#     # if input_len > 4096: this is for llama
#     if input_len > MAX_INPUT_LENGTH: # 
#         print(f"Warning: Input length exceeds {MAX_INPUT_LENGTH} tokens, which may lead to truncation.")
#     outputs = model.generate(
#         **inputs, 
#         max_new_tokens=4096,  # Adjust based on token limit
#         do_sample=True,
#         top_k=50,
#         top_p=0.95
#     )
#     end_time = time.time()
    
#     # Decode the output
#     output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Time taken for batch processing: {end_time - start_time:.2f} seconds")
#     final_answer = output_text.strip()
#     print(f"Generated Summaries:\n{final_answer}")
#     # save the file here - SUMMARY_OUTPUT_FILE
#     with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
#         f.write(final_answer)

# if __name__ == "__main__":
#     torch._dynamo.config.suppress_errors = True
#     main()

# FOR ONE USER 



# FOR MULTIPLE USERS

# Disable Triton for compatibility with Windows
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.pop("TORCH_COMPILE_BACKEND", None)
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE", None)

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from synthetic_data_v2.c0_Configuration.config_paths import ( 
   SYSTEM_PROMPT_TEMPLATE_PATH,
   SUMMARY_OUTPUT_FILE,
   FINAL_JSON_INPUT_LLM
)

# Constants
MODEL_NAME = "openai/gpt-oss-20b"
MAX_INPUT_LENGTH = 2048  # Maximum context length for the model
MAX_OUTPUT_LENGTH = 512  # Maximum output length for the model

def load_prompt_template():
    """
    Load the system prompt template from the specified path.
    """
    with open(SYSTEM_PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(system_prompt, user):
    """
    Build the final prompt for a single user by combining the system prompt with user data.
    """
    user_prompt = (
        f"User ID: {user['user_id']}\n"
        f"Check-ins: {user['user_metadata']}\n"
        f"Cluster Metadata: {user['cluster_metadata']}\n"
        f"Probable Profile Tag: {user['Probable_user_profile_tag']}\n"
    )
    final_prompt = f"{system_prompt}\n\n{user_prompt}\n\nGenerate a summary for this user."
    return final_prompt

def generate_summary(model, tokenizer, prompt, device):
    """
    Generate a summary for a single user using the model.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
    input_len = inputs["input_ids"].shape[-1]
    
    if input_len > MAX_INPUT_LENGTH:
        print(f"Warning: Input length exceeds {MAX_INPUT_LENGTH} tokens. The input will be truncated.")
    
    # Generate the output
    outputs = model.generate(
        **inputs, 
        max_new_tokens=MAX_OUTPUT_LENGTH,  # Adjust based on token limit
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text.strip()

def main():
    """
    Main function to generate user summaries using a language model.
    """
    # Load the system prompt template
    system_prompt = load_prompt_template()
    
    # Load the input JSON file
    with open(FINAL_JSON_INPUT_LLM, "r", encoding="utf-8") as f:
        user_data = json.load(f)
    
    # Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")
    
    device = "cuda"  # Force GPU usage
    print(f"Using device: {device}")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    print(f"Model is running on: {device}")
    
    # Generate summaries for each user
    all_summaries = {}
    for user in user_data:
        print(f"Processing User ID: {user['user_id']}")
        prompt = build_prompt(system_prompt, user)
        summary = generate_summary(model, tokenizer, prompt, device)
        all_summaries[user['user_id']] = summary
        print(f"Summary for User ID {user['user_id']}:\n{summary}\n")
    
    # Save all summaries to the output file
    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4)
    print(f"Summaries saved to {SUMMARY_OUTPUT_FILE}")

if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    main()

