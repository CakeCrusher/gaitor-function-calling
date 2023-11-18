# %%
function_calling_tokens = {
    "FUNCTIONS": {
        "start": "<FUNCTIONS>",
        "end": "</FUNCTIONS>"
    },
    "FUNCTION_CALL_NAME": {
        "start": "<FUNCTION_CALL_NAME>",
        "end": "</FUNCTION_CALL_NAME>"
    },
    "FUNCTION_CALL_ARGUMENTS": {
        "start": "<FUNCTION_CALL_ARGUMENTS>",
        "end": "</FUNCTION_CALL_ARGUMENTS>"
    },
    "all": ["<FUNCTIONS>", "</FUNCTIONS>", "<FUNCTION_CALL_NAME>", "</FUNCTION_CALL_NAME>", "<FUNCTION_CALL_ARGUMENTS>", "</FUNCTION_CALL_ARGUMENTS>"]
}

# %%
from datasets import load_dataset
import pandas as pd
from datasets import Dataset

relative_path_to_data = '../data/chat/production_train_chat-instruction.json'

dataset = load_dataset('json', data_files={'train': relative_path_to_data}, split="train")
base_model_name = "meta-llama/Llama-2-7b-chat-hf"
print(dataset[30]["text"])

filtered_data = [item for item in dataset if "</FUNCTION_CALL_ARGUMENTS>" in item["text"][-33:]]
print("Amount of data using FC:", len(filtered_data))

df = pd.DataFrame(filtered_data)
filtered_dataset = Dataset.from_pandas(df)

# %%
# Hugging Face model id
# model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
model_id = "meta-llama/Llama-2-7b-chat-hf" # gated

output_dir = "llama-7-chat-instruction-int4-fc-nat"

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

use_flash_attention = False

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


# %%
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=6,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True # disable tqdm since with packing values are in correct
)


# %%
from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=filtered_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_text_field="text",
    # formatting_func=format_instruction,
    args=args,
)


# %%
import wandb

wandb.init(project='fc_chat_instruction_prod', entity='sebastiansosa')


# %%
# train
trainer.train() # there will not be a progress bar since tqdm is disabled
# 4:30 mins per epoch
# 10:00 min per 2 epoch

# save model
trainer.save_model()


# %%
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json

# Set the path to the checkpoint directory
checkpoint_path = "/home/sosa.s/gaitor-function-calling/instruction_tune/llama-7-chat-instruction-int4-fc"

adapter_config_path = checkpoint_path + '/adapter_config.json'
try:
    with open(adapter_config_path, 'r') as file:
        config_data = json.load(file)

    # Update the 'base_model_name_or_path' field
    config_data['base_model_name_or_path'] = "meta-llama/Llama-2-7b-chat-hf"

    # Write the updated data back to the file
    with open(adapter_config_path, 'w') as file:
        json.dump(config_data, file, indent=4)

    print("File updated successfully.")
except FileNotFoundError:
    print(f"File not found: {adapter_config_path}")
except Exception as e:
    print(f"An error occurred: {e}")

print("checkpoint_path: ", checkpoint_path)

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    checkpoint_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)


# %%
sample = filtered_dataset[30]["text"]
sample

# %%
sample = """<s>[INST] <<SYS>>\nYour job is to identify weather or not the user input is related to the function specification delimited by <FUNCTIONS> and </FUNCTIONS>. If it is related then your response should be in the function_calling format: <FUNCTION_CALL_NAME>NAME_ASSOCIATED_WITH_THE_FUNCTION</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>ARGUMENTS_IN_STRINGIFIED_JSON_FORMAT</FUNCTION_CALL_ARGUMENTS>. Otherwise simply return a normal response. \n<FUNCTIONS>[{"name": "transcodeWebPage", "description": "Acquire precise webpage details or real-time search engine responses based on user-input content.", "parameters": {"type": "object", "properties": {"json": {"properties": {"link": {"type": "string", "description": "This parameter takes either a URL or a non-URL string. If a URL is given, the model will engage with the designated webpage to collect or interact with its data. If a non-URL string is given, the model will handle it as a search inquiry and try to find related real-time news or information. To guarantee the best results, make sure the input is a valid URL or a succinct search query."}}, "type": "object"}}}}]</FUNCTIONS>\n<</SYS>>\n\nJust checking [/INST] <FUNCTION_CALL_NAME>getSearchNews</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"params": {"country": "us", "language": "en", "topic": "Bath and Body Works"}}</FUNCTION_CALL_ARGUMENTS> </s>"""

# %%
from datasets import load_dataset
from random import randrange


# Load dataset from the hub and get a sample
sample = filtered_dataset[0]["text"]
sample = """<s>[INST] <<SYS>>\n<FUNCTION_CALL_NAME>\n<FUNCTIONS>[{"name": "transcodeWebPage", "description": "Acquire precise webpage details or real-time search engine responses based on user-input content.", "parameters": {"type": "object", "properties": {"json": {"properties": {"link": {"type": "string", "description": "This parameter takes either a URL or a non-URL string. If a URL is given, the model will engage with the designated webpage to collect or interact with its data. If a non-URL string is given, the model will handle it as a search inquiry and try to find related real-time news or information. To guarantee the best results, make sure the input is a valid URL or a succinct search query."}}, "type": "object"}}}}]</FUNCTIONS>\n<</SYS>>\n\n1 sentence about https://www.openplugin.io/ [/INST] <FUNCTION_CALL_NAME>transcodeWebPage</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"json": {"link": "https://www.openplugin.io/"}}</FUNCTION_CALL_ARGUMENTS> </s>"""

inp, target = sample.split("[/INST]")
prompt = inp + "[/INST]<FUNCTION_CALL_NAME>"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids,max_length=100, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Prompt:\n{prompt}\n\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy())[0][len(prompt):]}\n\n")
print(f"Ground truth:\n{target}\n\n")


# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
repo_name = "function_calling-llama_7b"
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

# %%
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Set the path to the checkpoint directory
hub_id = "SebastianS/function_calling-llama_7b"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    hub_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(hub_id)


# %%
# Load dataset from the hub and get a sample
sample = dataset[29]["text"]

inp, target = sample.split("[/INST]")
prompt = inp + "[/INST]"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Prompt:\n{prompt}\n\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy())[0][len(prompt):]}\n\n")
print(f"Ground truth:\n{target}\n\n")

# %%
expected_str = """<s>[INST] <<SYS>>
<FUNCTIONS>[{"name": "transcodeWebPage", "description": "Acquire precise webpage details or real-time search engine responses based on user-input content.", "parameters": {"type": "object", "properties": {"json": {"properties": {"link": {"type": "string", "description": "This parameter takes either a URL or a non-URL string. If a URL is given, the model will engage with the designated webpage to collect or interact with its data. If a non-URL string is given, the model will handle it as a search inquiry and try to find related real-time news or information. To guarantee the best results, make sure the input is a valid URL or a succinct search query."}}, "type": "object"}}}}]</FUNCTIONS>
<</SYS>>

1 sentence about https://www.openplugin.io/ [/INST] <FUNCTION_CALL_NAME>transcodeWebPage</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"json": {"link": "https://www.openplugin.io/"}}</FUNCTION_CALL_ARGUMENTS> </s>"""
expected_str_target = expected_str.split("[/INST]")[1]
generated_str = """<s>[INST] <<SYS>>
<FUNCTIONS>[{"name": "transcodeWebPage", "description": "Acquire precise webpage details or real-time search engine responses based on user-input content.", "parameters": {"type": "object", "properties": {"json": {"properties": {"link": {"type": "string", "description": "This parameter takes either a URL or a non-URL string. If a URL is given, the model will engage with the designated webpage to collect or interact with its data. If a non-URL string is given, the model will handle it as a search inquiry and try to find related real-time news or information. To guarantee the best results, make sure the input is a valid URL or a succinct search query."}}, "type": "object"}}}}]</FUNCTIONS>
<</SYS>>

1 sentence about https://www.openplugin.io/ [/INST] <FUNCTION_CALL_NAME>transcodeWebPage</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"json": {"link": "https://www.openplugin.io/"}}</FUNCTION_CALL_ARGUMENTS> </s>"""
generated_str_target = generated_str.split("[/INST]")[1]

# %%
import re
import json
def parse_prompt_back_to_data(prompt):
    """
    Function to parse a prompt back into the original data format, using a dictionary of function calling tokens.
    
    :param prompt: A string representing the constructed prompt.
    :param function_calling_tokens: A dictionary containing the start and end tokens for different function call elements.
    :return: A dictionary representing the original data instance.
    """
    # Building regular expression patterns using the function_calling_tokens
    functions_pattern = rf"{function_calling_tokens['FUNCTIONS']['start']}(.*?){function_calling_tokens['FUNCTIONS']['end']}"
    input_pattern = r"<</SYS>>\n\n(.*?) \[/INST\]"  # This remains unchanged as it's not part of function_calling_tokens
    target_content_pattern = r"\[/INST\] (.*)</s>"  # This also remains unchanged
    function_call_name_pattern = rf"{function_calling_tokens['FUNCTION_CALL_NAME']['start']}(.*?){function_calling_tokens['FUNCTION_CALL_NAME']['end']}"
    function_call_arguments_pattern = rf"{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['start']}(.*?){function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['end']}"

    # Extracting data using regular expressions
    functions_str = re.search(functions_pattern, prompt).group(1)
    input_content = re.search(input_pattern, prompt).group(1)
    target_content_match = re.search(target_content_pattern, prompt)

    # Parse functions JSON string
    functions = json.loads(functions_str)

    # Prepare the data dictionary
    data = {
        "input": [{
            "chatgptMessage": {"role": "user", "content": input_content},
            "functions": functions
        }],
        "target": {
            "chatgptMessage": {"role": "assistant"},
            "functions": functions  # Including functions in the target as well
        }
    }

    # Check if the target has a function call
    if function_calling_tokens['FUNCTION_CALL_NAME']['start'] in prompt:
        function_call_name = re.search(function_call_name_pattern, prompt).group(1)
        function_call_arguments = re.search(function_call_arguments_pattern, prompt).group(1)
        data["target"]["chatgptMessage"]["function_call"] = {
            "name": function_call_name,
            "arguments": function_call_arguments
        }
    else:
        # Handle case where regex might not find a match for target content
        if target_content_match:
            target_content = target_content_match.group(1)
            data["target"]["chatgptMessage"]["content"] = target_content

    return data
expected_data = parse_prompt_back_to_data(expected_str)
generated_data = parse_prompt_back_to_data(generated_str)

# %%
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import json

# Load a pre-trained model for sentence embedding (e.g., SBERT)
embedding_model_name = "sentence-transformers/bert-base-nli-mean-tokens"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def get_sentence_embedding(sentence):
    inputs = embedding_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)

    # Mean Pooling - Take attention mask into account for correct averaging
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
    sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled[0].numpy()

def sentence_similarity(sent1, sent2):
    embedding1 = get_sentence_embedding(sent1)
    embedding2 = get_sentence_embedding(sent2)
    return 1 - cosine(embedding1, embedding2)

def custom_metric(generated_json, expected_json):
    def compare_json(g_json, e_json, key_similarity_scores, value_similarity_scores):
        for e_key, e_value in e_json.items():
            # Check for exact key match or find the most similar key
            if e_key in g_json:
                g_key = e_key
                key_similarity_scores.append(1)
            else:
                # Compute similarity with all keys in generated_json and find the best match
                key_similarity = {gen_key: sentence_similarity(e_key, gen_key) for gen_key in g_json.keys()}
                g_key, key_sim_score = max(key_similarity.items(), key=lambda x: x[1])
                key_similarity_scores.append(key_sim_score)

            # Recursive comparison for nested objects, else compare values
            if isinstance(e_value, dict) and isinstance(g_json.get(g_key, {}), dict):
                compare_json(g_json[g_key], e_value, key_similarity_scores, value_similarity_scores)
            elif isinstance(e_value, str) and isinstance(g_json.get(g_key, ""), str):
                # Compare values only if they are strings at the root level
                value_sim_score = sentence_similarity(e_value, g_json[g_key])
                value_similarity_scores.append(value_sim_score)
            elif e_value == g_json.get(g_key, None):
                value_similarity_scores.append(1)  # Exact match for non-string root values
            else:
                value_similarity_scores.append(0)  # Non-matching root values

    key_similarity_scores = []
    value_similarity_scores = []
    compare_json(generated_json, expected_json, key_similarity_scores, value_similarity_scores)

    # Compute the average similarity scores
    avg_key_similarity = sum(key_similarity_scores) / len(key_similarity_scores) if key_similarity_scores else 0
    avg_value_similarity = sum(value_similarity_scores) / len(value_similarity_scores) if value_similarity_scores else 0

    return (avg_key_similarity + avg_value_similarity) / 2


# Example usage
generated_json = json.loads(generated_data["target"]["chatgptMessage"]["function_call"]["arguments"])
expected_json = json.loads(expected_data["target"]["chatgptMessage"]["function_call"]["arguments"])
score = custom_metric(generated_json, expected_json)
score


# %%


# %%
json.loads(expected_data["target"]["chatgptMessage"]["function_call"]["arguments"])

# %%
from huggingface_hub import HfApi, HfFolder, Repository
import os
api = HfApi()
api.create_repo("function_calling-llama_7b")


# %%
repo = Repository("function_calling-llama_7b", clone_from="SebastianS/function_calling-llama_7b")


# %%
if use_flash_attention:
    # unpatch flash attention
    from utils.llama_patch import unplace_flash_attn_with_attn
    unplace_flash_attn_with_attn()

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

args.output_dir = "llama-7-int4-fc"

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_id)



# %%
from datasets import load_dataset
from random import randrange


# Load dataset from the hub and get a sample
sample = dataset[0]["text"]

inp, target = sample.split("### Response:\n")
prompt = inp + "### Response:\n"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Prompt:\n{prompt}\n")
print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"Ground truth:\n{target}")



