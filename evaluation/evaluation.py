# %% [markdown]
# # Dataset

# %%
from gaitor_function_calling.data.prompting_utils import parse_prompt_back_to_data, function_calling_tokens, INSTRUCTION
from gaitor_function_calling.evaluation.evaluation_utils import FunctionCallingMetric
from datasets import load_dataset
import numpy
relative_path_to_data = './production_eval_chat-instruction.json'
dataset = load_dataset('json', data_files={'train': relative_path_to_data}, split="train")
print("Dataset size: ", len(dataset), end="\n\n")
print("Sample data: ", dataset[0]["text"], end="\n\n")
instruction = INSTRUCTION

# %% [markdown]
# # Load Models

# %%
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

# Set the path to the checkpoint directory
hub_id = "SebastianS/function_calling-llama_7b-nat-fc_only"

# load base LLM model and tokenizer
fc_model = AutoPeftModelForCausalLM.from_pretrained(
    hub_id,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
fc_tokenizer = AutoTokenizer.from_pretrained(hub_id)


# %% [markdown]
# # Metric

# %%
import re
import json
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

# Load a pre-trained model for sentence embedding (e.g., SBERT)
metric = FunctionCallingMetric()

# %% [markdown]
# # Iterator

# %%
metric_scores = []
for i, data in enumerate(dataset):
    if 0 < i < len(dataset):
        try:
            inp, target = data["text"].split("[/INST]")
            prompt = inp + "[/INST]"
            input_ids = fc_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = fc_model.generate(input_ids=input_ids, do_sample=True, top_p=0.9,temperature=0.9)
    
            expected_str = data["text"]
            generated_str = fc_tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]
            
    
            expected_data = parse_prompt_back_to_data(expected_str, instruction)
            generated_data = parse_prompt_back_to_data(generated_str, instruction)
            parse_prompt_back_to_data(generated_str, instruction)
    
            if "function_call" not in generated_data["target"]["chatgptMessage"]:
                metric_scores.append(0)
                print(f"{i} Metric score: {0}")
                continue
            generated_arguments = json.loads(generated_data["target"]["chatgptMessage"]["function_call"]["arguments"])
            expected_arguments = json.loads(expected_data["target"]["chatgptMessage"]["function_call"]["arguments"])
    
            metric_score = metric.run(generated_arguments, expected_arguments)
            metric_scores.append(metric_score)
    
            print(f"{i} Metric score: {metric_score:.2f}")
        except Exception as e:
            print("Error: ", e)

print(f"Average metric score: {sum(metric_scores) / len(metric_scores):.2f}")


# %%



