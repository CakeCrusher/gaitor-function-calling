import torch
import json
import re
from gaitor_function_calling.evaluation.evaluation_utils import asd
from wandb.sdk.data_types.trace_tree import Trace

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
INSTRUCTION = f"Your job is to identify weather or not the user input is related to the function specification delimited by {function_calling_tokens['FUNCTIONS']['start']} and {function_calling_tokens['FUNCTIONS']['end']}. If it is related then your response should be in the function_calling format: {function_calling_tokens['FUNCTION_CALL_NAME']['start']}NAME_ASSOCIATED_WITH_THE_FUNCTION{function_calling_tokens['FUNCTION_CALL_NAME']['end']}{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['start']}ARGUMENTS_IN_STRINGIFIED_JSON_FORMAT{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['end']}. Otherwise simply return a normal response. "

uberAsd = "ASD" + asd

def parse_prompt_back_to_data(prompt, instruction = None):
    """
    Function to parse a prompt back into the original data format, using a dictionary of function calling tokens.
    
    :param prompt: A string representing the constructed prompt.
    :param function_calling_tokens: A dictionary containing the start and end tokens for different function call elements.
    :return: A dictionary representing the original data instance.
    """
    # Building regular expression patterns using the function_calling_tokens
    functions_pattern = rf"{function_calling_tokens['FUNCTIONS']['start']}(.*?){function_calling_tokens['FUNCTIONS']['end']}"
    input_pattern = r"<s>([\s\S]*?)\[/INST\]"  # This remains unchanged as it's not part of function_calling_tokens
    target_content_pattern = r"\[/INST\] (.*)</s>"  # This also remains unchanged
    function_call_name_pattern = rf"{function_calling_tokens['FUNCTION_CALL_NAME']['start']}(.*?){function_calling_tokens['FUNCTION_CALL_NAME']['end']}"
    function_call_arguments_pattern = rf"{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['start']}(.*?){function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['end']}"

    instruction_adjusted_prompt = prompt
    if instruction:
        instruction_adjusted_prompt = "".join(prompt.split(instruction))

    instruction_adjusted_prompt = "<s>" + instruction_adjusted_prompt.split("<s>")[-1]
        
    # Extracting data using regular expressions
    functions_str = re.search(functions_pattern, instruction_adjusted_prompt, re.DOTALL).group(1)
    
    input_content = re.search(input_pattern, instruction_adjusted_prompt).group(1) 
    if "<</SYS>>" in input_content:
        input_content = input_content.split("<</SYS>>")[-1]
        if input_content.startswith("\n\n"):
            input_content = input_content[2:]
    if function_calling_tokens["FUNCTIONS"]["end"] in input_content:
        input_content = input_content.split(function_calling_tokens["FUNCTIONS"]["end"])[-1].strip()

    target_content_match = re.search(target_content_pattern, instruction_adjusted_prompt)

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
    if function_calling_tokens['FUNCTION_CALL_NAME']['start'] in instruction_adjusted_prompt:
        function_call_name = re.search(function_call_name_pattern, instruction_adjusted_prompt, re.DOTALL).group(1)
        function_call_arguments = re.search(function_call_arguments_pattern, instruction_adjusted_prompt, re.DOTALL).group(1)
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

def json_arguments_from_prompt(data_text, model, tokenizer, instruction, wandb_config=False):
    data_split = data_text.split("[/INST]")
    inp = data_split[:-1]
    inp = "[/INST]".join(inp)
    target = data_split[-1]
    prompt = inp + "[/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, temperature=0.9)

    expected_str = data_text
    generated_str = tokenizer.batch_decode(outputs.detach().cpu().numpy())[0]

    if wandb_config:
        current_idx = wandb_config["idx"]
        current_epoch = wandb_config["epoch"]
        root_span = Trace(
            name=f"Epoch: {current_epoch} test {current_idx} sample",
            kind="llm",  # kind can be "llm", "chain", "agent" or "tool"
            status_code="success",
            status_message="successfully generated text",
            metadata={
                "epoch": current_epoch,
                "test_idx": current_idx,
            },
            inputs={"expected_str": expected_str},
            outputs={"generated_str": generated_str},
        )
        root_span.log(name="fc_samples")
        

    expected_data = parse_prompt_back_to_data(expected_str, instruction)
    generated_data = parse_prompt_back_to_data(generated_str, instruction)
    parse_prompt_back_to_data(generated_str, instruction)

    if "function_call" not in generated_data["target"]["chatgptMessage"]:
        raise Exception("No function call found in generated data")
    
    generated_arguments = json.loads(generated_data["target"]["chatgptMessage"]["function_call"]["arguments"])
    expected_arguments = json.loads(expected_data["target"]["chatgptMessage"]["function_call"]["arguments"])

    return generated_arguments, expected_arguments, {"expected_str": expected_str, "generated_str": generated_str}

def build_prompt(instance, instruction = None, shots = None):
    """
    Function to dynamically build a prompt based on the given instance, using a dictionary of function calling tokens.
    
    :param instance: A dictionary representing a single instance of the data.
    :param instruction: A string representing the instruction to be included at the start of the system message.
    :param shots: An integer representing the number of chat examples (as shots) to be included in the prompt.
    """
    input_message = instance['input'][-1]['chatgptMessage']['content']
    target_message = instance['target']['chatgptMessage']
    functions = instance['input'][-1]['functions']

    # Extracting function details as is
    functions_str = json.dumps(functions)

    # Building the prompt using the tokens from function_calling_tokens
    if instruction and shots != None:
        system_message = f"<<SYS>>\n{instruction}\n<</SYS>>\n\n"
        
        if shots >= 1:
            system_message += """<FUNCTIONS>[{"name": "fetchImage", "description": "Get images based on the search query", "parameters": {"type": "object", "properties": {"params": {"type": "object", "properties": {"query": {"type": "string", "description": "The search term to find images for"}}, "required": ["query"]}}}}]</FUNCTIONS>

show me 1 image of a dog [/INST] <FUNCTION_CALL_NAME>fetchImage</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"params": {"query": "dog"}}</FUNCTION_CALL_ARGUMENTS> </s><s>[INST]"""
        if shots >= 2:
            system_message += """<FUNCTIONS>[{"name": "transcodeWebPage", "description": "Acquire precise webpage details or real-time search engine responses based on user-input content.", "parameters": {"type": "object", "properties": {"json": {"properties": {"link": {"type": "string", "description": "This parameter takes either a URL or a non-URL string. If a URL is given, the model will engage with the designated webpage to collect or interact with its data. If a non-URL string is given, the model will handle it as a search inquiry and try to find related real-time news or information. To guarantee the best results, make sure the input is a valid URL or a succinct search query."}}, "type": "object"}}}}]</FUNCTIONS>

"In two sentences tell me what this site is about https://www.openplugin.io/" [/INST] <FUNCTION_CALL_NAME>transcodeWebPage</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"json": {"link": "https://www.openplugin.io/"}}</FUNCTION_CALL_ARGUMENTS> </s><s>[INST]"""
        if shots >= 3:
            system_message += """<FUNCTIONS>[{"name": "getNews", "description": "This endpoint allows users to request real-time news updates or news categorized by topics such as politics, sports, entertainment, technology, business, fashion, and health. Each news item includes a link and publication date. The news is updated in real time.", "parameters": {"type": "object", "properties": {"params": {"type": "object", "properties": {"country": {"type": "string", "description": "The country code of the desired news, using lowercase abbreviations. For example, 'de' for Germany."}, "language": {"type": "string", "description": "The language code for the desired news. If not provided, the default language of the specified country will be used."}, "category": {"type": "string", "enum": ["politics", "sports", "entertainment", "technology", "business", "fashion", "health"], "description": "If the user mentions related keyword in any languages, use the corresponding category ('politics', 'sports', 'entertainment', 'technology', 'fashion', 'health') as a category parameter. If the user does not mention any specific keyword, the default value for the category parameter is an empty string."}}, "required": ["country", "language"]}}}}, {"name": "getSearchNews", "description": "This endpoint allows users to provide any topic-related description content at will, and returns the latest news on related topics around the world in a timely manner.", "parameters": {"type": "object", "properties": {"params": {"type": "object", "properties": {"country": {"type": "string", "description": "The country code of the desired news, using lowercase abbreviations. For example, 'de' for Germany."}, "language": {"type": "string", "description": "The language code for the desired news. If not provided, the default language of the specified country will be used."}, "topic": {"type": "string", "description": "The main topic that users want to search for news, such as Trump, star, game, movie, etc."}}, "required": ["country", "language", "topic"]}}}}]</FUNCTIONS>

Amazon fba news [/INST] <FUNCTION_CALL_NAME>getSearchNews</FUNCTION_CALL_NAME><FUNCTION_CALL_ARGUMENTS>{"params": {"country": "us", "language": "en", "topic": "Amazon FBA"}}</FUNCTION_CALL_ARGUMENTS> </s><s>[INST]"""
        system_message += f"{function_calling_tokens['FUNCTIONS']['start']}{functions_str}{function_calling_tokens['FUNCTIONS']['end']}\n"
    
    elif instruction:
        system_message = f"<<SYS>>\n{instruction}\n{function_calling_tokens['FUNCTIONS']['start']}{functions_str}{function_calling_tokens['FUNCTIONS']['end']}\n<</SYS>>"
    else:
        system_message = f"<<SYS>>\n{function_calling_tokens['FUNCTIONS']['start']}{functions_str}{function_calling_tokens['FUNCTIONS']['end']}\n<</SYS>>"
        
    if 'function_call' in target_message:
        function_call_name = target_message['function_call']['name']
        function_call_arguments = target_message['function_call']['arguments']
        prompt = f"<s>[INST] {system_message}\n\n{input_message} [/INST] {function_calling_tokens['FUNCTION_CALL_NAME']['start']}{function_call_name}{function_calling_tokens['FUNCTION_CALL_NAME']['end']}{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['start']}{function_call_arguments}{function_calling_tokens['FUNCTION_CALL_ARGUMENTS']['end']} </s>"
    else:
        target_content = target_message.get('content', '')
        prompt = f"<s>[INST] {system_message}\n\n{input_message} [/INST] {target_content}</s>"

    return prompt

def n_shot_to_no_shot(prompt):
    json_data = parse_prompt_back_to_data(prompt, INSTRUCTION)
    no_shot_prompt = build_prompt(json_data, INSTRUCTION)
    return no_shot_prompt

