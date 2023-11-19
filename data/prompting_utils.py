import json
import re
from gaitor_function_calling.evaluation.evaluation_utils import asd

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

def parse_prompt_back_to_data(prompt, instruction):
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

    instruction_adjusted_prompt = prompt
    if instruction:
        instruction_adjusted_prompt = "".join(prompt.split(instruction))
        
    # Extracting data using regular expressions
    functions_str = re.search(functions_pattern, instruction_adjusted_prompt).group(1)
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
        function_call_name = re.search(function_call_name_pattern, instruction_adjusted_prompt).group(1)
        function_call_arguments = re.search(function_call_arguments_pattern, instruction_adjusted_prompt).group(1)
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

def build_prompt(instance, instruction = None):
    """
    Function to dynamically build a prompt based on the given instance, using a dictionary of function calling tokens.
    
    :param instance: A dictionary representing a single instance of the data.
    :param function_calling_tokens: A dictionary containing the start and end tokens for different function call elements.
    :return: A string representing the constructed prompt.
    """
    input_message = instance['input'][0]['chatgptMessage']['content']
    target_message = instance['target']['chatgptMessage']
    functions = instance['input'][0]['functions']

    # Extracting function details as is
    functions_str = json.dumps(functions)

    # Building the prompt using the tokens from function_calling_tokens
    if instruction:
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