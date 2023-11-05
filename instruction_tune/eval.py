import json

def is_valid_parameter(param_schema, param_value):
    if 'type' in param_schema:
        if param_schema['type'] == 'string' and not isinstance(param_value, str):
            return False
        if param_schema['type'] == 'object' and not isinstance(param_value, dict):
            return False
        # Add additional type checks as necessary
    if 'enum' in param_schema and param_value not in param_schema['enum']:
        return False
    return True

def evaluate_function_call(input_json, response_json):
    try:
        # Load the input JSON to extract the expected function call parameters
        input_data = json.loads(input_json)
        expected_function = input_data['functions'][0]

        # Load the response JSON from the model
        response_data = json.loads(response_json)
        function_call_response = response_data['choices'][0]['message']['function_call']

        # Check if the function name matches
        if function_call_response['name'] != expected_function['name']:
            return False, f"Function name mismatch: expected {expected_function['name']} but got {function_call_response['name']}."

        # Get the required and optional parameters
        required_params = expected_function['parameters'].get('required', [])
        all_params = expected_function['parameters']['properties']

        # Check if all required arguments are present and correct
        for arg_name in required_params:
            if arg_name not in function_call_response['arguments']:
                return False, f"Missing required argument: {arg_name}."
            if not is_valid_parameter(all_params[arg_name], function_call_response['arguments'][arg_name]):
                return False, f"Invalid argument value or type for required: {arg_name}."

        # Check if optional arguments are valid if they are provided
        for arg_name, arg_value in function_call_response['arguments'].items():
            if arg_name in all_params and not is_valid_parameter(all_params[arg_name], arg_value):
                return False, f"Invalid argument value or type for optional: {arg_name}."

        # If all checks pass, the format is correct
        return True, "The function call format is valid."

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except KeyError as e:
        return False, f"Missing key in JSON structure: {str(e)}"

# You can use the same input_json and response_json from before to test this function.

# Example usage:
input_json = '''{
  "messages": [{"role": "user", "content": "What is the weather like in Boston?"}],
  "functions": [{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. San Francisco, CA"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  }]
}'''

response_json = '''{
  "id": "chatcmpl-123",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": { "location": "Boston, MA"}
      }
    },
    "finish_reason": "function_call"
  }]
}'''

# Call the evaluation function
is_valid, message = evaluate_function_call(input_json, response_json)
print(f"Is the response valid? {is_valid}\nMessage: {message}")
