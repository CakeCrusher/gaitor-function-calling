# Data
Data used for training and evaluation.

## Objective for conversations.json
[Helper Video](https://www.loom.com/share/99478f12313b4aa8bd00b9a15274da83?sid=db757ec6-f448-4dd4-a0aa-91c49df91f4e)
```json
[
  {
    "conversationId": "SOME_ID_HERE",
    "conversation": [
      { 
        "role": "user",
        "content": "How are you doing",
        "functions": null,
        "timeSent": "EALIEST_TIME_HERE",
      }, // This COULD serve as the INPUT for the model
      {
        "role": "assistant",
        "content": "Doing well, hou about you?",
        "functions": null,
        "timeSent": "TIME_HERE"
      }, // This COULD serve as the input for the model
      {
        "role": "user",
        "content": "What is the weather like in Boston?",
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // This COULD serve as the INPUT for the model
      {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{ \"location\": \"Boston, MA\"}",
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // This COULD serve as the INPUT or TARGET for the model
      {
        "role": "function",
        "name": "get_current_weather",
        "content": "{\"temperature\": \"22\", \"unit\": \"celsius\", \"description\": \"Sunny\"}",
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      },

      // Another question in the same conversation

      {
        "role": "user",
        "content": "Sorry, what is the size in Gainesville?",
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // This COULD serve as the INPUT for the model
      {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_city_size",
          "arguments": "{ \"city\": \"Gainesville, FL\"}",
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // This would serve as the TARGET for the model
      {
        "role": "function",
        "name": "get_city_size",
        "content": "{\"size\": \"100\", \"unit\": \"km^2\"}",
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "LATEST_TIME_HERE"
      }
    ]
  }
]

```

## Files
- [development_data.json](development_data.json): Raw collection of OpenPlugin data collected in development.
### Map Plugin Namespaces to Functions
Note: Plugins use the attribute `namespace` (abbreviated as `ns`) as their identifier.
- [openplugins.json](openplugins.json): All potential plugins used in dataset.
- [ns_to_function_map.ipynb](ns_to_function_map.ipynb): Notebook used to generate [ns_to_function_map.json](ns_to_function_map.json).
- [ns_to_function_map.json](ns_to_function_map.json): Mapping of namespaces to functions.
- [ns_to_function_map.log](ns_to_function_map.log): Log file tracking processed plugin namespaces.