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
        "chatgptMessage": {
          "role": "user",
          "content": "How are you doing"
        },
        "functions": null,
        "timeSent": "EALIEST_TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "assistant",
          "content": "Doing well, how about you?"
        },
        "functions": null,
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "user",
          "content": "What is the weather like in Boston?"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "assistant",
          "content": null,
          "function_call": {
            "name": "get_current_weather",
            "arguments": "{ \"location\": \"Boston, MA\"}"
          }
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT or TARGET to the model
      {
        "chatgptMessage": {
          "role": "function",
          "content": "{\"temperature\": \"22\", \"unit\": \"celsius\", \"description\": \"Sunny\"}",
          "name": "get_current_weather"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "assitant",
          "content": "The current weather is 22"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "user",
          "content": "Sorry, what is the size in Gainesville?"
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the INPUT to the model
      {
        "chatgptMessage": {
          "role": "assistant",
          "content": null,
          "function_call": {
            "name": "get_city_size",
            "arguments": "{ \"city\": \"Gainesville, FL\"}"
          }
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }, // this COULD be the TARGET to the model
      {
        "chatgptMessage": {
          "role": "function",
          "content": "{\"size\": \"100\", \"unit\": \"km^2\"}"
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "LATEST_TIME_HERE"
      }
    ]
  }
]
```

## Objective for training_data.json

```json
[
  {
    "input": [
      {
        "chatgptMessage": {
          "role": "user",
          "content": "What is the weather like in Boston?"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }
    ],
    "target": {
      "chatgptMessage": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{ \"location\": \"Boston, MA\"}"
        }
      },
      "functions": [{ "SOME_FUNCTION_HERE": {} }],
      "timeSent": "TIME_HERE"
    }
  },
  {
    "input": [
      {
        "chatgptMessage": {
          "role": "user",
          "content": "How are you doing"
        },
        "functions": null,
        "timeSent": "EALIEST_TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "assistant",
          "content": "Doing well, how about you?"
        },
        "functions": null,
        "timeSent": "TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "user",
          "content": "What is the weather like in Boston?"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }
    ],
    "target": {
      "chatgptMessage": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{ \"location\": \"Boston, MA\"}"
        }
      },
      "functions": [{ "SOME_FUNCTION_HERE": {} }],
      "timeSent": "TIME_HERE"
    }
  },
  {
    "input": [
      {
        "chatgptMessage": {
          "role": "user",
          "content": "What is the weather like in Boston?"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "assistant",
          "content": null,
          "function_call": {
            "name": "get_current_weather",
            "arguments": "{ \"location\": \"Boston, MA\"}"
          }
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "function",
          "content": "{\"temperature\": \"22\", \"unit\": \"celsius\", \"description\": \"Sunny\"}",
          "name": "get_current_weather"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }
    ],
    "target": {
      "chatgptMessage": {
        "role": "assitant",
        "content": "The current weather is 22"
      },
      "functions": [{ "SOME_FUNCTION_HERE": {} }],
      "timeSent": "TIME_HERE"
    }
  },
  {
    "input": [
      {
        "chatgptMessage": {
          "role": "user",
          "content": "Sorry, what is the size in Gainesville?"
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }
    ],
    "target": {
      "chatgptMessage": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_city_size",
          "arguments": "{ \"city\": \"Gainesville, FL\"}"
        }
      },
      "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
      "timeSent": "TIME_HERE"
    }
  },
  {
    "input": [
      {
        "chatgptMessage": {
          "role": "function",
          "content": "{\"temperature\": \"22\", \"unit\": \"celsius\", \"description\": \"Sunny\"}",
          "name": "get_current_weather"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "assitant",
          "content": "The current weather is 22"
        },
        "functions": [{ "SOME_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      },
      {
        "chatgptMessage": {
          "role": "user",
          "content": "Sorry, what is the size in Gainesville?"
        },
        "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
        "timeSent": "TIME_HERE"
      }
    ],
    "target": {
      "chatgptMessage": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_city_size",
          "arguments": "{ \"city\": \"Gainesville, FL\"}"
        }
      },
      "functions": [{ "SOME_DIFFERENT_FUNCTION_HERE": {} }],
      "timeSent": "TIME_HERE"
    }
  }
]
```

## Directory Structure
- [OpenPlugin](OpenPlugin): raw OpenPlugin datasets.
- [base](base): OpenPlugin datasets processed down to input and output.
- [train_test](train_test): Data ready for training.
- [prompts](prompts): Pairs of expected and generated prompts with the generated prompts evaluation scores.
- [data_utils.py](data_utils.py): Utilities for using, building, and storing train_test data.
- [prompting_utils](prompting_utils.py): Utilities for creating and interchangably transforming prompts to strictly formatted json.  

### Map Plugin Namespaces to Functions

Note: Plugins use the attribute `namespace` (abbreviated as `ns`) as their identifier.

- [openplugins.json](openplugins.json): All potential plugins used in dataset.
- [ns_to_function_map.ipynb](ns_to_function_map.ipynb): Notebook used to generate [ns_to_function_map.json](ns_to_function_map.json).
- [ns_to_function_map.json](ns_to_function_map.json): Mapping of namespaces to functions.
- [ns_to_function_map.log](ns_to_function_map.log): Log file tracking processed plugin namespaces.
