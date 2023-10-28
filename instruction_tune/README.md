# instruction_tune
Instruction-tune Llama 2 for function-calling.

### Step 1: Create JSON-Formatted Instruction-Output Pairs
Synthetically generate pairs (10000) of instructions and the desired output in JSON format.

**Example Pair 1:**
```json
{
  "instruction": "Generate a JSON response for a function call based on the user's question and the function call example.",
  "input": {
    "user_question": "What is the weather like in Boston?",
    "function_call_example": {
      "name": "get_current_weather",
      "arguments": {
        "location": "Boston, MA",
        "unit": "fahrenheit"
      }
    }
  },
  "output": {
    "function_call": {
      "name": "get_current_weather",
      "arguments": {
        "location": "Boston, MA",
        "unit": "fahrenheit"
      }
    }
  }
}
```

**Example Pair 2:**
```json
{
  "instruction": "Generate a JSON response for a function call based on the user's question and the function call example.",
  "input": {
    "user_question": "Convert 100 USD to EUR",
    "function_call_example": {
      "name": "currency_conversion",
      "arguments": {
        "amount": 100,
        "from": "USD",
        "to": "EUR"
      }
    }
  },
  "output": {
    "function_call": {
      "name": "currency_conversion",
      "arguments": {
        "amount": 100,
        "from": "USD",
        "to": "EUR"
      }
    }
  }
}
```

### Step 2: Preprocess Data
Ensure JSON-formatted instruction-output pairs are consistent and correctly formatted. 

### Step 3: Fine-Tune the Model
1. **Upload Data**: Save JSON-formatted instruction-output pairs in a file and upload it to Hi-Per-Gator.
2. **Configure Fine-Tuning**: Set the required parameters for fine-tuning.
3. **Run Fine-Tuning**: Start the fine-tuning process.

### Step 4: Test the Model
After the fine-tuning process, rigorously test the model to ensure it generates accurate and reliable JSON responses.

### Step 5: Iterate to Improve
If the performance is not satisfactory, adjust instruction-output pairs, reprocess data, or tweak the fine-tuning parameters, and then re-run the process.

### Next Step:
1. Determine how to encode input data for training
2. Choose valid hyperparameters

## References
- [Fine tune with PEFT](https://huggingface.co/blog/llama2#fine-tuning-with-peft)
- [Instruction tune llama 2](https://www.philschmid.de/instruction-tune-llama-2)
- [Train llama 2 with rlhf](https://www.philschmid.de/instruction-tune-llama-2)
- [Llamma 2 text embeddings](https://medium.com/@liusimao8/using-llama-2-models-for-text-embedding-with-langchain-79183350593d)
- [Instruction fine-tuning Llama 2 with PEFT’s QLoRa](https://medium.com/@ud.chandra/instruction-fine-tuning-llama-2-with-pefts-qlora-method-d6a801ebb19)
