# gaitor-function-calling
Replicate OpenAI's function-calling model

## Todo
- [ ] shuffle data for all runs. [related to data/train_test](data/train_test).
- [ ] remove the BrowserOp openplugin test shot. [related to data/prompting_utils](data/prompting_utils.py).
- [ ] fix the custom evaluation metric to make it more fault tolerant to provide feedback eve nwhen not perfectly structured.
  - [ ] align with huggingface datasets [custom metric class](https://huggingface.co/docs/datasets/how_to_metrics#custom-metric-loading-script) to leverage huggingface libraries.
- [ ] incorporate the [rizerphe/glaive-function-calling-v2-llama](https://huggingface.co/datasets/rizerphe/glaive-function-calling-v2-llama?row=0) dataset into runs.