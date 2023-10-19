# Pre Fine-tuning
Try and replicate function-calling through non finetuning techniques.

## References
- [ReAct](https://arxiv.org/abs/2210.03629): inspired the few-shot prompting approach.
  - May want to incorporate this concept: "when the majority answer among n CoT-SC samples occurs less than n/2
times (i.e. internal knowledge might not support the task confidently), back off to ReAct"
- [jsonformer](https://github.com/1rgs/jsonformer): generate structured json from language models