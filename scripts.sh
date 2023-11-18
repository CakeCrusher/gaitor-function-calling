pip install "transformers==4.34.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.23.0" "bitsandbytes==0.41.1" "trl==0.4.7" "safetensors>=0.3.1" ipywidgets wandb --upgrade
python -c "import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'"
pip install ninja packaging
MAX_JOBS=1 pip install flash-attn --no-build-isolation