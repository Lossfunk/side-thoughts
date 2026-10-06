cd .. && \
python3 -m venv check && \
source check/bin/activate && \
cd side-thoughts && \
pip install --upgrade pip setuptools wheel && \
pip install -qqq --upgrade vllm==0.15.1 torchvision bitsandbytes xformers unsloth && \
pip install "torchao<0.11" --force-reinstall --no-deps && \
pip install -qqq triton && \
pip install transformers==4.56.2 && \
pip install --no-deps trl==0.22.2 && \
pip install -r requirements.txt && \
export VLLM_USE_MODELSCOPE=False && \
export VLLM_USE_ASCEND=False
