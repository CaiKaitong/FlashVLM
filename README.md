# FlashVLM

FlashVLM is a text-guided visual token selection method for accelerating vision-language model inference. This README summarizes the configuration, environment setup, model preparation, dataset preparation, and evaluation commands needed to run FlashVLM.

## Overview

FlashVLM reduces the number of visual tokens processed by the language model while preserving task-relevant visual information. The main configuration entry is in:

```text
llava/model/llava_arch.py
```

Use this file to enable FlashVLM logic and adjust the number of retained visual tokens used during inference or evaluation.

## Environment Setup

1. Clone the FlashVLM project and enter the project directory.

```bash
git clone <your-flashvlm-repo-url> FlashVLM
cd FlashVLM
```

2. Create and activate a conda environment.

```bash
conda create -n flashvlm python=3.10 -y
conda activate flashvlm
```

3. Install the project dependencies.

```bash
pip install -e .
```

4. Optional: install FlashAttention for faster inference.

```bash
pip install flash-attn --no-build-isolation
```

## Model Preparation

Download the corresponding LLaVA checkpoints before evaluation.

| Version                | LLM        | Checkpoint                         |
| ---------------------- | ---------- | ---------------------------------- |
| LLaVA-1.5              | Vicuna-7B  | `liuhaotian/llava-v1.5-7b`         |
| LLaVA-1.5              | Vicuna-13B | `liuhaotian/llava-v1.5-13b`        |
| LLaVA-1.6 / LLaVA-NeXT | Vicuna-7B  | `liuhaotian/llava-v1.6-vicuna-7b`  |
| LLaVA-1.6 / LLaVA-NeXT | Vicuna-13B | `liuhaotian/llava-v1.6-vicuna-13b` |

If the evaluation scripts use a `CKPT` variable, set it to the checkpoint path or Hugging Face model name that matches the model you want to test.

Example:

```bash
CKPT=liuhaotian/llava-v1.5-7b
```

For 13B evaluation, replace the 7B checkpoint with the corresponding 13B checkpoint.

## Data Preparation

Prepare each benchmark dataset according to the project evaluation guide, usually `EVAL.md`.

Common evaluation datasets include:

- VQAv2
- TextVQA
- GQA
- ScienceQA
- POPE
- MME
- MMBench

Make sure dataset paths in the evaluation scripts match your local storage paths before running evaluation.

## FlashVLM Configuration

The core FlashVLM configuration is controlled by the visual token budget. The retained token number is usually passed as an argument to the evaluation script.

Example token budgets:

```text
64
128
256
```

Lower token budgets usually provide faster inference, while higher token budgets usually preserve more visual information. Choose the token number based on the speed-accuracy tradeoff required by your experiment.

Main implementation and configuration file:

```text
llava/model/llava_arch.py
```

Recommended configuration checklist:

- Confirm FlashVLM logic is enabled in `llava/model/llava_arch.py`.
- Confirm the retained visual token number is passed correctly by the script.
- Confirm the checkpoint path is correct.
- Confirm dataset paths are correct.
- Confirm GPU settings are suitable for the selected model size.

## Evaluation

Run the benchmark scripts with the retained visual token number as the script argument.

Evaluate VQAv2 with 128 retained visual tokens:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/eval/vqav2.sh 128
```

Evaluate TextVQA with 64 retained visual tokens:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh 64
```

For 13B models, update the checkpoint setting in the corresponding script.

Example:

```bash
CKPT=liuhaotian/llava-v1.5-13b
```

## Suggested Experiment Workflow

1. Install the environment.
2. Download the required LLaVA checkpoint.
3. Prepare benchmark datasets.
4. Check `llava/model/llava_arch.py` and confirm FlashVLM is enabled.
5. Set the retained visual token number.
6. Run the target evaluation script.
7. Compare accuracy and inference speed under different token budgets.

## Troubleshooting

If dependencies fail to install, first confirm the Python, PyTorch, CUDA, and compiler versions are compatible.

If FlashAttention installation fails, run the project without it first, then install a wheel matching your CUDA and PyTorch versions.

If evaluation scripts cannot find data, check the dataset paths defined in the scripts and in `EVAL.md`.

If GPU memory is insufficient, reduce the batch size, use fewer retained visual tokens, or evaluate with a smaller checkpoint.

## License

Follow the license included with the FlashVLM project.
