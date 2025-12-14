我们使用 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)作为评测框架对模型进行评估。

1. 环境配置

请首先按照 lmms-eval 官方仓库中的环境配置说明完成环境搭建，以确保评测流程的正确性和兼容性。

2. FlashAttention 安装

在完成 lmms-eval 环境配置后，请安装与当前 PyTorch 和 Python 版本相匹配的 FlashAttention。

请确保：

FlashAttention 的版本与已安装的 PyTorch 版本兼容；

CUDA 版本与本地 GPU 架构受支持。

FlashAttention 的安装方式和详细说明请参考其官方仓库：

3. 模型代码修改

为使用 FlashVLM 进行评估，仅需对 lmms-eval 中的一处模型导入代码进行修改。

在以下文件中：

```python
model/simple/qwen2_5vl.py
```

将原始导入语句：

```python
from transformers import Qwen2_5_VLForConditionalGeneration
```

替换为：

```python
from flashvlm_qwen_2_5_VL import Qwen2_5_VLForConditionalGeneration
```

除上述修改外，无需进行任何其他代码调整。完成后即可按照 lmms-eval 的标准流程直接运行评测。
