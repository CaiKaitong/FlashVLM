# FlashVLM

### 🔧 Code Structure for FlashVLM and VisPruner

FlashVLM is developed as an extension of the VisPruner framework.
 The key implementation for both **VisPruner** and **FlashVLM** can be found in:

> ```
> VisPruner/llava/model/llava_arch.py
> ```

Both methods are integrated within this file, and users can switch between them by enabling or commenting out the corresponding code blocks:

| Mode          | How to Enable                           | Environment Requirements              |
| ------------- | --------------------------------------- | ------------------------------------- |
| **FlashVLM**  | Comment out the VisPruner-specific code | Use the same environment as VisPruner |
| **VisPruner** | Comment out the FlashVLM-specific code  | Use the same environment settings     |

FlashVLM is fully compatible with the VisPruner environment setup.
 For detailed installation and dependency configuration, please refer to:

> ```
> VisPruner/readme.md
> ```

