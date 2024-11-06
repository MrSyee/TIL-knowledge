# Quantization

## Reference
- Diffusers bitsandbytes Docs: https://huggingface.co/docs/diffusers/v0.31.0/quantization/bitsandbytes?bnb=8-bit
- LLM.int8(int8): https://huggingface.co/blog/hf-bitsandbytes-integration
- QLoRA(NF4): https://huggingface.co/blog/4bit-transformers-bitsandbytes

## Information
![image](https://github.com/user-attachments/assets/e47a1aae-830f-49a5-a11a-ea6ab1cf2b39)
![image](https://github.com/user-attachments/assets/a3264c58-55fa-470b-97c3-40aae483f177)

|Type|Name|bits|Algorithms|
|-|-|-|-|
|FP32|Full precision|32||
|FP16|Half precision|16||
|bfloat16|Half precision|16||
|INT8|int8|8|LLM.int8|
|FP8|fp8|8|E4M3, E5M2|
|FP4|fp4|4|QLoRA|
|NF4|fp4|4|QLoRA|
