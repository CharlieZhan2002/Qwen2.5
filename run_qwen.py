import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
# 强制加载到 GPU，如果 GPU 不可用，则使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # 自动选择数据类型
    device_map="auto" if torch.cuda.is_available() else None,  # 自动分配设备
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model and tokenizer loaded successfully.")

# 初始上下文：定义链上数据分析助手
messages = [
    {"role": "system", "content": (
        "You are Qwen, an advanced assistant specialized in blockchain data analysis. "
        "You can help users understand blockchain transactions, smart contract interactions, token balances, and on-chain analytics. "
        "Provide clear, accurate, and concise explanations to assist the user with blockchain-related queries."
    )}
]

# 交互式问答
while True:
    user_question = input("You: ")  # 用户输入问题
    if user_question.lower() in ["exit", "quit"]:  # 输入 'exit' 或 'quit' 退出程序
        print("Exiting chat. Goodbye!")
        break

    # 添加用户问题到上下文
    messages.append({"role": "user", "content": user_question})

    # 构建输入文本
    print("Processing input...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回答
    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10240,  # 限制生成长度
        temperature=0.7,  # 增加生成的多样性
        top_p=0.9,  # 保留累积概率前 90% 的 token
        top_k=50,  # 只考虑前 50 个最高概率的候选项
        repetition_penalty=1.05  # 防止重复生成
    )

    # 解码生成的文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 提取并打印模型的回答
    assistant_response = response.split("assistant")[-1].strip()
    print(f"Qwen: {assistant_response}")

    # 将模型回答添加到上下文
    messages.append({"role": "assistant", "content": assistant_response})