import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto" if torch.cuda.is_available() else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model and tokenizer loaded successfully.")

messages = [
    {"role": "system", "content": (
        "You are Qwen, an advanced assistant specialized in blockchain data analysis. "
        "You can help users understand blockchain transactions, smart contract interactions, token balances, and on-chain analytics. "
        "Provide clear, accurate, and concise explanations to assist the user with blockchain-related queries."
    )}
]

while True:
    user_question = input("You: ")
    if user_question.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break

    messages.append({"role": "user", "content": user_question})
    
    print("Processing input...")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 记录显存占用（生成前）
    if device == "cuda":
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转换为 MB

    print("Generating response...")

    # 记录首个 token 生成时间
    start_time = time.time()
    first_token_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    first_token_done = False
    prev_time = start_time  # 用于计算 token 之间的延迟

    # 生成回答
    generated_ids = []
    for token in model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
        return_dict_in_generate=True,
        output_scores=True,
    ).sequences[:, model_inputs.input_ids.shape[1]:]:
        generated_ids.append(token.item())

        # 记录首个 token 生成时间
        if not first_token_done:
            if device == "cuda":
                first_token_event.record()
                torch.cuda.synchronize()
                first_token_time = first_token_event.elapsed_time(first_token_event)
            else:
                first_token_time = (time.time() - start_time) * 1000  # 转换为毫秒
            first_token_done = True

        # 计算 token 之间的间隔
        now_time = time.time()
        token_latency = (now_time - prev_time) * 1000  # 毫秒
        prev_time = now_time

        print(f"Generated token: {token.item()} | Token latency: {token_latency:.2f} ms")

    # 记录显存占用（生成后）
    if device == "cuda":
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转换为 MB

    # 解码生成的文本
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nQwen: {response}\n")
    print(f"First token latency: {first_token_time:.2f} ms")
    print(f"GPU Memory before generation: {memory_before:.2f} MB")
    print(f"GPU Memory after generation: {memory_after:.2f} MB")
    print(f"Total GPU memory used: {memory_after - memory_before:.2f} MB\n")

    messages.append({"role": "assistant", "content": response})
