import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载 Baichuan 模型和 tokenizer
model_path = "/home/syq/fine_tuning/baichuan/LLaMA-Factory/Baichuan2-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# 输入和输出文件路径
input_file = "/home/syq/fine_tuning/baichuan/eval_test.json"
output_file = "/home/syq/fine_tuning/baichuan/output.json"

def extract_json_from_string(string):
    """
    从字符串中提取 JSON 对象。
    """
    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)
    match = json_pattern.search(string)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def compare_json_objects(json1, json2):
    """
    比较两个 JSON 对象是否相同。
    """
    return json1 == json2

def calculate_joint_acc(string1, string2):
    """
    提取字符串中的 JSON 并比较是否相同。
    """
    json1 = extract_json_from_string(string1)
    json2 = extract_json_from_string(string2)
    if json1 and json2:
        return compare_json_objects(json1, json2)
    return False

def read_conversations(input_file):
    """
    从 input.json 文件中提取每一轮对话的 human 和 gpt 内容。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = []
    if isinstance(data, list):
        for item in data:
            for i, conv in enumerate(item.get("conversations", [])):
                if conv.get("from") == "human":
                    # 找到 human 的对话内容
                    conversation_entry = {
                        "input": conv["value"],  # human 的输入
                        "gpt_res": None          # 预占位，用于存储对应 gpt 的响应
                    }
                    # 尝试获取同一轮的 gpt 响应
                    if i + 1 < len(item["conversations"]) and item["conversations"][i + 1].get("from") == "gpt":
                        conversation_entry["gpt_res"] = item["conversations"][i + 1]["value"]
                    conversations.append(conversation_entry)
    return conversations

def save_to_json(output, output_file):
    """
    将结果保存到 JSON 文件。
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def generate_response(model, tokenizer, input_text):
    """利用模型生成回答"""
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 读取对话内容，包括 human 的输入和 gpt 的响应
    conversations = read_conversations(input_file)

    # 用于存储最终输出结果
    result_output = {"responses": []}
    total_count = 0
    correct_count = 0

    for conv in conversations:
        human_input = conv["input"]
        gpt_res = conv["gpt_res"]

        print(f"Processing input: {human_input}")  # 调试信息
        
        # 调用 API 获取新的输出
        output = generate_response(model, tokenizer, human_input)
        
        if output:
            # 比较 output 和 gpt_res 是否一致
            is_correct = calculate_joint_acc(output, gpt_res)
            total_count += 1
            if is_correct:
                correct_count += 1
            
            # 将 input、output 和 gpt_res 保存到结果中
            result_output["responses"].append({
                "input": human_input,
                "output": output,
                "gpt_res": gpt_res,
                "is_correct": is_correct
            })
    
    # 计算整体正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    result_output["accuracy"] = accuracy

    # 保存结果到 JSON 文件
    save_to_json(result_output, output_file)
    print(f"Results saved to {output_file}")
    print(f"Total Accuracy: {accuracy:.2%}")