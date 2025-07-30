import json
import time
from openai import OpenAI
from tqdm import trange

# 初始化 DeepSeek 客户端
client = OpenAI(
    api_key="sk-661e3b6220124875bed6a43acca5e31e",
    base_url="https://api.deepseek.com/v1"
)

# 文件路径
json_file = "preference_need_dataset_en.json"

# 构造 prompt
GEN_PROMPT = """
I am building a dataset for classifying whether a robotic instruction requires user preference knowledge.

Please generate 32 short English instructions for a household robot. For each instruction, indicate whether it requires customized user preference (label=1) or not (label=0).

The output format should be a list of JSON entries like:
[
  {"instruction": "Bring me my favorite tea.", "label": 1},
  {"instruction": "Open the window.", "label": 0}
]

Only include the JSON list in your output. Do not include any explanations.
"""

# 封装追加函数
def append_to_json_file(filename, new_data):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.extend(new_data)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

# 主循环：生成 N 批数据
def generate_dataset(num_batches=200):
    for i in trange(num_batches,desc="Generating batches"):
        print(f"🔁 Generating batch {i+1}/{num_batches}...")

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": GEN_PROMPT}
                ],
                stream=False
            )
            output = response.choices[0].message.content
            samples = json.loads(output)

            # 检查是否32条，容错
            if isinstance(samples, list) and all("instruction" in s and "label" in s for s in samples):
                append_to_json_file(json_file, samples)
                print(f"✅ Batch {i+1} saved ({len(samples)} samples)")
            else:
                print(f"⚠️ Batch {i+1} returned invalid format, skipped.")
        except Exception as e:
            print(f"❌ Error in batch {i+1}: {e}")
            time.sleep(2)

        time.sleep(1.5)  # 避免过快请求

# 执行
if __name__ == "__main__":
    generate_dataset(200)
