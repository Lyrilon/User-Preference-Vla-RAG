# -*- coding: utf-8 -*-
"""
多模型、多策略微调实验框架

功能：
在一个二分类任务上，评估不同 Transformer 预训练模型（如 BERT, RoBERTa）
和不同参数高效微调方法（如 LoRA, Prefix Tuning, BitFit）的效果。

核心任务：
1.  从用户提供的 JSON 数据集加载数据。
2.  对多个预训练模型（Backbones）和多种微调策略（Fine-tuning Methods）进行组合。
3.  对每种组合进行训练、验证和测试。
4.  将训练日志、评估指标记录到本地运行的 WandB 服务。
5.  保存微调后的最佳模型。
6.  自动跳过已经完成的实验组合。

运行示例：
python this_script_name.py --data_path ./your_data.json --output_dir ./experiment_results --num_epochs 3

JSON 数据格式要求：
[
  {"instruction": "some text here", "label": 1},
  {"instruction": "another text here", "label": 0}
]
"""
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    set_seed,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    PeftModel,
    PeftConfig
)
import wandb

# --- 全局配置 ---
# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# 为保证实验可复现性，设置随机种子
set_seed(42)

# 定义支持的预训练模型（Backbones）
SUPPORTED_BACKBONES = [
    "bert-base-uncased",
    "roberta-base",
    "sentence-transformers/all-MiniLM-L6-v2",
]
# 定义支持的微调策略
SUPPORTED_FINETUNE_METHODS = [
    "full_finetune",  # 全量微调
    "freeze_base",    # 冻结基座，只训练分类头
    "lora",           # Low-Rank Adaptation
    "prefix_tuning",  # Prefix Tuning
    "bitfit",         # 只训练 bias 参数
]
# Weights & Biases 项目名称
WANDB_PROJECT_NAME = "vla-preference-classification-local"


# --- 任务 1: 数据加载与划分 ---
def load_and_split_data(data_path: str, test_size: float = 0.1, val_size: float = 0.1) -> DatasetDict:
    """
    从 JSON 文件加载数据，并划分为训练、验证、测试集。

    Args:
        data_path (str): JSON 数据文件路径。
        test_size (float): 测试集所占比例。
        val_size (float): 验证集所占比例（在剩余数据中）。

    Returns:
        DatasetDict: 包含 'train', 'validation', 'test' 的数据集字典。
    """
    logging.info(f"① 从 {data_path} 加载数据...")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        raise

    # 确保数据是列表形式
    if not isinstance(data, list):
        raise ValueError("JSON 数据必须是一个列表 (list of objects)。")

    texts = [item['instruction'] for item in data]
    labels = [item['label'] for item in data]
    
    # 第一次划分：分出训练+验证集 和 测试集
    # 使用 stratify 保证标签分布在划分后保持一致
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 第二次划分：从训练+验证集中分出训练集和验证集
    # 调整验证集比例
    relative_val_size = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=relative_val_size, random_state=42, stratify=train_val_labels
    )

    logging.info(f"数据划分完毕：")
    logging.info(f"  - 训练集: {len(train_texts)} 条")
    logging.info(f"  - 验证集: {len(val_texts)} 条")
    logging.info(f"  - 测试集: {len(test_texts)} 条")

    # 转换为 HuggingFace 的 Dataset 格式
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


# --- 任务 2 & 3: 模型加载与微调策略应用 ---
def get_model_and_tokenizer(backbone: str, finetune_method: str):
    """
    根据指定的 backbone 和微调策略加载模型和 tokenizer。

    Args:
        backbone (str): 预训练模型的名称或本地路径。
        finetune_method (str): 微调策略的名称。

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: 加载好的模型和分词器。
    """
    logging.info(f"② 加载模型: {backbone}，使用微调策略: {finetune_method}")
    
    # 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone)
    except Exception as e:
        logging.error(f"加载 Tokenizer '{backbone}' 失败: {e}")
        raise

    # 加载基础模型，指定为二分类任务
    try:
        model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2)
    except Exception as e:
        logging.error(f"加载模型 '{backbone}' 失败: {e}")
        raise

    # 根据微调策略调整模型参数的可训练性
    if finetune_method == "full_finetune":
        # 所有参数都可训练，无需额外操作
        pass
    elif finetune_method == "freeze_base":
        # 冻结基座模型的所有参数
        for param in model.base_model.parameters():
            param.requires_grad = False
    elif finetune_method == "bitfit":
        # 只训练 bias 参数
        for name, param in model.named_parameters():
            if '.bias' not in name:
                param.requires_grad = False
    elif finetune_method == "lora":
        # 应用 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
    elif finetune_method == "prefix_tuning":
        # 应用 Prefix Tuning
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=20
        )
        model = get_peft_model(model, peft_config)
    else:
        raise ValueError(f"不支持的微调方法: {finetune_method}")

    # 打印可训练参数的数量
    if isinstance(model, PeftModel):
        model.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(
            f"可训练参数: {trainable_params} | 总参数: {total_params} | 可训练比例: {100 * trainable_params / total_params:.2f}%"
        )
        
    return model, tokenizer


# --- 任务 4: 数据预处理 ---
def create_preprocess_function(tokenizer):
    """
    创建用于数据集的预处理函数。

    Args:
        tokenizer: HuggingFace Tokenizer 实例。

    Returns:
        Callable: 一个函数，用于对数据集进行编码。
    """
    def preprocess(examples):
        # 对文本进行编码，进行填充和截断
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    return preprocess


# --- 任务 5 & 6 & 7: 训练、评估与保存 ---
def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    计算评估指标（这里是准确率）。

    Args:
        p (EvalPrediction): Trainer 的评估预测结果。

    Returns:
        Dict[str, float]: 包含准确率的字典。
    """
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

def run_training_pipeline(
    backbone: str,
    finetune_method: str,
    datasets: DatasetDict,
    output_dir_base: str,
    num_epochs: int
):
    """
    执行完整的训练、评估和保存流程。
    """
    # 清理 backbone 名称，使其适用于文件路径
    backbone_sanitized = backbone.replace("/", "_")
    run_name = f"{backbone_sanitized}_{finetune_method}"
    output_dir = Path(output_dir_base) / backbone_sanitized / finetune_method
    
    logging.info("="*80)
    logging.info(f"🚀 开始新一轮实验: {run_name}")
    logging.info(f"   - 模型输出路径: {output_dir}")
    logging.info("="*80)

    # 初始化 WandB
    try:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            name=run_name,
            reinit=True,
            config={
                "backbone": backbone,
                "finetune_method": finetune_method,
                "num_epochs": num_epochs,
            }
        )
    except Exception as e:
        logging.warning(f"WandB 初始化失败，将禁用日志记录: {e}")
        os.environ["WANDB_DISABLED"] = "true"


    # 加载模型和 Tokenizer
    model, tokenizer = get_model_and_tokenizer(backbone, finetune_method)

    # 数据预处理
    logging.info("③ 数据预处理...")
    preprocess_function = create_preprocess_function(tokenizer)
    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # 设置训练参数
    logging.info("④ 设置训练参数...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",  # 每个 epoch 结束后进行评估
        save_strategy="epoch",        # 每个 epoch 结束后保存模型
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="accuracy", # 使用准确率作为评估指标
        greater_is_better=True,
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "true" else "none",
        fp16=torch.cuda.is_available(), # 如果有 GPU，使用半精度训练
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 开始训练
    logging.info("⑤ 开始训练...")
    trainer.train()
    logging.info("训练完成。")

    # 在测试集上进行最终评估
    logging.info("⑥ 在测试集上进行最终评估...")
    test_results = trainer.predict(tokenized_datasets["test"])
    logging.info(f"测试集准确率: {test_results.metrics['test_accuracy']:.4f}")
    
    # 将测试结果上报到 WandB
    if os.environ.get("WANDB_DISABLED") != "true":
        wandb.log({"test_accuracy": test_results.metrics['test_accuracy']})

    # 保存最终模型和 Tokenizer
    logging.info(f"⑦ 保存最终模型到 {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # 如果是 PEFT 模型，也保存 PEFT 配置
    if isinstance(model, PeftModel):
        model.peft_config[model.active_adapter].save_pretrained(str(output_dir))

    logging.info(f"实验 {run_name} 完成。")
    if os.environ.get("WANDB_DISABLED") != "true":
        wandb.finish()


def main():
    """
    主函数，解析命令行参数并启动实验流程。
    """
    parser = argparse.ArgumentParser(description="多模型、多策略微调实验框架")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="包含 'instruction' 和 'label' 的 JSON 数据文件路径。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetune_results",
        help="保存模型和日志的根目录。"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="训练的轮次（Epochs）。"
    )
    parser.add_argument(
        "--backbones",
        nargs='+',
        default=SUPPORTED_BACKBONES,
        help=f"要测试的预训练模型列表。支持: {', '.join(SUPPORTED_BACKBONES)}"
    )
    parser.add_argument(
        "--finetune_methods",
        nargs='+',
        default=SUPPORTED_FINETUNE_METHODS,
        help=f"要测试的微调策略列表。支持: {', '.join(SUPPORTED_FINETUNE_METHODS)}"
    )
    args = parser.parse_args()

    # 1. 加载和划分数据（一次性完成）
    datasets = load_and_split_data(args.data_path)

    # 2. 遍历所有模型和微调策略的组合
    for backbone in args.backbones:
        if backbone not in SUPPORTED_BACKBONES:
            logging.warning(f"跳过不支持的模型: {backbone}")
            continue
        for method in args.finetune_methods:
            if method not in SUPPORTED_FINETUNE_METHODS:
                logging.warning(f"跳过不支持的微调方法: {method}")
                continue
            
            # --- 新增的检查逻辑 ---
            # 提前构建输出目录路径以进行检查
            backbone_sanitized = backbone.replace("/", "_")
            output_dir = Path(args.output_dir) / backbone_sanitized / method
            
            # 检查输出目录是否已存在且不为空，如果是，则跳过
            if output_dir.exists() and any(output_dir.iterdir()):
                logging.info(f"✅ 结果目录 {output_dir} 已存在，跳过实验: {backbone_sanitized}_{method}")
                continue
            # --- 结束新增逻辑 ---

            # 为每个组合运行训练流程
            try:
                run_training_pipeline(
                    backbone=backbone,
                    finetune_method=method,
                    datasets=datasets,
                    output_dir_base=args.output_dir,
                    num_epochs=args.num_epochs
                )
            except Exception as e:
                logging.error(f"实验失败: backbone={backbone}, method={method}")
                logging.error(f"错误详情: {e}", exc_info=True)
                # 即使一个实验失败，也继续下一个
                if os.environ.get("WANDB_DISABLED") != "true" and wandb.run is not None:
                    wandb.finish(exit_code=1) # 标记此 wandb run 失败
                continue
    
    logging.info("所有实验已完成！")


if __name__ == "__main__":
    main()
