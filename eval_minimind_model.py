#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind模型评估脚本
支持计算准确率、F1、Recall、BLEU、ROUGE等指标
集成wandb记录评估结果
"""

import argparse
import json
import random
import warnings
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
try:
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
except ImportError:
    print("警告: sklearn未安装，将使用简化的度量计算")
    def accuracy_score(y_true, y_pred):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if len(y_true) > 0 else 0.0

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    HAS_NLTK = True
except ImportError:
    print("警告: nltk未安装，BLEU分数将被跳过")
    HAS_NLTK = False

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    print("警告: rouge未安装，ROUGE分数将被跳过")
    HAS_ROUGE = False
import re
import time
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')


class MiniMindEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model, self.tokenizer = self.init_model()
        
        # 初始化ROUGE和BLEU组件
        self.rouge = Rouge() if HAS_ROUGE else None
        self.smoothing = SmoothingFunction().method1 if HAS_NLTK else None
        
        # 初始化wandb
        self.wandb = None
        if args.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=args.wandb_project,
                    name=f"MiniMind-Eval-{args.model_mode}-{args.hidden_size}",
                    config=vars(args)
                )
            except ImportError:
                print("警告: wandb未安装，跳过wandb日志记录")
        
        # 评估结果存储
        self.results = {
            'predictions': [],
            'references': [],
            'inputs': [],
            'metrics': {}
        }

    def init_model(self):
        """初始化模型和tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained('./model/')
        
        if self.args.load == 0:
            moe_path = '_moe' if self.args.use_moe else ''
            
            # 使用自定义模型路径或默认路径
            if self.args.model_path != 'auto':
                ckp = self.args.model_path
            else:
                modes = {0: 'pretrain', 1: 'full_sft_kg', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
                ckp = f'./{self.args.out_dir}/{modes[self.args.model_mode]}_{self.args.hidden_size}{moe_path}.pth'

            model = MiniMindForCausalLM(MiniMindConfig(
                hidden_size=self.args.hidden_size,
                num_hidden_layers=self.args.num_hidden_layers,
                use_moe=self.args.use_moe
            ))

            print(f"Loading model from: {ckp}")
            model.load_state_dict(torch.load(ckp, map_location=self.args.device), strict=True)

            if self.args.lora_name != 'None':
                apply_lora(model)
                load_lora(model, f'./{self.args.out_dir}/lora/{self.args.lora_name}_{self.args.hidden_size}.pth')
        else:
            transformers_model_path = './MiniMind2'
            tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
            model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)
        
        print(f'MiniMind model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
        return model.eval().to(self.args.device), tokenizer

    def auto_select_dataset(self):
        """根据模型模式自动选择数据集"""
        if self.args.test_data_path != 'auto':
            return self.args.test_data_path
        
        if self.args.model_mode == 0:
            # 预训练模式使用预训练数据
            return 'dataset/pretrain_hq.jsonl'
        elif self.args.model_mode == 1:
            # SFT模式使用增强的SFT数据
            return 'dataset/sft_aviationqa_kg.jsonl'
        elif self.args.model_mode == 2:
            # RLHF模式，建议使用增强SFT数据进行评估
            print("RLHF mode: Using enhanced SFT data for evaluation")
            return 'dataset/sft_aviationqa_kg.jsonl'
        elif self.args.model_mode == 3:
            # Reason模式，建议使用增强SFT数据进行评估
            print("Reason mode: Using enhanced SFT data for evaluation")
            return 'dataset/sft_aviationqa_kg.jsonl'
        elif self.args.model_mode == 4:
            # RLAIF模式，建议使用增强SFT数据进行评估
            print("RLAIF mode: Using enhanced SFT data for evaluation")
            return 'dataset/sft_aviationqa_kg.jsonl'
        else:
            # 未知模式需要用户指定
            raise ValueError(f"Unknown model mode {self.args.model_mode}, please manually specify --test_data_path parameter")

    def load_test_data(self):
        """加载测试数据"""
        # 自动选择数据集
        data_path = self.auto_select_dataset()
        self._actual_data_path = data_path  # 记录实际使用的数据路径
        print(f"Using dataset: {data_path}")
        
        test_data = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    
                    if 'conversations' in data:
                        # SFT格式数据
                        user_msg = data['conversations'][0]['content']
                        assistant_msg = data['conversations'][1]['content']
                        test_data.append({
                            'input': user_msg,
                            'reference': assistant_msg
                        })
                    elif 'text' in data:
                        # 预训练格式数据 - 用于续写任务
                        text = data['text']
                        # 将文本分成前半段作为输入，后半段作为参考
                        words = text.split()
                        if len(words) > 20:  # 确保有足够长度
                            split_point = len(words) // 2
                            input_text = ' '.join(words[:split_point])
                            reference_text = ' '.join(words[split_point:])
                            test_data.append({
                                'input': input_text,
                                'reference': reference_text
                            })
                        
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            for _, row in df.iterrows():
                question = str(row['Question'])
                if question.startswith('predict answer:'):
                    question = question[len('predict answer:'):].strip()
                test_data.append({
                    'input': question,
                    'reference': str(row['Answer'])
                })
        
        # 随机打乱数据（使用固定种子确保可重现）
        random.seed(self.args.seed)
        random.shuffle(test_data)
        
        # 限制评估数据量
        if self.args.max_eval_samples > 0:
            test_data = test_data[:self.args.max_eval_samples]
        
        print(f"Loaded {len(test_data)} test samples")
        return test_data

    def generate_response(self, input_text):
        """生成模型回复"""
        if self.args.model_mode == 0:
            # 预训练模式
            prompt = self.tokenizer.bos_token + input_text
        else:
            # SFT模式
            messages = [{"role": "user", "content": input_text}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_input_length
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.args.max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_p=self.args.top_p,
                temperature=self.args.temperature
            )

        response = self.tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response

    def calculate_exact_match(self, predictions, references):
        """计算精确匹配准确率"""
        exact_matches = []
        for pred, ref in zip(predictions, references):
            # 清理文本
            pred_clean = re.sub(r'[^\w\s]', '', pred.lower().strip())
            ref_clean = re.sub(r'[^\w\s]', '', ref.lower().strip())
            exact_matches.append(1 if pred_clean == ref_clean else 0)
        return np.mean(exact_matches)

    def calculate_token_accuracy(self, predictions, references):
        """计算基于token的准确率"""
        all_pred_tokens = []
        all_ref_tokens = []
        
        for pred, ref in zip(predictions, references):
            # 简单的token分割
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            # 对齐长度
            max_len = max(len(pred_tokens), len(ref_tokens))
            pred_tokens.extend([''] * (max_len - len(pred_tokens)))
            ref_tokens.extend([''] * (max_len - len(ref_tokens)))
            
            all_pred_tokens.extend(pred_tokens)
            all_ref_tokens.extend(ref_tokens)
        
        return accuracy_score(all_ref_tokens, all_pred_tokens)

    def calculate_bleu_score(self, predictions, references):
        """计算BLEU分数"""
        if not HAS_NLTK:
            return 0.0
            
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]  # BLEU expects list of references
            
            if len(pred_tokens) == 0:
                bleu_scores.append(0.0)
            else:
                score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(score)
        
        return np.mean(bleu_scores)

    def calculate_rouge_score(self, predictions, references):
        """计算ROUGE分数"""
        if not HAS_ROUGE:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            
        try:
            # 过滤空字符串
            valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                          if len(p.strip()) > 0 and len(r.strip()) > 0]
            
            if not valid_pairs:
                return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            
            pred_list, ref_list = zip(*valid_pairs)
            scores = self.rouge.get_scores(list(pred_list), list(ref_list), avg=True)
            return scores
        except Exception as e:
            print(f"ROUGE计算错误: {e}")
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

    def calculate_semantic_similarity(self, predictions, references):
        """计算语义相似度（简化版本，基于关键词重叠）"""
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0:
                similarities.append(0.0)
            else:
                intersection = len(pred_words.intersection(ref_words))
                union = len(pred_words.union(ref_words))
                jaccard = intersection / union if union > 0 else 0.0
                similarities.append(jaccard)
        
        return np.mean(similarities)

    def calculate_batch_metrics(self, predictions, references):
        """计算一批数据的指标"""
        if not predictions or not references:
            return {}
        
        metrics = {}
        
        # 精确匹配准确率
        metrics['exact_match_accuracy'] = self.calculate_exact_match(predictions, references)
        
        # Token级准确率
        metrics['token_accuracy'] = self.calculate_token_accuracy(predictions, references)
        
        # BLEU分数
        metrics['bleu_score'] = self.calculate_bleu_score(predictions, references)
        
        # ROUGE分数
        rouge_scores = self.calculate_rouge_score(predictions, references)
        metrics['rouge_1_f'] = rouge_scores['rouge-1']['f']
        metrics['rouge_2_f'] = rouge_scores['rouge-2']['f']
        metrics['rouge_l_f'] = rouge_scores['rouge-l']['f']
        
        # 语义相似度
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(predictions, references)
        
        # 回复长度统计
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        metrics['avg_prediction_length'] = np.mean(pred_lengths)
        metrics['avg_reference_length'] = np.mean(ref_lengths)
        metrics['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0
        
        return metrics

    def print_batch_metrics(self, metrics, batch_num, total_processed):
        """Print batch evaluation metrics"""
        print(f"\n{'='*60}")
        print(f"Batch {batch_num} Evaluation Results (Processed {total_processed} samples)")
        print(f"{'='*60}")
        print(f"Exact Match Accuracy:  {metrics['exact_match_accuracy']:.4f}")
        print(f"Token-level Accuracy:  {metrics['token_accuracy']:.4f}")
        print(f"BLEU Score:           {metrics['bleu_score']:.4f}")
        print(f"ROUGE-1 F1:          {metrics['rouge_1_f']:.4f}")
        print(f"ROUGE-2 F1:          {metrics['rouge_2_f']:.4f}")
        print(f"ROUGE-L F1:          {metrics['rouge_l_f']:.4f}")
        print(f"Semantic Similarity:  {metrics['semantic_similarity']:.4f}")
        print(f"Avg Prediction Length: {metrics['avg_prediction_length']:.2f} words")
        print(f"Avg Reference Length:  {metrics['avg_reference_length']:.2f} words")
        print(f"Length Ratio:         {metrics['length_ratio']:.4f}")
        print(f"{'='*60}")

    def evaluate_model(self):
        """Execute model evaluation"""
        print("Starting model evaluation...")
        test_data = self.load_test_data()
        
        predictions = []
        references = []
        inputs = []
        
        batch_num = 0
        batch_metrics_history = []
        
        # 生成预测并分批评估
        for i, item in enumerate(tqdm(test_data, desc="Generating predictions and evaluating")):
            input_text = item['input']
            reference = item['reference']
            
            # 生成预测
            prediction = self.generate_response(input_text)
            
            predictions.append(prediction)
            references.append(reference)
            inputs.append(input_text)
            
            # 保存详细结果
            self.results['predictions'].append(prediction)
            self.results['references'].append(reference)
            self.results['inputs'].append(input_text)
            
            # 打印部分示例
            if i < self.args.show_examples:
                print(f"\n=== Example {i+1} ===")
                print(f"Input: {input_text}")
                print(f"Reference: {reference}")
                print(f"Prediction: {prediction}")
                print("-" * 50)
            
            # 每处理完指定数量的数据就进行一次评估
            if (i + 1) % self.args.eval_batch_size == 0 or (i + 1) == len(test_data):
                batch_num += 1
                
                # 计算当前批次的指标
                print(f"\nCalculating evaluation metrics for batch {batch_num}...")
                batch_metrics = self.calculate_batch_metrics(predictions, references)
                batch_metrics_history.append({
                    'batch': batch_num,
                    'samples': i + 1,
                    'metrics': batch_metrics
                })
                
                # 打印批次结果
                self.print_batch_metrics(batch_metrics, batch_num, i + 1)
                
                # 记录到wandb（如果启用）
                if self.wandb:
                    wandb_metrics = {f"batch_{k}": v for k, v in batch_metrics.items()}
                    wandb_metrics['batch_num'] = batch_num
                    wandb_metrics['total_samples'] = i + 1
                    self.wandb.log(wandb_metrics)

        # 计算最终指标（使用全部数据）
        print("\nCalculating final evaluation metrics...")
        final_metrics = self.calculate_batch_metrics(predictions, references)
        
        # 添加批次历史到结果中
        self.results['batch_history'] = batch_metrics_history
        self.results['metrics'] = final_metrics
        
        return final_metrics

    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60)
        print(f"Exact Match Accuracy:  {metrics['exact_match_accuracy']:.4f}")
        print(f"Token-level Accuracy:  {metrics['token_accuracy']:.4f}")
        print(f"BLEU Score:           {metrics['bleu_score']:.4f}")
        print(f"ROUGE-1 F1:          {metrics['rouge_1_f']:.4f}")
        print(f"ROUGE-2 F1:          {metrics['rouge_2_f']:.4f}")
        print(f"ROUGE-L F1:          {metrics['rouge_l_f']:.4f}")
        print(f"Semantic Similarity:  {metrics['semantic_similarity']:.4f}")
        print(f"Avg Prediction Length: {metrics['avg_prediction_length']:.2f} words")
        print(f"Avg Reference Length:  {metrics['avg_reference_length']:.2f} words")
        print(f"Length Ratio:         {metrics['length_ratio']:.4f}")
        print("="*60)

    def save_results(self):
        """保存评估结果"""
        # 创建结果保存目录
        os.makedirs('eval_results', exist_ok=True)
        
        # 生成文件名（包含时间戳）
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据模型配置生成文件名前缀
        moe_suffix = "_moe" if self.args.use_moe else ""
        lora_suffix = f"_lora_{self.args.lora_name}" if self.args.lora_name != 'None' else ""
        
        file_prefix = f"eval_mode{self.args.model_mode}_h{self.args.hidden_size}{moe_suffix}{lora_suffix}_{timestamp}"
        
        # 保存详细结果JSON
        output_file = f"eval_results/{file_prefix}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 添加元数据
            full_results = {
                'metadata': {
                    'model_mode': self.args.model_mode,
                    'hidden_size': self.args.hidden_size,
                    'num_hidden_layers': self.args.num_hidden_layers,
                    'use_moe': self.args.use_moe,
                    'lora_name': self.args.lora_name,
                    'test_data_path': getattr(self, '_actual_data_path', self.args.test_data_path),
                    'max_eval_samples': self.args.max_eval_samples,
                    'eval_batch_size': self.args.eval_batch_size,
                    'temperature': self.args.temperature,
                    'top_p': self.args.top_p,
                    'max_new_tokens': self.args.max_new_tokens,
                    'seed': self.args.seed,
                    'evaluation_time': timestamp
                },
                'results': self.results
            }
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {output_file}")
        
        # 保存CSV格式的结果
        csv_file = f"eval_results/{file_prefix}.csv"
        df = pd.DataFrame({
            'input': self.results['inputs'],
            'reference': self.results['references'],
            'prediction': self.results['predictions']
        })
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"结果CSV已保存到: {csv_file}")
        
        # 保存指标摘要
        metrics_file = f"eval_results/{file_prefix}_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("MiniMind模型评估指标摘要\n")
            f.write("="*50 + "\n")
            f.write(f"评估时间: {timestamp}\n")
            f.write(f"模型模式: {self.args.model_mode}\n")
            f.write(f"隐藏层大小: {self.args.hidden_size}\n")
            f.write(f"隐藏层数量: {self.args.num_hidden_layers}\n")
            f.write(f"使用MoE: {self.args.use_moe}\n")
            f.write(f"LoRA名称: {self.args.lora_name}\n")
            f.write(f"测试数据: {getattr(self, '_actual_data_path', self.args.test_data_path)}\n")
            f.write(f"评估样本数: {len(self.results['inputs'])}\n")
            f.write(f"批次大小: {self.args.eval_batch_size}\n")
            f.write("="*50 + "\n\n")
            
            # 最终指标
            f.write("最终评估指标:\n")
            f.write("-" * 30 + "\n")
            metrics = self.results['metrics']
            f.write(f"精确匹配准确率:     {metrics['exact_match_accuracy']:.4f}\n")
            f.write(f"Token级准确率:      {metrics['token_accuracy']:.4f}\n")
            f.write(f"BLEU分数:          {metrics['bleu_score']:.4f}\n")
            f.write(f"ROUGE-1 F1:        {metrics['rouge_1_f']:.4f}\n")
            f.write(f"ROUGE-2 F1:        {metrics['rouge_2_f']:.4f}\n")
            f.write(f"ROUGE-L F1:        {metrics['rouge_l_f']:.4f}\n")
            f.write(f"语义相似度:         {metrics['semantic_similarity']:.4f}\n")
            f.write(f"平均预测长度:       {metrics['avg_prediction_length']:.2f} 词\n")
            f.write(f"平均参考长度:       {metrics['avg_reference_length']:.2f} 词\n")
            f.write(f"长度比率:          {metrics['length_ratio']:.4f}\n")
            
            # 批次历史（如果有）
            if 'batch_history' in self.results and self.results['batch_history']:
                f.write(f"\n批次评估历史:\n")
                f.write("-" * 30 + "\n")
                for batch_info in self.results['batch_history']:
                    batch_num = batch_info['batch']
                    samples = batch_info['samples']
                    batch_metrics = batch_info['metrics']
                    f.write(f"批次 {batch_num} (样本数: {samples}):\n")
                    f.write(f"  精确匹配: {batch_metrics['exact_match_accuracy']:.4f}, ")
                    f.write(f"BLEU: {batch_metrics['bleu_score']:.4f}, ")
                    f.write(f"ROUGE-1: {batch_metrics['rouge_1_f']:.4f}\n")
        print(f"指标摘要已保存到: {metrics_file}")

    def log_to_wandb(self, metrics):
        """记录到wandb"""
        if self.wandb:
            self.wandb.log(metrics)
            print("Results logged to wandb")

    def run_evaluation(self):
        """执行完整的评估流程"""
        start_time = time.time()
        
        # 执行评估
        metrics = self.evaluate_model()
        
        # 打印结果
        self.print_results(metrics)
        
        # 保存结果
        self.save_results()
        
        # 记录到wandb
        self.log_to_wandb(metrics)
        
        end_time = time.time()
        print(f"\nEvaluation completed, total time: {end_time - start_time:.2f} seconds")
        
        if self.wandb:
            self.wandb.finish()


def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    try:
        parser = argparse.ArgumentParser(description="MiniMind Model Evaluation")
    
        # 模型相关参数
        parser.add_argument('--lora_name', default='None', type=str)
        parser.add_argument('--out_dir', default='out', type=str)
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
        parser.add_argument('--hidden_size', default=512, type=int)
        parser.add_argument('--num_hidden_layers', default=8, type=int)
        parser.add_argument('--use_moe', default=False, type=bool)
        parser.add_argument('--load', default=0, type=int, help="0: Native torch weights, 1: transformers loading")
        parser.add_argument('--model_mode', default=1, type=int,
                            help="0: Pretrain model, 1: SFT-Chat model, 2: RLHF-Chat model, 3: Reason model, 4: RLAIF-Chat model")
        parser.add_argument('--model_path', default='auto', type=str,
                            help="Custom model path. 'auto' will use default path based on model_mode and hidden_size")

        # 生成参数
        parser.add_argument('--temperature', default=0.7, type=float)
        parser.add_argument('--top_p', default=0.85, type=float)
        parser.add_argument('--max_new_tokens', default=512, type=int)
        parser.add_argument('--max_input_length', default=1024, type=int)

        # 评估参数
        parser.add_argument('--test_data_path', default='auto', type=str,
                            help="Test data path, supports jsonl and csv formats. 'auto' means auto-select based on model mode: mode=0 uses pretrain_hq.jsonl, mode=1 uses sft_aviationqa_kg.jsonl")
        parser.add_argument('--max_eval_samples', default=1000, type=int,
                            help="Maximum evaluation samples, 0 means all")
        parser.add_argument('--eval_batch_size', default=50, type=int,
                            help="Evaluate every N processed samples to avoid waiting for all data generation")
        parser.add_argument('--show_examples', default=5, type=int,
                            help="Number of examples to display")

        # wandb参数
        parser.add_argument('--use_wandb', action='store_true', help="Use wandb for logging")
        parser.add_argument('--wandb_project', default='MiniMind-Evaluation', type=str)

        # 其他参数
        parser.add_argument('--seed', default=42, type=int)

        args = parser.parse_args()
        # 设置随机种子
        setup_seed(args.seed)
        # 创建评估器并运行
        evaluator = MiniMindEvaluator(args)
        evaluator.run_evaluation()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
