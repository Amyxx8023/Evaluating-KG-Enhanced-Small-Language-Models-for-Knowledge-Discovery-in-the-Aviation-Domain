#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 AviationQA.csv 转换为 SFT 格式的 JSONL 文件
"""

import pandas as pd
import json
import re
from tqdm import tqdm


def clean_question(question):
    """
    清理问题文本，移除 'predict answer:' 前缀
    """
    if question.startswith('predict answer:'):
        # 移除 'predict answer:' 前缀并去除首尾空格
        return question[len('predict answer:'):].strip()
    return question.strip()


def convert_to_sft_format(question, answer):
    """
    将问题和答案转换为 SFT 格式
    """
    return {
        "conversations": [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant", 
                "content": answer
            }
        ]
    }


def main():
    # 读取 CSV 文件
    print("正在读取 AviationQA.csv 文件...")
    try:
        df = pd.read_csv('./dataset/AviationQA.csv')
        print(f"成功读取 {len(df)} 条数据")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 检查列名
    if 'Question' not in df.columns or 'Answer' not in df.columns:
        print("错误: CSV 文件必须包含 'Question' 和 'Answer' 列")
        return
    
    # 转换数据
    print("正在转换数据格式...")
    converted_data = []
    
    # 处理每一行数据
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
        # 清理问题文本
        question = clean_question(str(row['Question']))
        answer = str(row['Answer'])
        
        # 跳过空的问题或答案
        if not question or not answer or question == 'nan' or answer == 'nan':
            continue
            
        # 转换为 SFT 格式
        sft_item = convert_to_sft_format(question, answer)
        converted_data.append(sft_item)
    
    # 保存为 JSONL 文件
    output_file = './dataset/sft_aviationqa.jsonl'
    print(f"正在保存到 {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(converted_data, desc="保存进度"):
                # 每行一个 JSON 对象
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"转换完成！共处理 {len(converted_data)} 条有效数据")
        print(f"输出文件: {output_file}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"- 原始数据条数: {len(df)}")
        print(f"- 有效转换条数: {len(converted_data)}")
        print(f"- 跳过的无效数据: {len(df) - len(converted_data)}")
        
        # 显示前3个转换示例
        print(f"\n前3个转换示例:")
        for i, item in enumerate(converted_data[:3]):
            print(f"\n示例 {i+1}:")
            print(f"用户: {item['conversations'][0]['content'][:100]}...")
            print(f"助手: {item['conversations'][1]['content'][:100]}...")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == "__main__":
    main()
