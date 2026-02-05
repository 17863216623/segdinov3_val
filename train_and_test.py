#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练和测试脚本（Python版本）
支持更灵活的配置和错误处理
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """执行命令并处理错误"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        print(f"\n错误: {description}失败，退出码: {result.returncode}")
        sys.exit(result.returncode)
    
    return result

def find_ckpt_dir(save_root, dataset, dino_size, input_h, input_w, use_wavelet, use_ugbd, use_sdf):
    """自动查找checkpoint目录"""
    tag = "mamba"
    if use_wavelet:
        tag += "_wav"
    if use_ugbd:
        tag += "_ugbd"
    if use_sdf:
        tag += "_sdf"
    
    ckpt_dir = Path(save_root) / f"segdino_{dino_size}_{input_h}x{input_w}_{dataset}_{tag}" / "ckpts"
    
    if not ckpt_dir.exists():
        # 尝试查找任何包含ckpts的目录
        possible_dirs = list(Path(save_root).rglob("ckpts"))
        if possible_dirs:
            ckpt_dir = possible_dirs[0]
            print(f"警告: 使用找到的checkpoint目录: {ckpt_dir}")
        else:
            print(f"错误: 找不到checkpoint目录: {ckpt_dir}")
            sys.exit(1)
    
    return ckpt_dir

def main():
    parser = argparse.ArgumentParser(description="一键训练和测试脚本")
    
    # 基础配置
    parser.add_argument("--dino_ckpt", type=str, default="./web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="usod10k")
    parser.add_argument("--save_root", type=str, default="./runs_usod10k_wavelet")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_h", type=int, default=384)
    parser.add_argument("--input_w", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["cosine", "plateau", "constant"])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--best_metric", type=str, default="Fw_beta")
    
    # 模型配置
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--use_wavelet", action="store_true")
    parser.add_argument("--use_ugbd", action="store_true")
    parser.add_argument("--use_sdf", action="store_true")
    
    # 测试配置
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--test_only_best", action="store_true")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，直接测试")
    parser.add_argument("--skip_test", action="store_true", help="只训练，不测试")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="手动指定checkpoint目录")
    
    args = parser.parse_args()
    
    print("="*60)
    print("一键训练和测试脚本")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"调度器: {args.scheduler}")
    print("="*60)
    
    # 构建训练命令
    if not args.skip_train:
        train_cmd = [
            "python", "train_segdinov3.py",
            "--dino_ckpt", args.dino_ckpt,
            "--data_dir", args.data_dir,
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--input_h", str(args.input_h),
            "--input_w", str(args.input_w),
            "--lr", str(args.lr),
            "--scheduler", args.scheduler,
            "--patience", str(args.patience),
            "--early_stop", str(args.early_stop),
            "--best_metric", args.best_metric,
            "--dino_size", args.dino_size,
            "--save_root", args.save_root,
        ]
        
        if args.save_every_epoch:
            train_cmd.append("--save_every_epoch")
        if args.use_wavelet:
            train_cmd.append("--use_wavelet")
        if args.use_ugbd:
            train_cmd.append("--use_ugbd")
        if args.use_sdf:
            train_cmd.append("--use_sdf")
        
        train_cmd_str = " ".join(train_cmd)
        run_command(train_cmd_str, "训练")
    else:
        print("\n跳过训练阶段")
    
    # 确定checkpoint目录
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
    else:
        ckpt_dir = find_ckpt_dir(
            args.save_root, args.dataset, args.dino_size,
            args.input_h, args.input_w,
            args.use_wavelet, args.use_ugbd, args.use_sdf
        )
    
    print(f"\nCheckpoint目录: {ckpt_dir}")
    
    # 构建测试命令
    if not args.skip_test:
        test_cmd = [
            "python", "test_all_checkpoints.py",
            "--dino_ckpt", args.dino_ckpt,
            "--ckpt_dir", str(ckpt_dir),
            "--data_dir", args.data_dir,
            "--dataset", args.dataset,
            "--test_split", args.test_split,
            "--input_h", str(args.input_h),
            "--input_w", str(args.input_w),
            "--dino_size", args.dino_size,
        ]
        
        if args.use_ugbd:
            test_cmd.append("--use_ugbd")
        if args.use_sdf:
            test_cmd.append("--use_sdf")
        if args.test_only_best:
            test_cmd.append("--test_only_best")
        
        output_csv = ckpt_dir.parent / "test_results_all_ckpts.csv"
        if args.test_only_best:
            output_csv = ckpt_dir.parent / "test_results_best_only.csv"
        test_cmd.extend(["--output_csv", str(output_csv)])
        
        test_cmd_str = " ".join(test_cmd)
        run_command(test_cmd_str, "测试所有checkpoint")
        
        print("\n" + "="*60)
        print("全部完成！")
        print("="*60)
        print(f"训练checkpoint目录: {ckpt_dir}")
        print(f"测试结果CSV: {output_csv}")
        print(f"\n查看结果: cat {output_csv}")
    else:
        print("\n跳过测试阶段")

if __name__ == "__main__":
    main()
