# test_all_checkpoints.py
# -*- coding: utf-8 -*-
"""
测试脚本：加载训练过程中保存的所有checkpoint，在测试集上评估并生成报告
"""
import os
import csv
import glob
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 导入训练脚本中的模块
from train_segdinov3 import SmartFolderDataset, evaluate_with_module

def load_checkpoint(backbone, decoder, refine, sdf_head, ckpt_path, device):
    """加载checkpoint"""
    state = torch.load(ckpt_path, map_location=device)
    
    # 加载各个组件的权重
    if "backbone" in state:
        backbone.load_state_dict(state["backbone"], strict=False)
    if "decoder" in state:
        decoder.load_state_dict(state["decoder"], strict=False)
    if refine is not None and "refine" in state:
        refine.load_state_dict(state["refine"], strict=False)
    if sdf_head is not None and "sdf_head" in state:
        sdf_head.load_state_dict(state["sdf_head"], strict=False)
    
    epoch = state.get("epoch", 0)
    return epoch

def find_all_checkpoints(ckpt_dir):
    """查找所有checkpoint文件"""
    checkpoints = []
    
    # 查找所有epoch checkpoint
    epoch_ckpts = glob.glob(os.path.join(ckpt_dir, "epoch_*.pth"))
    epoch_ckpts.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    # 查找best checkpoint
    best_ckpts = glob.glob(os.path.join(ckpt_dir, "best_*.pth"))
    
    # 查找latest checkpoint
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    if os.path.exists(latest_ckpt):
        checkpoints.append(("latest", latest_ckpt))
    
    # 添加best checkpoints
    for ckpt in best_ckpts:
        name = os.path.basename(ckpt)
        checkpoints.append((f"best_{name}", ckpt))
    
    # 添加epoch checkpoints
    for ckpt in epoch_ckpts:
        epoch = int(os.path.basename(ckpt).split("_")[1].split(".")[0])
        checkpoints.append((f"epoch_{epoch:03d}", ckpt))
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description="测试所有保存的checkpoint")
    
    # 基础参数
    parser.add_argument("--data_dir", type=str, default="./segdata")
    parser.add_argument("--dataset", type=str, default="usod10k")
    parser.add_argument("--img_dir_name", type=str, default="image")
    parser.add_argument("--label_dir_name", type=str, default="mask")
    parser.add_argument("--img_ext", type=str, default=None)
    parser.add_argument("--mask_ext", type=str, default=".png")
    parser.add_argument("--input_h", type=int, default=384)
    parser.add_argument("--input_w", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # DINOv3配置
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--dino_size", type=str, default="s", choices=["b","s"])
    
    # 解码器配置
    parser.add_argument("--mamba_num_queries", type=int, default=12)
    parser.add_argument("--mamba_dim", type=int, default=192)
    parser.add_argument("--mamba_layers", type=int, default=3)
    parser.add_argument("--mamba_groups", type=int, default=4)
    parser.add_argument("--mamba_kernel", type=int, default=9)
    parser.add_argument("--mamba_mask_embed_dim", type=int, default=64)
    
    # UGBD和SDF配置
    parser.add_argument("--use_ugbd", action="store_true")
    parser.add_argument("--ugbd_steps", type=int, default=3)
    parser.add_argument("--ugbd_mid_ch", type=int, default=64)
    parser.add_argument("--use_sdf", action="store_true")
    parser.add_argument("--sdf_mid_ch", type=int, default=64)
    parser.add_argument("--sdf_lambda", type=float, default=0.5)
    
    # 评估配置
    parser.add_argument("--ckpt_dir", type=str, required=True, help="checkpoint目录路径")
    parser.add_argument("--test_split", type=str, default="test", help="测试集split名称")
    parser.add_argument("--test_datasets", type=str, nargs="+", default=None, 
                        help="要测试的多个数据集名称列表，例如: --test_datasets usod_test cod10k_test 第三个测试集名称。如果不指定，则使用--dataset参数")
    parser.add_argument("--dual_sam_eval", action="store_true")
    parser.add_argument("--eval_tta", action="store_true")
    parser.add_argument("--tta_scales", type=str, default="1.0,0.75,1.25")
    parser.add_argument("--tta_no_flip", action="store_true")
    parser.add_argument("--test_vis_dir", type=str, default=None, help="测试可视化目录")
    
    # 输出配置
    parser.add_argument("--output_csv", type=str, default=None, help="输出CSV文件路径")
    parser.add_argument("--test_only_best", action="store_true", help="只测试best checkpoint")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    
    # 构建模型
    if args.dino_size=="b":
        vit = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        vit = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
    
    from dpt import DPT
    backbone = DPT(nclass=1, backbone=vit).to(device)
    
    from mamba_decoder import GMSAMDecoderLite
    decoder = GMSAMDecoderLite(
        num_queries=args.mamba_num_queries, token_dim=args.mamba_dim,
        num_layers=args.mamba_layers, groups=args.mamba_groups, kernel_size=args.mamba_kernel,
        mask_embed_dim=args.mamba_mask_embed_dim
    ).to(device)
    
    refine = None
    if args.use_ugbd:
        from ugbd_refiner import UGBDRefiner
        refine = UGBDRefiner(mid_ch=args.ugbd_mid_ch, steps=args.ugbd_steps).to(device)
    
    sdf_head = None
    if args.use_sdf:
        from sdf_head import SDFHead
        sdf_head = SDFHead(mid_ch=args.sdf_mid_ch, fuse_lambda=args.sdf_lambda).to(device)
    
    # 加载测试集（支持多个测试集）
    H, W = args.input_h, args.input_w
    
    # 确定要测试的数据集列表
    if args.test_datasets:
        test_dataset_names = args.test_datasets
    else:
        # 如果没有指定，使用原来的逻辑
        test_dataset_names = [args.dataset.rstrip("/") + "_test"]
    
    # 为每个测试集创建数据加载器
    test_loaders = {}
    for test_ds_name in test_dataset_names:
        # 测试集：优先使用独立的test目录（如usod_test），如果没有则尝试dataset下的test子目录
        test_root = os.path.join(args.data_dir, test_ds_name)
        if not os.path.isdir(test_root):
            # 尝试去掉_test后缀，使用原始数据集名称下的test子目录
            base_name = test_ds_name.replace("_test", "")
            test_root = os.path.join(args.data_dir, base_name)
        
        try:
            test_set = SmartFolderDataset(
                test_root, args.test_split, args.img_dir_name, args.label_dir_name, 
                args.img_ext, args.mask_ext, (H, W), use_aug=False
            )
            test_loader = DataLoader(
                test_set, batch_size=args.batch_size, shuffle=False, 
                num_workers=max(1, args.num_workers//2), drop_last=False, pin_memory=True
            )
            test_loaders[test_ds_name] = test_loader
            print(f"[Test Dataset] {test_ds_name}: {len(test_set)} samples (from '{test_root}', split='{args.test_split}')")
        except Exception as e:
            print(f"[Warning] Failed to load test dataset '{test_ds_name}': {e}")
            print(f"  Tried path: {test_root}")
    
    if not test_loaders:
        print(f"[Error] No valid test datasets loaded!")
        return
    
    print(f"[Loaded] {len(test_loaders)} test dataset(s): {list(test_loaders.keys())}")
    
    # 查找所有checkpoint
    checkpoints = find_all_checkpoints(args.ckpt_dir)
    if args.test_only_best:
        checkpoints = [ckpt for ckpt in checkpoints if "best" in ckpt[0]]
    
    if not checkpoints:
        print(f"[Error] No checkpoints found in {args.ckpt_dir}")
        return
    
    print(f"[Found] {len(checkpoints)} checkpoints to test")
    
    # 评估参数（需要创建一个类似训练时的args对象）
    class EvalArgs:
        def __init__(self):
            self.dual_sam_eval = args.dual_sam_eval
            self.eval_tta = args.eval_tta
            self.tta_scales = args.tta_scales
            self.tta_no_flip = args.tta_no_flip
            self.val_vis_dir = args.test_vis_dir
    
    eval_args = EvalArgs()
    
    # 评估所有checkpoint（在多个测试集上）
    results = []
    for ckpt_name, ckpt_path in tqdm(checkpoints, desc="Testing checkpoints"):
        print(f"\n{'='*80}")
        print(f"[Testing] {ckpt_name}: {ckpt_path}")
        print(f"{'='*80}")
        
        # 加载checkpoint
        epoch = load_checkpoint(backbone, decoder, refine, sdf_head, ckpt_path, device)
        
        # 在多个测试集上评估
        all_metrics = {}
        for test_ds_name, test_loader in test_loaders.items():
            print(f"\n[Evaluating on] {test_ds_name}")
            
            # 为每个测试集创建独立的可视化目录
            if args.test_vis_dir:
                vis_dir_for_ds = os.path.join(args.test_vis_dir, test_ds_name, ckpt_name)
            else:
                vis_dir_for_ds = None
            
            # 创建评估参数（每个测试集可能有不同的可视化目录）
            eval_args_ds = EvalArgs()
            eval_args_ds.val_vis_dir = vis_dir_for_ds
            
            # 评估
            metrics = evaluate_with_module(
                backbone, decoder, test_loader, device, eval_args_ds, 
                refine=refine, sdf_head=sdf_head
            )
            
            # 为每个指标添加数据集前缀
            for key, value in metrics.items():
                all_metrics[f"{test_ds_name}_{key}"] = value
            
            print(f"[Result on {test_ds_name}] mIoU: {metrics['mIoU']:.4f}, Fwβ: {metrics['Fw_beta']:.4f}, maxF: {metrics['maxF']:.4f}, MAE: {metrics.get('MAE', 'N/A')}")
        
        # 计算平均指标（跨所有测试集）
        avg_metrics = {}
        metric_keys = ["val_loss", "mIoU", "S_alpha", "Fw_beta", "mE_phi", "E_phi_adp", "MAE", "maxF"]
        for metric_key in metric_keys:
            values = []
            for test_ds_name in test_loaders.keys():
                key = f"{test_ds_name}_{metric_key}"
                if key in all_metrics:
                    val = all_metrics[key]
                    # 检查值是否有效（不是None，不是NaN，不是Inf）
                    if val is not None:
                        if isinstance(val, (int, float)):
                            if not (math.isnan(val) or math.isinf(val)):
                                values.append(float(val))
                        else:
                            try:
                                val_float = float(val)
                                if not (math.isnan(val_float) or math.isinf(val_float)):
                                    values.append(val_float)
                            except (ValueError, TypeError):
                                pass
            if values:
                avg_metrics[f"avg_{metric_key}"] = sum(values) / len(values)
        
        result = {
            "checkpoint": ckpt_name,
            "epoch": epoch,
            "path": ckpt_path,
            **all_metrics,
            **avg_metrics
        }
        results.append(result)
        
        # 打印汇总结果
        print(f"\n[Summary for {ckpt_name}]")
        for test_ds_name in test_loaders.keys():
            mIoU = all_metrics.get(f"{test_ds_name}_mIoU", 0)
            maxF = all_metrics.get(f"{test_ds_name}_maxF", 0)
            mae = all_metrics.get(f"{test_ds_name}_MAE", "N/A")
            print(f"  {test_ds_name:20s}: mIoU={mIoU:.4f}, maxF={maxF:.4f}, MAE={mae}")
        if avg_metrics:
            print(f"  {'Average':20s}: mIoU={avg_metrics.get('avg_mIoU', 0):.4f}, maxF={avg_metrics.get('avg_maxF', 0):.4f}, MAE={avg_metrics.get('avg_MAE', 'N/A')}")
    
    # 保存结果
    if args.output_csv:
        output_dir = os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建完整的fieldnames，包含所有测试集的指标
        fieldnames = ["checkpoint", "epoch", "path"]
        
        # 为每个测试集添加指标列
        metric_keys = ["val_loss", "mIoU", "S_alpha", "Fw_beta", "mE_phi", "E_phi_adp", "MAE", "maxF"]
        for test_ds_name in test_loaders.keys():
            for metric_key in metric_keys:
                fieldnames.append(f"{test_ds_name}_{metric_key}")
        
        # 添加平均指标
        for metric_key in metric_keys:
            fieldnames.append(f"avg_{metric_key}")
        
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row)
        
        print(f"\n[Saved] Results saved to {args.output_csv}")
    
    # 打印最佳结果（为每个测试集和平均指标）
    print("\n" + "="*80)
    print("[Best Results Summary]")
    print("="*80)
    
    # 为每个测试集打印最佳结果
    metrics_to_show = ["mIoU", "S_alpha", "Fw_beta", "mE_phi", "maxF"]
    for test_ds_name in test_loaders.keys():
        print(f"\n--- {test_ds_name} ---")
        for metric in metrics_to_show:
            metric_key = f"{test_ds_name}_{metric}"
            best = max(results, key=lambda x: x.get(metric_key, -1))
            value = best.get(metric_key, 0)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                print(f"Best {metric:10s}: {best['checkpoint']:20s} @ epoch {best['epoch']:3d} = {value:.4f}")
        
        mae_key = f"{test_ds_name}_MAE"
        best_mae = min(results, key=lambda x: x.get(mae_key, 1e9))
        mae_value = best_mae.get(mae_key, "N/A")
        if isinstance(mae_value, (int, float)) and not (isinstance(mae_value, float) and (math.isnan(mae_value) or math.isinf(mae_value))):
            print(f"Best {'MAE':10s}: {best_mae['checkpoint']:20s} @ epoch {best_mae['epoch']:3d} = {mae_value:.4f}")
    
    # 打印平均指标的最佳结果
    if avg_metrics:
        print(f"\n--- Average (across all test sets) ---")
        for metric in metrics_to_show:
            metric_key = f"avg_{metric}"
            best = max(results, key=lambda x: x.get(metric_key, -1))
            value = best.get(metric_key, 0)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                print(f"Best {metric:10s}: {best['checkpoint']:20s} @ epoch {best['epoch']:3d} = {value:.4f}")
        
        best_mae_avg = min(results, key=lambda x: x.get("avg_MAE", 1e9))
        mae_value = best_mae_avg.get("avg_MAE", "N/A")
        if isinstance(mae_value, (int, float)) and not (isinstance(mae_value, float) and (math.isnan(mae_value) or math.isinf(mae_value))):
            print(f"Best {'MAE':10s}: {best_mae_avg['checkpoint']:20s} @ epoch {best_mae_avg['epoch']:3d} = {mae_value:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
