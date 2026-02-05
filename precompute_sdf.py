# precompute_sdf.py
# -*- coding: utf-8 -*-
import os, argparse, glob, numpy as np
from tqdm import tqdm
import cv2

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    raise RuntimeError("需要 scipy:  pip install scipy")

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def signed_distance(mask01: np.ndarray) -> np.ndarray:
    """ inside<0, outside>0; 使用 edt 计算 signed distance, 并归一化到 [-1,1] """
    m = (mask01 > 0.5).astype(np.uint8)
    if m.sum()==0:
        dist_out = distance_transform_edt(1 - m).astype(np.float32)
        sdf = dist_out
    elif m.sum()==m.size:
        dist_in = distance_transform_edt(m).astype(np.float32)
        sdf = -dist_in
    else:
        dist_out = distance_transform_edt(1 - m).astype(np.float32)
        dist_in  = distance_transform_edt(m).astype(np.float32)
        sdf = dist_out - dist_in
    s = max(1.0, float(np.abs(sdf).max()))
    sdf = sdf / s
    return sdf.astype(np.float32)

def collect_masks(root, mask_dir_name, mask_ext):
    # 支持 split/flat 两种布局
    paths=[]
    split_dir = [os.path.join(root, "train", mask_dir_name),
                 os.path.join(root, "test",  mask_dir_name)]
    flat_dir  = os.path.join(root, mask_dir_name)
    for d in split_dir:
        if os.path.isdir(d):
            for p in glob.glob(os.path.join(d, f"*{mask_ext}")):
                paths.append(p)
    if not paths and os.path.isdir(flat_dir):
        for ext in (mask_ext, mask_ext.upper(), *IMG_EXTS, *[e.upper() for e in IMG_EXTS]):
            paths += glob.glob(os.path.join(flat_dir, f"*{ext}"))
    paths = sorted(list(dict.fromkeys(paths)))
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./segdata")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--label_dir_name", type=str, default="mask")
    ap.add_argument("--mask_ext", type=str, default=".png")
    ap.add_argument("--out_dir_name", type=str, default="sdf")
    args = ap.parse_args()

    root = os.path.join(os.path.abspath(args.data_dir), args.dataset)
    out_root = os.path.join(root, args.out_dir_name)
    os.makedirs(out_root, exist_ok=True)

    mask_paths = collect_masks(root, args.label_dir_name, args.mask_ext)
    if not mask_paths:
        raise RuntimeError(f"未找到标注: {root}/{args.label_dir_name}")

    print(f"[SDF-Precompute] masks={len(mask_paths)}  -> out_dir={out_root}")
    for mp in tqdm(mask_paths):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        m01 = (m>127).astype(np.float32)
        sdf = signed_distance(m01)
        # 输出目录与 mask 同级（把 mask_dir_name 替换成 out_dir_name）
        rel = os.path.relpath(mp, root)
        rel_out = rel.replace(args.label_dir_name, args.out_dir_name)
        out_path = os.path.join(root, os.path.splitext(rel_out)[0] + ".npy")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, sdf)
    print("[Done]")

if __name__ == "__main__":
    main()
