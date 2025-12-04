# Face alignment & canonicalization pipeline
# Paste into your notebook; assumes DATA_RAW, DATA_LM, DATA_PROCESSED exist (Path objects)

import os, math
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# parameters - adjust as needed
OUTPUT_SIZE = (256, 256)            # final crop size (width, height)
TARGET_FACE_SCALE = 0.55            # fraction of OUTPUT_SIZE used for face bbox (tuned)
MIN_LANDMARKS = 5                   # minimum pts required (sanity)
DATA_RAW = Path(DATA_RAW)           # from your notebook
DATA_LM = Path(DATA_LM)
DATA_ALIGNED = Path(DATA_PROCESSED) / "aligned"
DATA_ALIGNED.mkdir(parents=True, exist_ok=True)
FAIL_LOG = DATA_ALIGNED / "alignment_failures.txt"

# ---------- helpers ----------
def load_landmarks(lm_path):
    try:
        pts = np.load(str(lm_path))
        if pts is None or pts.size == 0:
            return None
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return None
        return pts
    except Exception:
        return None

def bbox_diag(pts):
    xs, ys = pts[:,0], pts[:,1]
    return np.linalg.norm([xs.max()-xs.min(), ys.max()-ys.min()])

def compute_normalized_shape(pts):
    """
    Normalize a shape by subtracting centroid and dividing by diagonal bbox length.
    Returns normalized pts (N,2), centroid, scale.
    """
    centroid = pts.mean(axis=0)
    scale = bbox_diag(pts)
    if scale <= 1e-6:
        scale = 1.0
    norm = (pts - centroid) / scale
    return norm, centroid, scale

# ---------- 1) build canonical (mean) normalized shape ----------
def build_mean_normalized_shape(landmarks_root, sample_limit=None):
    """
    Walk landmarks_root (mirror of data/raw) and compute average normalized shape.
    Returns mean_norm_shape (N,2). Only uses files that have same number of points.
    """
    norm_list = []
    shapes = []
    files_used = []
    for root, _, files in os.walk(landmarks_root):
        for fn in files:
            if not fn.lower().endswith('.npy'): 
                continue
            lm_path = Path(root) / fn
            pts = load_landmarks(lm_path)
            if pts is None or pts.shape[0] < MIN_LANDMARKS:
                continue
            norm, c, s = compute_normalized_shape(pts)
            norm_list.append(norm)
            shapes.append(pts.shape[0])
            files_used.append(lm_path)
            if sample_limit and len(norm_list) >= sample_limit:
                break
    if len(norm_list) == 0:
        raise RuntimeError("No valid landmarks found to compute mean shape.")
    # verify same landmark counts
    unique_counts = set(shapes)
    if len(unique_counts) != 1:
        # If shapes differ, pick the most common count and filter
        from collections import Counter
        cnt = Counter(shapes)
        common = cnt.most_common(1)[0][0]
        new_norm = [n for n,s in zip(norm_list, shapes) if s==common]
        norm_list = new_norm
        if len(norm_list) == 0:
            raise RuntimeError("No common-n landmark sets found.")
    mean_norm = np.mean(np.stack(norm_list, axis=0), axis=0)  # (N,2)
    return mean_norm

# ---------- 2) compute target (pixel) canonical shape in output canvas ----------
def canonical_target_shape(mean_norm_shape, output_size=OUTPUT_SIZE, face_scale=TARGET_FACE_SCALE):
    """
    Map mean normalized shape to pixel coordinates inside the output canvas.
    face_scale controls how large the face occupies the output (fraction of min(output_dim)).
    """
    w, h = output_size
    center_out = np.array([w/2.0, h/2.0], dtype=np.float32)
    # choose a target_scale in pixels: fraction of min dimension
    target_scale = min(w,h) * face_scale
    target_pts = mean_norm_shape * target_scale + center_out
    return target_pts.astype(np.float32)

# ---------- 3) compute similarity (affine partial) transform from src_pts -> target_pts ----------
def estimate_similarity_transform(src_pts, dst_pts):
    """
    Uses cv2.estimateAffinePartial2D to compute a similarity transform (scale+rot+trans).
    Returns 2x3 affine matrix. Raises if fails.
    """
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    # cv2.estimateAffinePartial2D returns (M, inliers)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000)
    if M is None:
        # fallback to least squares (no RANSAC)
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Could not estimate similarity transform.")
    return M

# ---------- 4) apply transform to image and landmarks ----------
def warp_image_and_landmarks(img_path, lm_path, M, output_size=OUTPUT_SIZE):
    img = np.array(Image.open(str(img_path)).convert('RGB'))
    h_out, w_out = output_size[1], output_size[0]
    # warp image: note cv2.warpAffine expects (width,height) order for dsize
    warped = cv2.warpAffine(img, M, (w_out, h_out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # transform landmarks
    pts = load_landmarks(lm_path)
    if pts is None:
        raise RuntimeError("Landmarks missing when warping.")
    # convert to homogeneous coordinates for multiplication
    ones = np.ones((pts.shape[0],1), dtype=np.float32)
    homo = np.concatenate([pts.astype(np.float32), ones], axis=1)  # (N,3)
    M_full = np.vstack([M, [0,0,1]])   # make 3x3
    transformed = (M_full @ homo.T).T[:, :2]
    return warped, transformed

# ---------- 5) batch processing function ----------
def align_all_images(landmarks_root=DATA_LM, raw_root=DATA_RAW, out_root=DATA_ALIGNED,
                     sample_limit=None, output_size=OUTPUT_SIZE, face_scale=TARGET_FACE_SCALE,
                     verbose=True):
    mean_norm = build_mean_normalized_shape(landmarks_root, sample_limit=sample_limit)
    target_pts = canonical_target_shape(mean_norm, output_size=output_size, face_scale=face_scale)
    # write mean_norm and target_pts for debugging
    np.save(out_root / "mean_norm_shape.npy", mean_norm)
    np.save(out_root / "target_pts.npy", target_pts)
    if verbose:
        print("Computed canonical target shape with", target_pts.shape[0], "points. Output size:", output_size)
    # process each image
    failures = []
    count = 0
    for root, _, files in os.walk(raw_root):
        for fn in files:
            if not fn.lower().endswith(('.jpg','.png','.jpeg')):
                continue
            rel = os.path.relpath(os.path.join(root, fn), raw_root)
            lm_path = Path(landmarks_root) / (os.path.splitext(rel)[0] + ".npy")
            if not lm_path.exists():
                failures.append((rel, "no_landmarks"))
                if verbose: print("Skipping (no landmarks):", rel)
                continue
            pts = load_landmarks(lm_path)
            if pts is None or pts.shape[0] < MIN_LANDMARKS:
                failures.append((rel, "bad_landmarks"))
                if verbose: print("Skipping (bad landmarks):", rel)
                continue
            if pts.shape[0] != mean_norm.shape[0]:
                # landmark count mismatch: skip or try to resample? skip for now
                failures.append((rel, f"n_pts_mismatch_{pts.shape[0]}"))
                if verbose: print("Skipping (landmark count mismatch):", rel, "has", pts.shape[0], "expected", mean_norm.shape[0])
                continue
            # estimate transform src_pts -> target_pts
            try:
                M = estimate_similarity_transform(pts, target_pts)
            except Exception as e:
                failures.append((rel, "estimation_failed:" + str(e)))
                if verbose: print("Transform estimation failed for", rel, ":", e)
                continue
            try:
                warped, trans_pts = warp_image_and_landmarks(os.path.join(root, fn), lm_path, M, output_size=output_size)
            except Exception as e:
                failures.append((rel, "warp_failed:" + str(e)))
                if verbose: print("Warp failed for", rel, ":", e)
                continue
            # save outputs: aligned image and aligned landmarks
            out_dir = Path(out_root) / Path(rel).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = out_dir / (Path(fn).stem + "_aligned.png")
            out_lm_path = out_dir / (Path(fn).stem + "_aligned.npy")
            Image.fromarray(warped).save(out_img_path)
            np.save(out_lm_path, trans_pts)
            count += 1
            if verbose and (count % 50 == 0):
                print("Aligned:", count)
    # write failure log
    with open(str(FAIL_LOG), "w") as f:
        for item in failures:
            f.write(f"{item[0]}\t{item[1]}\n")
    if verbose:
        print("Done. Aligned:", count, "Fail:", len(failures), " (log:", FAIL_LOG, ")")
    return count, failures

# ---------- visualization helper ----------
def visualize_alignment_examples(aligned_root=DATA_ALIGNED, raw_root=DATA_RAW, n=6):
    examples = []
    for root, _, files in os.walk(aligned_root):
        for fn in files:
            if fn.endswith("_aligned.png"):
                examples.append(os.path.join(root, fn))
            if len(examples) >= n: break
        if len(examples) >= n: break
    import matplotlib.pyplot as plt
    for p in examples:
        rawname = str(Path(p).name).replace("_aligned.png", "")
        # find and show
        img = np.array(Image.open(p).convert('RGB'))
        lm_path = Path(p).with_suffix('.npy')
        pts = np.load(lm_path) if lm_path.exists() else None
        plt.figure(figsize=(3,3)); plt.imshow(img); plt.axis('off')
        if pts is not None:
            for (x,y) in pts:
                plt.scatter([x],[y], c='r', s=6)
        plt.show()

# ---------- Example usage ----------
# count, fails = align_all_images(sample_limit=500, output_size=(256,256), face_scale=0.55)
# visualize_alignment_examples(n=6)
