# utils/extract_landmarks.py
import os, json
import numpy as np
from PIL import Image
import face_alignment
from face_alignment import LandmarksType
import torch
from tqdm import tqdm

ROOT_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_LM = os.path.join(os.path.dirname(__file__), "..", "data", "landmarks")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, device=DEVICE, flip_input=False)

def largest_face_index(preds):
    areas = []
    for pts in preds:
        xs = pts[:,0]; ys = pts[:,1]
        areas.append((xs.max()-xs.min())*(ys.max()-ys.min()))
    return int(np.argmax(areas))

def process_file(in_path, out_base):
    img = np.array(Image.open(in_path).convert('RGB'))
    preds = fa.get_landmarks(img)
    meta = {"ok": False}
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    if preds is None:
        np.save(out_base + ".npy", np.array([]))
        meta.update({"ok": False, "num_faces": 0})
        with open(out_base + ".json", "w") as f:
            json.dump(meta, f)
        return False
    idx = largest_face_index(preds)
    pts = preds[idx].astype(np.float32)
    np.save(out_base + ".npy", pts)
    xs, ys = pts[:,0], pts[:,1]
    meta.update({"ok": True, "num_faces": len(preds),
                 "chosen_face_index": int(idx),
                 "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                 "shape": list(pts.shape)})
    with open(out_base + ".json", "w") as f:
        json.dump(meta, f)
    return True

def walk_and_process():
    for root, _, files in os.walk(ROOT_RAW):
        rel = os.path.relpath(root, ROOT_RAW)
        for fn in tqdm(files):
            if not fn.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            in_path = os.path.join(root, fn)
            basename = os.path.splitext(fn)[0]
            out_base = os.path.join(OUT_LM, rel, basename)
            process_file(in_path, out_base)

if __name__ == "__main__":
    walk_and_process()
