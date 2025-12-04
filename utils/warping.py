from pathlib import Path
from utils.warp_delaunay import warp_image_delaunay
import numpy as np
from PIL import Image

src = np.array(Image.open("data/raw/Actor/p_0001.jpg").convert("RGB"))
src_pts = np.load("data/landmarks/Actor/p_0001.npy")
# create exaggerated target points (example: scale jaw points outward)
dst_pts = src_pts.copy()
center = dst_pts.mean(axis=0)
# example exaggeration factor on landmarks 0..16 (jaw)
dst_pts[0:17] = center + 1.2*(dst_pts[0:17]-center)
out = warp_image_delaunay(src, src_pts, dst_pts, output_shape=src.shape[:2])
Image.fromarray(out).save("data/processed/p_0001_warped.png")
