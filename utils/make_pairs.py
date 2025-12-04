# utils/make_pairs.py
import os, csv
ROOT_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_PAIRS = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pairs")
os.makedirs(OUT_PAIRS, exist_ok=True)

def make_pairs_by_basename():
    # example: if p_0001.jpg and c_0001.jpg exist in same folder, pair them
    pairs = []
    for actor in os.listdir(ROOT_RAW):
        a_dir = os.path.join(ROOT_RAW, actor)
        if not os.path.isdir(a_dir): continue
        files = sorted([f for f in os.listdir(a_dir) if f.lower().endswith(('.jpg','.png'))])
        photos = [f for f in files if f.startswith('p')]
        caris = [f for f in files if f.startswith('c')]
        for p,c in zip(photos, caris):
            pairs.append((os.path.join(actor,p), os.path.join(actor,c)))
    # save csv
    csvf = os.path.join(OUT_PAIRS, "pairs_by_basename.csv")
    with open(csvf, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["photo","caricature"])
        writer.writerows(pairs)
    print("Saved", csvf)

if __name__ == "__main__":
    make_pairs_by_basename()
