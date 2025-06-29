import os
from tqdm import tqdm

def check_label_file(path):
    with open(path, 'r') as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[ERROR] {path} line {i}: wrong format")
                return False
            try:
                vals = list(map(float, parts[1:]))
            except Exception as e:
                print(f"[ERROR] {path} line {i}: {e}")
                return False
            for v in vals:
                if not (0.0 <= v <= 1.0):
                    print(f"[ERROR] {path} line {i}: value {v} out of [0,1]")
                    return False
    return True

def main():
    label_dir = input("Enter path to YOLO label directory: ").strip()
    if not os.path.isdir(label_dir):
        print(f"Directory not found: {label_dir}")
        return
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    if not files:
        print("No .txt label files found.")
        return
    all_ok = True
    for fname in tqdm(files, desc="Checking labels"):
        path = os.path.join(label_dir, fname)
        if not check_label_file(path):
            all_ok = False
    print(f"\nChecked {len(files)} label files.")
    if all_ok:
        print("All label files are normalized (all values in [0, 1]).")
    else:
        print("Some label files have out-of-range values or errors.")

if __name__ == "__main__":
    main() 