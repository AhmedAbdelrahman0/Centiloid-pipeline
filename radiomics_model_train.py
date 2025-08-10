
#!/usr/bin/env python3
import argparse, os, sys, subprocess
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--save_model", default="best_model.pkl")
    ap.add_argument("--out_dir", default="out_train")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cmd = [sys.executable, "centiloid_pipeline.py", "--out_dir", a.out_dir, "--train_model", "--cohort_csv", a.cohort_csv, "--target", a.target, "--save_model", a.save_model]
    print("Running:", " ".join(cmd)); sys.exit(subprocess.call(cmd))
