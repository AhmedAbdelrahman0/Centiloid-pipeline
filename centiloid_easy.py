
#!/usr/bin/env python3
# Minimal, user-friendly wrapper around centiloid_pipeline.py
import os, sys, argparse, subprocess, shlex

def main():
    ap = argparse.ArgumentParser(description="One-click Centiloid/ARIA runner")
    ap.add_argument("--pet", help="PET NIfTI (or use --bids_root --bids_sub)")
    ap.add_argument("--mri", help="T1w NIfTI")
    ap.add_argument("--ct", help="CT NIfTI")
    ap.add_argument("--flair", help="FLAIR NIfTI")
    ap.add_argument("--swi", help="SWI NIfTI")
    ap.add_argument("--gre", help="GRE/T2* NIfTI")
    ap.add_argument("--bids_root", help="BIDS root")
    ap.add_argument("--bids_sub", help="BIDS subject")
    ap.add_argument("--bids_ses", help="BIDS session")
    ap.add_argument("--masks_dir", required=False, help="(Optional) masks folder if you have them")
    ap.add_argument("--roi_mask", help="ROI mask (e.g., cortex)")
    ap.add_argument("--ref_mask", help="Reference region (e.g., cerebellum)")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--threads", default="0")
    ap.add_argument("--log_level", default="INFO")
    # Embedded ML
    ap.add_argument("--train_model", action="store_true")
    ap.add_argument("--cohort_csv")
    ap.add_argument("--target")
    ap.add_argument("--predict")
    ap.add_argument("--save_model", default="best_model.pkl")
    ap.add_argument("--pred_out", default="predictions.csv")
    args = ap.parse_args()

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__),"centiloid_pipeline.py"),
           "--out_dir", args.out, "--log_level", args.log_level, "--threads", str(args.threads)]
    for k in ["pet","mri","ct","flair","swi","gre","bids_root","bids_sub","bids_ses","roi_mask","ref_mask"]:
        v = getattr(args, k)
        if v: cmd += [f"--{k}", v]
    # ML
    if args.train_model:
        cmd += ["--train_model","--cohort_csv", args.cohort_csv, "--target", args.target, "--save_model", args.save_model]
    if args.predict:
        cmd += ["--predict", args.predict, "--save_model", args.save_model, "--pred_out", args.pred_out]

    print("Running:\n", " ".join(shlex.quote(c) for c in cmd))
    os.makedirs(args.out, exist_ok=True)
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
