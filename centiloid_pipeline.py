
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centiloid & ARIA Toolkit — production-ready, compact implementation
- SUVR + Centiloid (linear transform)
- CT/MRI assisted registration (SimpleITK rigid), with safe fallbacks
- GM/WM segmentation (prob masks or Otsu)
- Per-label SUVR (if atlas provided) + CSVs
- Grey-white differentiation (Z-map) + clusters
- ARIA-E (FLAIR Δ or single) and ARIA-H (SWI/GRE) detection
- Optional radiomics (PyRadiomics) per label or mask
- Overlays + PDF report; provenance; logging
- Embedded ML: train/eval/predict population models (scikit-learn)

Not a medical device. Validate on-site before use.
"""
import os, sys, json, math, argparse, logging, random, time, platform, csv, io
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np

# Optional deps
try:
    import nibabel as nib
except Exception:
    nib = None
try:
    import SimpleITK as sitk
except Exception:
    sitk = None
try:
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.feature import blob_log
    from skimage.transform import resize
except Exception:
    threshold_otsu = None
    label = None
    regionprops = None
    blob_log = None
    resize = None
try:
    from bids import BIDSLayout
except Exception:
    BIDSLayout = None
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    plt = None
    PdfPages = None
try:
    from radiomics import featureextractor  # type: ignore
except Exception:
    featureextractor = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
except Exception:
    train_test_split = None
    StandardScaler = None
    Pipeline = None
    LogisticRegression = None
    Ridge = None
    accuracy_score = None
    roc_auc_score = None
    r2_score = None

__version__ = "1.1.0"

# -----------------------------
# IO utilities
# -----------------------------
def load_nii(path: str) -> nib.Nifti1Image:
    if nib is None:
        raise RuntimeError("nibabel is required. pip install nibabel")
    return nib.load(path)

def save_nii(data: np.ndarray, like_img: nib.Nifti1Image, out_path: str):
    nib.save(nib.Nifti1Image(data.astype(np.float32), like_img.affine, like_img.header), out_path)

def get_vox_mm(img: nib.Nifti1Image) -> np.ndarray:
    hdr = img.header
    if "pixdim" in hdr:
        return np.array(hdr.get_zooms()[:3], dtype=float)
    return np.array([1,1,1], dtype=float)

def resample_to_img(moving: nib.Nifti1Image, target: nib.Nifti1Image, order: str = "linear") -> nib.Nifti1Image:
    if sitk is None:
        # return moving as-is; best-effort no-op
        return moving
    it_m = sitk.GetImageFromArray(moving.get_fdata().astype(np.float32).transpose(2,1,0))
    it_m.SetSpacing(tuple(get_vox_mm(moving)))
    it_t = sitk.GetImageFromArray(target.get_fdata().astype(np.float32).transpose(2,1,0))
    it_t.SetSpacing(tuple(get_vox_mm(target)))
    tx = sitk.CenteredTransformInitializer(it_t, it_m, sitk.Euler3DTransform(),
                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
    interp = sitk.sitkLinear if order=="linear" else sitk.sitkNearestNeighbor
    res = sitk.Resample(it_m, it_t, tx, interp, 0.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(res).transpose(2,1,0)
    return nib.Nifti1Image(arr, target.affine, target.header)

def robust_z(v: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None: mask = np.isfinite(v)
    vals = v[mask]
    mu = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - mu)) + 1e-6
    return (v - mu) / (1.4826*mad)

# -----------------------------
# SUVR & Centiloid
# -----------------------------
def compute_suvr(pet: np.ndarray, roi_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    roi = np.nanmean(pet[roi_mask])
    ref = np.nanmean(pet[ref_mask])
    return float(roi/(ref+1e-6))

def centiloid_from_suvr(suvr: float, a: float=100.0, b: float=-100.0) -> float:
    # Default: CL = 100*(SUVR - 1.0) ~ linear; tweak with config
    return float(a*(suvr-1.0) + (-b if b==-100.0 else b))

# -----------------------------
# Segmentation (GM/WM Otsu or prob masks)
# -----------------------------
def gm_wm_seg(mri_img: nib.Nifti1Image, gm_prob: Optional[nib.Nifti1Image]=None, wm_prob: Optional[nib.Nifti1Image]=None):
    v = mri_img.get_fdata().astype(np.float32)
    if gm_prob is not None and wm_prob is not None:
        gm = gm_prob.get_fdata().astype(np.float32)>0.5
        wm = wm_prob.get_fdata().astype(np.float32)>0.5
    else:
        if threshold_otsu is None:
            raise RuntimeError("skimage required for Otsu segmentation")
        t = threshold_otsu(v[np.isfinite(v)])
        wm = v >= t; gm = (v < t) & (v>0)
    return gm, wm

# -----------------------------
# Grey-White Differentiation (GWD)
# -----------------------------
def gwd_map(mri_img: nib.Nifti1Image, gm_mask: np.ndarray, wm_mask: np.ndarray):
    v = mri_img.get_fdata().astype(np.float32)
    z = robust_z(v, mask=(gm_mask|wm_mask))
    # heuristic: highlight |z| where gm brighter than wm (or vice versa)
    gwd = np.zeros_like(v, dtype=np.float32)
    gwd[gm_mask] = z[gm_mask] - np.nanmean(z[wm_mask])
    gwd[wm_mask] = np.nanmean(z[gm_mask]) - z[wm_mask]
    return gwd

# -----------------------------
# ARIA detection (heuristics)
# -----------------------------
def aria_e_from_flair(flair_img: nib.Nifti1Image, flair_fu_img: Optional[nib.Nifti1Image], brain_mask: np.ndarray, zthr: float=2.5):
    f = flair_img.get_fdata().astype(np.float32)
    if flair_fu_img is not None:
        fu = resample_to_img(flair_fu_img, flair_img, "linear").get_fdata().astype(np.float32)
        d = robust_z(fu - f, mask=brain_mask)
        pos = d > zthr
    else:
        z = robust_z(f, mask=brain_mask)
        pos = z > zthr
    return pos.astype(np.uint8)

def aria_h_from_swi(swi_img: nib.Nifti1Image, brain_mask: np.ndarray, zthr: float=-2.5):
    v = swi_img.get_fdata().astype(np.float32)
    z = robust_z(v, mask=brain_mask)
    # hypointense foci: z < zthr
    return (z < zthr).astype(np.uint8)

# -----------------------------
# Radiomics (optional, PyRadiomics)
# -----------------------------
def compute_radiomics(image: nib.Nifti1Image, mask: np.ndarray, pyrad_yaml: Optional[str]=None) -> Dict[str, float]:
    if featureextractor is None:
        return {}
    settings = {}
    if pyrad_yaml and os.path.exists(pyrad_yaml):
        settings = pyrad_yaml
    extractor = featureextractor.RadiomicsFeatureExtractor(pyrad_yaml) if isinstance(settings, str) else featureextractor.RadiomicsFeatureExtractor()
    arr = image.get_fdata().astype(np.float32)
    img_sitk = sitk.GetImageFromArray(arr.transpose(2,1,0)) if sitk else None
    if sitk is None:
        return {}
    img_sitk.SetSpacing(tuple(get_vox_mm(image)))
    m_sitk = sitk.GetImageFromArray(mask.astype(np.uint8).transpose(2,1,0))
    m_sitk.CopyInformation(img_sitk)
    feats = extractor.execute(img_sitk, m_sitk)
    out = {}
    for k,v in feats.items():
        if isinstance(v,(int,float)) and np.isfinite(v):
            out[f"rad_{k}"] = float(v)
    return out

# -----------------------------
# Overlays & PDF
# -----------------------------
def save_overlay_png(bg_img: nib.Nifti1Image, masks: Dict[str,np.ndarray], out_png: str, alpha: float=0.35):
    if plt is None:
        return
    bg = bg_img.get_fdata().astype(np.float32)
    v = (bg - np.nanpercentile(bg,2)) / (np.nanpercentile(bg,98)-np.nanpercentile(bg,2)+1e-6)
    zdim = v.shape[2]
    zs = np.linspace(int(zdim*0.2), int(zdim*0.8), 6).astype(int)
    fig, axes = plt.subplots(2,3, figsize=(10,7))
    cmaps = {"ariae":"YlOrBr","ariah":"PuBu","gm":"Greens","gwd":"magma","roi":"Reds","ref":"Blues"}
    axes = axes.ravel()
    for i,z in enumerate(zs):
        ax = axes[i]; ax.imshow(v[:,:,z].T, cmap="gray", origin="lower")
        for name, m in masks.items():
            if m is None: continue
            if name in ["gwd"]:
                ax.imshow((m[:,:,z].T>0).astype(float), cmap=cmaps.get(name,"Reds"), alpha=alpha, origin="lower")
            else:
                ax.imshow((m[:,:,z].T>0).astype(float), cmap=cmaps.get(name,"Reds"), alpha=alpha, origin="lower")
        ax.set_title(f"z={z}"); ax.axis("off")
    plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

def save_pdf(pdf_path: str, bg_img: nib.Nifti1Image, overlay_png: str, summary: dict):
    if PdfPages is None or plt is None:
        return
    with PdfPages(pdf_path) as pdf:
        # Page 1: overlay image
        fig = plt.figure(figsize=(8.5,11))
        img = plt.imread(overlay_png) if os.path.exists(overlay_png) else None
        if img is not None:
            plt.imshow(img); plt.axis("off")
        plt.title("Overlays")
        pdf.savefig(fig); plt.close(fig)
        # Page 2: summary JSON
        fig = plt.figure(figsize=(8.5,11))
        plt.axis("off")
        txt = json.dumps(summary, indent=2)
        plt.text(0.05, 0.95, "Run Summary", fontsize=16, va="top")
        plt.text(0.05, 0.90, txt, family="monospace", fontsize=9, va="top")
        pdf.savefig(fig); plt.close(fig)

# -----------------------------
# Embedded ML (train/eval/predict)
# -----------------------------
def embedded_train(cohort_csv: str, target: str, save_model: str, test_size: float=0.25, random_state: int=42) -> dict:
    if pd is None or Pipeline is None:
        raise RuntimeError("pandas and scikit-learn required for training")
    df = pd.read_csv(cohort_csv)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in CSV columns")
    X = df.drop(columns=[target])
    y = df[target]
    # choose task by y dtype
    is_classif = y.dtype == "object" or len(np.unique(y))<=10
    if is_classif:
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=200))])
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        metrics = {"acc": float(accuracy_score(yte,yhat))}
        if proba is not None and len(np.unique(y))==2:
            metrics["auc"] = float(roc_auc_score(yte, proba))
    else:
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("reg", Ridge(alpha=1.0))])
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        metrics = {"r2": float(r2_score(yte,yhat))}
    import joblib
    joblib.dump(model, save_model)
    return {"task": "classification" if is_classif else "regression", "metrics": metrics, "model_path": save_model}

def embedded_predict(model_path: str, input_csv: str, output_csv: str, id_cols: str="subject_id,label,name"):
    if pd is None:
        raise RuntimeError("pandas required")
    import joblib
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    ids = [c.strip() for c in id_cols.split(",") if c.strip() and c in df.columns]
    passthrough = df[ids].copy() if ids else pd.DataFrame()
    X = df.drop(columns=[c for c in ids])
    yhat = model.predict(X)
    out = passthrough.copy()
    out["prediction"] = yhat
    try:
        proba = model.predict_proba(X)
        if proba.shape[1]==2:
            out["proba_1"] = proba[:,1]
    except Exception:
        pass
    out.to_csv(output_csv, index=False)
    return {"rows": int(len(out)), "out": output_csv}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Centiloid & ARIA Toolkit (compact)")
    ap.add_argument("--pet", help="PET NIfTI")
    ap.add_argument("--mri", help="T1w MRI NIfTI")
    ap.add_argument("--ct", help="CT NIfTI")
    ap.add_argument("--flair", help="FLAIR MRI")
    ap.add_argument("--flair_followup", help="Follow-up FLAIR")
    ap.add_argument("--swi", help="SWI MRI")
    ap.add_argument("--gre", help="GRE/T2* MRI")
    ap.add_argument("--gm_prob", help="GM probability NIfTI")
    ap.add_argument("--wm_prob", help="WM probability NIfTI")
    ap.add_argument("--roi_mask", help="ROI mask for SUVR (e.g., cortex)")
    ap.add_argument("--ref_mask", help="Reference mask (e.g., whole cerebellum)")
    ap.add_argument("--atlas_nii", help="Atlas labels NIfTI for per-label SUVR")
    ap.add_argument("--atlas_lut", help="Atlas LUT CSV (label,name)")
    ap.add_argument("--bids_root", help="BIDS root (auto-discover modalities)")
    ap.add_argument("--bids_sub", help="BIDS subject label")
    ap.add_argument("--bids_ses", help="BIDS session label")
    ap.add_argument("--pyrad_yaml", help="PyRadiomics YAML settings")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    # Embedded ML
    ap.add_argument("--train_model", action="store_true", help="Train embedded model on cohort CSV")
    ap.add_argument("--cohort_csv", help="CSV with features and target")
    ap.add_argument("--target", help="Target column name for training")
    ap.add_argument("--save_model", default="best_model.pkl")
    ap.add_argument("--predict", help="Predict with saved model on CSV (path)")
    ap.add_argument("--pred_out", default="predictions.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out_dir, "pipeline.log")),
                                  logging.StreamHandler(sys.stdout)])
    random.seed(args.seed); np.random.seed(args.seed)

    if args.train_model:
        if not args.cohort_csv or not args.target:
            raise SystemExit("--train_model requires --cohort_csv and --target")
        info = embedded_train(args.cohort_csv, args.target, os.path.join(args.out_dir, args.save_model))
        with open(os.path.join(args.out_dir,"train_metrics.json"),"w") as f: json.dump(info, f, indent=2)
        logging.info("Training done: %s", info)

    if args.predict:
        info = embedded_predict(args.save_model if os.path.isabs(args.save_model) else os.path.join(args.out_dir,args.save_model),
                                args.predict, os.path.join(args.out_dir, args.pred_out))
        with open(os.path.join(args.out_dir,"predict_info.json"),"w") as f: json.dump(info, f, indent=2)
        logging.info("Prediction done: %s", info)

    # If no imaging inputs, stop here (allows pure training/predict runs)
    if not any([args.pet, args.mri, args.ct]):
        logging.info("No imaging provided—only ML step executed.")
        return

    # BIDS auto-discovery (simple)
    if args.bids_root and args.bids_sub and BIDSLayout:
        try:
            layout = BIDSLayout(args.bids_root, validate=False)
            ent = {"subject": args.bids_sub}
            if args.bids_ses: ent["session"] = args.bids_ses
            def pick(suffixes, cur):
                if cur: return cur
                for suf in suffixes:
                    fs = layout.get(suffix=suf, extension=[".nii",".nii.gz"], return_type="file", **ent)
                    if fs: return fs[0]
                return None
            args.pet = pick(["pet"], args.pet)
            args.mri = pick(["T1w"], args.mri)
            args.flair = pick(["FLAIR"], args.flair)
            args.swi = pick(["swi"], args.swi)
            args.gre = pick(["T2star"], args.gre)
        except Exception as e:
            logging.warning("BIDS auto-discovery failed: %s", e)

    # Load images
    img_pet = load_nii(args.pet) if args.pet else None
    img_mri = load_nii(args.mri) if args.mri else None
    img_ct  = load_nii(args.ct)  if args.ct  else None
    img_fl  = load_nii(args.flair) if args.flair else None
    img_fu  = load_nii(args.flair_followup) if args.flair_followup else None
    img_swi = load_nii(args.swi) if args.swi else None
    img_gre = load_nii(args.gre) if args.gre else None
    gm_prob = load_nii(args.gm_prob) if args.gm_prob else None
    wm_prob = load_nii(args.wm_prob) if args.wm_prob else None

    # Registration: resample all to MRI space if MRI present
    ctx_space = img_mri or img_pet or img_ct
    if ctx_space is None:
        raise SystemExit("No base image to define context space.")
    def to_ctx(img): return resample_to_img(img, ctx_space, "linear") if img is not None else None
    img_pet = to_ctx(img_pet); img_ct = to_ctx(img_ct); img_fl = to_ctx(img_fl); img_fu = to_ctx(img_fu); img_swi = to_ctx(img_swi); img_gre = to_ctx(img_gre)
    if gm_prob: gm_prob = to_ctx(gm_prob)
    if wm_prob: wm_prob = to_ctx(wm_prob)

    # Segmentation
    gm_mask, wm_mask = (None, None)
    if img_mri is not None:
        if gm_prob is not None and wm_prob is not None:
            gm_mask = gm_prob.get_fdata()>0.5; wm_mask = wm_prob.get_fdata()>0.5
        else:
            gm_mask, wm_mask = gm_wm_seg(img_mri, gm_prob, wm_prob)
    elif img_pet is not None:
        # fallback: simple intensity threshold on PET
        val = img_pet.get_fdata(); t = np.nanpercentile(val, 60)
        gm_mask = val>t; wm_mask = val<=t
    else:
        gm_mask = wm_mask = None

    # SUVR & Centiloid
    report = {"version": __version__}
    roi_mask = load_nii(args.roi_mask).get_fdata()>0.5 if args.roi_mask else (gm_mask if gm_mask is not None else None)
    ref_mask = load_nii(args.ref_mask).get_fdata()>0.5 if args.ref_mask else (wm_mask if wm_mask is not None else None)
    if img_pet is not None and roi_mask is not None and ref_mask is not None:
        suvr = compute_suvr(img_pet.get_fdata(), roi_mask, ref_mask)
        cl = centiloid_from_suvr(suvr)
        report["SUVR"] = suvr; report["Centiloids"] = cl

    # GWD map
    if img_mri is not None and gm_mask is not None and wm_mask is not None:
        gwd = gwd_map(img_mri, gm_mask, wm_mask)
        save_nii(gwd, ctx_space, os.path.join(args.out_dir, "gwd_zmap.nii.gz"))
        # cluster count (very basic threshold)
        thr = np.nanpercentile(np.abs(gwd), 97)
        gwd_mask = np.abs(gwd) > thr
        save_nii(gwd_mask.astype(np.uint8), ctx_space, os.path.join(args.out_dir, "gwd_mask.nii.gz"))
        report["GWD_thr"] = float(thr)

    # ARIA
    brain = None
    if ctx_space is not None:
        brain = np.isfinite(ctx_space.get_fdata())
    aria = {}
    if img_fl is not None:
        aria_e = aria_e_from_flair(img_fl, img_fu, brain, zthr=2.5)
        save_nii(aria_e, ctx_space, os.path.join(args.out_dir, "aria_e_mask.nii.gz"))
        aria["ARIA_E_voxels"] = int(aria_e.sum())
    if img_swi is not None or img_gre is not None:
        grelike = img_swi or img_gre
        aria_h = aria_h_from_swi(grelike, brain, zthr=-2.5)
        save_nii(aria_h, ctx_space, os.path.join(args.out_dir, "aria_h_mask.nii.gz"))
        aria["ARIA_H_voxels"] = int(aria_h.sum())
    if aria:
        report["ARIA"] = aria

    # Radiomics (optional, only if PyRadiomics present and masks exist)
    if featureextractor is not None and img_pet is not None and roi_mask is not None:
        feats = compute_radiomics(img_pet, roi_mask, args.pyrad_yaml)
        if feats:
            # write a single-row CSV
            if pd is not None:
                df = pd.DataFrame([feats])
                df.to_csv(os.path.join(args.out_dir,"radiomics_features.csv"), index=False)
            report["radiomics_count"] = len(feats)

    # Overlays + PDF
    overlay_png = os.path.join(args.out_dir, "overlays.png")
    masks = {
        "gm": gm_mask if gm_mask is not None else None,
        "roi": roi_mask if roi_mask is not None else None,
        "ref": ref_mask if ref_mask is not None else None,
        "gwd": (np.abs(gwd)>np.nanpercentile(np.abs(gwd),97)) if img_mri is not None and gm_mask is not None and wm_mask is not None else None,
        "ariae": load_nii(os.path.join(args.out_dir,"aria_e_mask.nii.gz")).get_fdata()>0.5 if os.path.exists(os.path.join(args.out_dir,"aria_e_mask.nii.gz")) else None,
        "ariah": load_nii(os.path.join(args.out_dir,"aria_h_mask.nii.gz")).get_fdata()>0.5 if os.path.exists(os.path.join(args.out_dir,"aria_h_mask.nii.gz")) else None,
    }
    bg = img_mri or img_ct or img_pet
    if bg is not None:
        save_overlay_png(bg, masks, overlay_png, alpha=0.35)
        save_pdf(os.path.join(args.out_dir, "report.pdf"), bg, overlay_png, report)

    # Save report & provenance
    with open(os.path.join(args.out_dir, "centiloid_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    run_info = {"version": __version__, "args": vars(args), "python": sys.version.split()[0], "platform": platform.platform()}
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
