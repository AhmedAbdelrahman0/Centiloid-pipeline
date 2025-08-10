
#!/usr/bin/env python3
# Streamlit GUI wrapper
import streamlit as st, os, sys, subprocess, tempfile, shutil

st.set_page_config(page_title="Centiloid & ARIA Toolkit", layout="centered")
st.title("Centiloid & ARIA Toolkit")

with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        bids_root = st.text_input("BIDS root (optional)", value="")
        bids_sub = st.text_input("BIDS subject", value="")
        bids_ses = st.text_input("BIDS session", value="")
        outdir = st.text_input("Output folder", value="out_gui")
    with col2:
        pet = st.file_uploader("PET NIfTI", type=["nii","nii.gz"])
        mri = st.file_uploader("T1 NIfTI", type=["nii","nii.gz"])
        ct  = st.file_uploader("CT NIfTI", type=["nii","nii.gz"])
        flair = st.file_uploader("FLAIR NIfTI", type=["nii","nii.gz"])
        swi = st.file_uploader("SWI NIfTI", type=["nii","nii.gz"])
        gre = st.file_uploader("GRE/T2* NIfTI", type=["nii","nii.gz"])
    run = st.form_submit_button("Run")

def save_upload(u, tmp, name):
    if u is None: return None
    path = os.path.join(tmp, name)
    with open(path, "wb") as f: f.write(u.read())
    return path

if run:
    tmp = tempfile.mkdtemp()
    try:
        pet_p = save_upload(pet, tmp, "pet.nii.gz")
        mri_p = save_upload(mri, tmp, "t1.nii.gz")
        ct_p  = save_upload(ct, tmp, "ct.nii.gz")
        flair_p = save_upload(flair, tmp, "flair.nii.gz")
        swi_p   = save_upload(swi, tmp, "swi.nii.gz")
        gre_p   = save_upload(gre, tmp, "gre.nii.gz")
        cmd = [sys.executable, "centiloid_easy.py", "--out", outdir]
        for label, val in [("bids_root",bids_root),("bids_sub",bids_sub),("bids_ses",bids_ses)]:
            if val: cmd += [f"--{label}", val]
        for label, val in [("pet",pet_p),("mri",mri_p),("ct",ct_p),("flair",flair_p),("swi",swi_p),("gre",gre_p)]:
            if val: cmd += [f"--{label}", val]
        st.code(" ".join(cmd))
        subprocess.check_call(cmd)
        st.success("Done!")
        for k in ["centiloid_report.json","report.pdf","overlays.png"]:
            p = os.path.join(outdir,k)
            if os.path.exists(p):
                st.write(f"✓ {k}: {p}")
    except subprocess.CalledProcessError as e:
        st.error(f"Run failed: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
