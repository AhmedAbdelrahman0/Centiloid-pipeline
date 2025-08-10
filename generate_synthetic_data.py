
#!/usr/bin/env python3
# Generate synthetic BIDS dataset (2 subjects x 2 sessions) + basic MASKS
import os, json, numpy as np
import nibabel as nib

def aff(vox=2.0): import numpy as _np; return _np.array([[vox,0,0,0],[0,vox,0,0],[0,0,vox,0],[0,0,0,1]], float)

def save(path, data, vox=2.0):
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff(vox)), path)

def sphere(shape, center, radius):
    Z,Y,X = np.indices(shape); return ((X-center[0])**2+(Y-center[1])**2+(Z-center[2])**2)<=radius**2

def make_sub(root, sub, ses):
    sesdir = os.path.join(root, f"sub-{sub}", f"ses-{ses}")
    petd = os.path.join(sesdir,"pet"); anatd = os.path.join(sesdir,"anat"); ctd = os.path.join(sesdir,"ct")
    os.makedirs(petd, exist_ok=True); os.makedirs(anatd, exist_ok=True); os.makedirs(ctd, exist_ok=True)
    shape=(64,64,48); rng=np.random.default_rng(42+int(sub)*10+int(ses))
    pet=rng.normal(1.0,0.05,shape); t1=rng.normal(700,80,shape); flair=rng.normal(900,100,shape); swi=rng.normal(1000,120,shape); ct=rng.normal(30,10,shape)
    pet[sphere(shape,(20,32,24),10)] += 0.25; pet[sphere(shape,(44,32,24),10)] += 0.15; pet[sphere(shape,(32,12,8),8)] -= 0.10
    if ses=="02": flair[sphere(shape,(32,40,30),6)] += 300
    if sub=="02": swi[sphere(shape,(40,28,18),3)] -= 400
    save(os.path.join(petd, f"sub-{sub}_ses-{ses}_pet.nii.gz"), pet)
    save(os.path.join(anatd, f"sub-{sub}_ses-{ses}_T1w.nii.gz"), t1)
    save(os.path.join(anatd, f"sub-{sub}_ses-{ses}_FLAIR.nii.gz"), flair)
    save(os.path.join(anatd, f"sub-{sub}_ses-{ses}_swi.nii.gz"), swi)
    save(os.path.join(ctd,  f"sub-{sub}_ses-{ses}_ct.nii.gz"), ct)
    json.dump({"TracerName":"FBP"}, open(os.path.join(petd,f"sub-{sub}_ses-{ses}_pet.json"),"w"))
    json.dump({"Modality":"MR","Sequence":"T1w"}, open(os.path.join(anatd,f"sub-{sub}_ses-{ses}_T1w.json"),"w"))
    json.dump({"Modality":"MR","Sequence":"FLAIR"}, open(os.path.join(anatd,f"sub-{sub}_ses-{ses}_FLAIR.json"),"w"))
    json.dump({"Modality":"MR","Sequence":"SWI"}, open(os.path.join(anatd,f"sub-{sub}_ses-{ses}_swi.json"),"w"))
    json.dump({"Modality":"CT"}, open(os.path.join(ctd,f"sub-{sub}_ses-{ses}_ct.json"),"w"))

def make_masks(root="MASKS"):
    os.makedirs(root, exist_ok=True); import numpy as np
    shape=(64,64,48); from numpy import indices, sqrt
    Z,Y,X = indices(shape); cx,cy,cz=32,32,24; r=np.sqrt((X-cx)**2+(Y-cy)**2+(Z-cz)**2)
    gm=np.zeros(shape,np.float32); wm=np.zeros(shape,np.float32); ctx=np.zeros(shape,np.uint8)
    gm[(r>10)&(r<20)]=0.8; wm[(r<=10)]=0.9; ctx[(r>12)&(r<18)]=1
    save(os.path.join(root,"gm_prob.nii.gz"), gm); save(os.path.join(root,"wm_prob.nii.gz"), wm); save(os.path.join(root,"ctx_mask.nii.gz"), ctx)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--bids_root", default="data"); ap.add_argument("--subs", nargs="+", default=["01","02"]); ap.add_argument("--sessions", nargs="+", default=["01","02"]); a=ap.parse_args()
    os.makedirs(a.bids_root, exist_ok=True)
    for sub in a.subs:
        for ses in a.sessions: make_sub(a.bids_root, sub, ses)
    make_masks("MASKS")
    print(f"Created synthetic data at {a.bids_root} and MASKS/")
