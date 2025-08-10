
#!/usr/bin/env python3
# One-click: generate -> run -> open PDF
import os, sys, subprocess, time, webbrowser
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def run(cmd):
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)

def main():
    bids = os.path.join(ROOT,"data"); out = os.path.join(ROOT,"out_synth_01_01")
    run([sys.executable, os.path.join(HERE,"generate_synthetic_data.py"), "--bids_root", bids, "--subs","01","02","--sessions","01","02"])
    run([sys.executable, os.path.join(ROOT,"centiloid_easy.py"), "--bids_root", bids, "--bids_sub","01","--bids_ses","01", "--out", out])
    pdf = os.path.join(out,"report.pdf")
    if os.path.exists(pdf): webbrowser.open("file://"+pdf)

if __name__=="__main__":
    main()
