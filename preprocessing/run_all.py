import subprocess as sp, sys

STEPS = [
    ["python","preprocessing/audio_lld_opensmile.py","--env","configs/env.yaml"],
    ["python","preprocessing/dvlog_ingest.py","--env","configs/env.yaml"],
    ["python","preprocessing/daic_landmarks.py","--env","configs/env.yaml"],
    ["python","preprocessing/edaic_privileged.py","--env","configs/env.yaml"],
    ["python","preprocessing/mean_shape_fit.py","--env","configs/env.yaml"],
    ["python","preprocessing/sync_window.py","--env","configs/env.yaml","--win","configs/window.yaml"],
]

if __name__=="__main__":
    for cmd in STEPS:
        print(">>", " ".join(cmd))
        ret = sp.call(cmd)
        if ret!=0:
            sys.exit(ret)
