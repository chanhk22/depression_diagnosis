#!/usr/bin/env bash
set -e
# Read transcript ( find Hi i'm ellie) and find t0
# save t0 in json file
# cut timeline after t0 -> adapt in feature files ( DAIC - clnf, covarep   EDAIC - densenet201, mfcc, openface_pose_gaze_au, vgg16)

python -m scripts.feature_extract.daicwoz.daic_audio_trim

python -m scripts.feature_extract.daicwoz.daic_feature_extract

python -m scripts.feature_extract.edaic.edaic_feature_extract