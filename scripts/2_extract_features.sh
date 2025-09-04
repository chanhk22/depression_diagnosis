#!/usr/bin/env bash
set -e

# cut timeline after t0 -> adapt in feature files ( DAIC - clnf, covarep   EDAIC - densenet201, mfcc, openface_pose_gaze_au, vgg16)

python -m scripts.feature_extract.daicwoz.daic_feature_mask


python -m scripts.feature_extract.edaic.edaic_feature_mask

# from preprocessing.clnf_parser, preprocessing.dvlog_visual_parser normalize the data and save in processed/features file
# CLNF_features.txt -> CLNF_features.npy
python -m scripts.preprocess_landmarks