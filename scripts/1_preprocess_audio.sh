#!/usr/bin/env bash
set -e
# Read transcript ( find Hi i'm ellie) and find t0
# save t0 in json file
# cut timeline after t0 -> adapt in feature files ( DAIC - clnf, covarep   EDAIC - densenet201, mfcc, openface_pose_gaze_au, vgg16)

python -m scripts.feature_extract.daicwoz.daic_audio_trim

# extract egemaps and save in processed file
# DAIC-WOZ egemaps in processed DAIC-WOZ/ReEgemaps25LLD (300-492 audio 189 participants)
# E-DAIC egemaps in processed E-DAIC/ReEgemaps25LLD (600-716 audio )

python -m scripts.feature_extract.daicwoz.daic_egemaps_extract 

python -m scripts.feature_extract.edaic.edaic_egemaps_extract