#!/usr/bin/env bash
set -e

# extract egemaps and save in processed file
# DAIC-WOZ egemaps in processed DAIC-WOZ/ReEgemaps25LLD (300-492 audio 189 participants)
# E-DAIC egemaps in processed E-DAIC/ReEgemaps25LLD (600-716 audio )

python -m scripts.feature_extract.daicwoz.daic_egemaps_extract 

python -m scripts.feature_extract.edaic.edaic_egemaps_extract