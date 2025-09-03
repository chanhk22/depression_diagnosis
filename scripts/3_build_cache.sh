#!/usr/bin/env bash
set -e
python -m precache.window_cache --config configs/default.yaml --dataset all
echo "Window cache built."