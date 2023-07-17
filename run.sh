#!/usr/bin/env bash
ls

python3.9 -m pip install -r /home/hlcv_team017/HLCV2023/requirements.txt

python3.9 /home/hlcv_team017/HLCV2023/extract_attributes.py

python3.9 /home/hlcv_team017/HLCV2023/LENS/test_stripped_lens_long.py
