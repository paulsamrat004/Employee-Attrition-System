#!/usr/bin/env bash
set -e
python src/train_model.py
python app.py
