#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    python3 -m venv .venv
    .venv/bin/pip install -r requirements_mac.txt
else
    # Other platforms
    python -m venv .venv
    pip install -r requirements.txt
fi
