#!/usr/bin/env bash

# Make sure spaCy model is downloaded (extra precaution)
python -m spacy download en_core_web_sm

# Download NLTK data
python nltk_download.py
