#!/bin/bash
set -e

# Create virtual environment in .venv
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Prompt user for Gemini API key and save as environment variable
read -p "Enter your Gemini API key: " GEMINI_API_KEY
echo "export GEMINI_API_KEY=\"$GEMINI_API_KEY\"" >> .venv/bin/activate

echo "Setup complete. Virtual environment created and packages installed."