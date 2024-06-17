#!/bin/bash

REQUIRED_PYTHON="3.10"

version_ge() {
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1" ]
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not be found."
        return 1
    fi

    # Get the current Python version as a string
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')

    # Check if the current Python version is greater than or equal to the required version
    if ! version_ge "$PYTHON_VERSION" "$REQUIRED_PYTHON"; then
        echo "This script requires Python $REQUIRED_PYTHON or higher but found $PYTHON_VERSION"
        return 1
    fi

    return 0
}

setup_hf() {
    echo "Please enter your Hugging Face token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing HF_TOKEN in .env file..."
        echo "HF_TOKEN=$token" >> .env
        
        echo "Installing Hugging Face CLI..."
        yes | pip install --upgrade huggingface_hub
        echo "Logging in to Hugging Face CLI..."
        huggingface-cli login --token $token
    else
        echo "No token entered. Skipping..."
    fi
}

setup_together() {
    echo "Please enter your Together AI token (press Enter to skip):"
    read -r token
    if [ -n "$token" ]; then
        echo "Storing TOGETHER_API_KEY in .env file..."
        echo "TOGETHER_API_KEY=$token" >> .env
    else
        echo "No token entered. Skipping..."
    fi
}

setup_venv() {
    echo "Setting up venv..."

    python -m venv venv
    source venv/bin/activate

    echo "Done setting up venv!"
}

install_requirements() {
    echo "Installing requirements..."

    yes | pip install -r requirements.txt --upgrade

    echo "Done installing requirements!"
}

echo "Running set up..."

echo "" > .env

check_python
if [ $? -ne 0 ]; then
    return 1
fi

setup_hf
setup_together
setup_venv
install_requirements

echo "All set up!"
