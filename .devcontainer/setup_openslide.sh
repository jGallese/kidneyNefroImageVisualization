#!/bin/bash

# Install OpenSlide and related dependencies
echo "Installing OpenSlide dependencies..."
sudo apt-get update
sudo apt-get install -y libopenslide-dev openslide-tools libjpeg-dev libtiff-dev

# Verify installation paths
echo "Updating shared library paths..."
sudo ldconfig

# Confirm if OpenSlide is installed
if ldconfig -p | grep -q "libopenslide.so.0"; then
    echo "OpenSlide installed successfully."
else
    echo "Failed to locate libopenslide.so.0. Check library paths."
fi

# Install Python dependencies
pip install openslide-python

# Verify the Python installation
python -c "import openslide; print('OpenSlide Python library version:', openslide.__version__)"
