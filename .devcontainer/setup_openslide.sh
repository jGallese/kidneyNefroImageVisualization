#!/bin/bash

# Install OpenSlide and related dependencies
echo "Installing OpenSlide dependencies..."
sudo apt-get update
sudo apt-get install -y libopenslide-dev openslide-tools libjpeg-dev libtiff-dev

# Update shared library paths to ensure OpenSlide is discoverable
echo "Updating shared library paths..."
sudo ldconfig

# Confirm OpenSlide installation
if ldconfig -p | grep -q "libopenslide.so.0"; then
    echo "OpenSlide system libraries installed successfully."
else
    echo "Failed to locate libopenslide.so.0. Check library paths."
fi

# Now install Python dependencies
echo "Installing Python libraries..."
pip install openslide-python

# Verify the Python OpenSlide installation
python -c "import openslide; print('OpenSlide Python library version:', openslide.__version__)"
