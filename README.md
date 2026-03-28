# About

A GPU-accelerated linguistic text compressor (currently for the English Language only, will expand to other languages later) using hierarchical NLP and symbolic encoding made using python.

Primarily created as a research interest for https://github.com/openai/parameter-golf.

# Usage

Requires python3.11.x, any version beyond that will break due to spaCy.

# Installation

```
# How I did it, since I am on fedora

# fedora
sudo dnf install python3.11
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf clean all

sudo dnf module disable nvidia-driver # If using RPM Fusion or proprietary drivers
sudo dnf -y install cuda

export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# create a venv
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -m spacy download en_core_web_lg
.venv/bin/python -m pip install cupy-cuda12x
.venv/bin/python -m pip check
.venv/bin/python -m pytest compression/tests -v (should pass all tests)

# To run the full compression and decompression pipeline
.venv/bin/python -m test_round_trip_pipeline.py
```

# Note

- Your IDE may flag an import error in `stage4_discourse.py` for `fastcoref` if it's not launched from inside the virtual environment.

# Current testing sources

- Moby Dick (Project Gutenberg)
