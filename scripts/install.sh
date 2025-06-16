set -x 

# conda create -n haptic python=3.10 -y
# conda activate haptic


pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt211/download.html

pip install -r requirements.txt

git submodule update --init --recursive
pip install -v -e third-party/ViTPose
