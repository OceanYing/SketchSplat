### Create env
conda create -n sketchsplat python=3.8.18

### Install dependents:
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y cudatoolkit=11.8 -c pytorch -c nvidia
### (Optional) If your environment does not already include cudatoolkit, you may also need the CUDA development toolkit:
conda install -y cudatoolkit-dev=11.8 -c conda-forge

pip install numpy==1.26.4
pip install ipdb==0.13.13 tensorboard==2.17.0 tqdm==4.66.4 open3d==0.18.0 dacite==1.8.1 plyfile==1.0.3
pip install gsplat==1.0.0
pip install scikit-image==0.21.0 scikit-learn==1.5.0 point-cloud-utils==0.31.0