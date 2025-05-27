# Image Generation

### Installation

Create a conda environment with the following command:

```bash
conda create -n slcd_image python=3.10
conda activate slcd_image
pip install uv
export UV_TORCH_BACKEND=auto # for auto selecting the torch backend
uv pip install torch # you can change this for the version of cuda you have.
uv pip install -r requirements.txt
```

Download the initial dataset and pre-calculated rewards (or generate by scripts/sample_images.py, scripts/score_imgs_aes.py, scripts/score_imgs_comp.py):

```
https://drive.google.com/drive/folders/1aaWRbe1uUaXsiS84CMPjdYSvA2fvoG95
```

Download the prepared statistical data (used for FID calculation from 70,000 images):

```
https://drive.google.com/drive/folders/1aaWRbe1uUaXsiS84CMPjdYSvA2fvoG95
```

Download the model checkpoints:

```
https://drive.google.com/drive/folders/1aaWRbe1uUaXsiS84CMPjdYSvA2fvoG95
```

### Training

```bash
python image_generation/valuefunction/DAgger_train_value_function_aes.py --init_latent_dir {PATH TO DATASET}
```
and 

```bash
python image_generation/valuefunction/DAgger_train_value_function_comp.py --init_latent_dir {PATH TO DATASET}
```

### Sampling

use the `guided_generation` function in the `image_generation/valuefunction/DAgger_train_value_function_aes.py` or `image_generation/valuefunction/DAgger_train_value_function_comp.py`

like this

```python
from valuefunction.DAgger_train_value_function_comp import guided_generation
from compressibility_scorer import SinusoidalTimeConvNet
import torch
device = 'cuda'
latent_dim = 4  
convnet = torch.load({PATH TO CKPT}).to(device)
convnet.eval()
eta = 10
guidance_strength = 150
random_seed = 43
images, scores = guided_generation(convnet, eta, guidance_strength, scale_coeff = 0.0, eval_num=1, return_dis=True, random_seed=random_seed, caption='cat')
```

### For FID calculation

```bash
pip install pytorch-fid
```

and then use the following script:

```bash
python -m pytorch_fid {GENERATED IMAGE PATH} {PATH to forfid.npz downloaded before} --dim 64
```

### Acknowledgement  

Our codebase is directly built on top of [SVDD-image](https://github.com/masa-ue/SVDD-image)  

