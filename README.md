# TriCoLo

<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

This repo is the official implementation for TriCoLo: **Tri**modal **Co**ntrastive **Lo**ss for Text to Shape Retrieval

([*Paper*](https://arxiv.org/pdf/2201.07366.pdf)) ([*Project Page*](https://3dlg-hcvc.github.io/tricolo/))

## Setup
### Conda (recommended)
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies.

```shell
# create and activate the conda environment
conda create -n tricolo python=3.10
conda activate tricolo

# install PyTorch 2.0.1
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

# install Python libraries
pip install .
```

### Pip (without conda)
```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch 2.0.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# install Python libraries
pip install .
```

## Data Preparation

### ShapeNet
Download [ShapeNet](https://shapenet.org/), and place `ShapeNetCore.v2` in the `data/text2shape-data` folder.


### Text2Shape (Chair & Table)

1. Download [Text2Shape](http://text2shape.stanford.edu/) and place `shapenet.json` and `processed_caption_{train/val/test}.p` in the `text2shape-data/chair_table` folder.
2. Download [ShapeNet solid voxels (Chair & Table)](http://text2shape.stanford.edu/):
   ```shell
   cd text2shape-data
   mkdir chair_table
   cd chair_table
   wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip
   wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_64_solid.zip
   wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_128_solid.zip
   unzip nrrd_256_filter_div_32_solid.zip
   unzip nrrd_256_filter_div_64_solid.zip
   unzip nrrd_256_filter_div_128_solid.zip
   ```
   Finally, the dataset files should be organized as follows:
   ```shell
   tricolo
   ├── data
   │   ├── preprocess_all_data.py
   │   ├── text2shape-data
   │   │   ├── ShapeNetCore.v2
   │   │   ├── chair_table
   │   │   │   ├── nrrd_256_filter_div_32_solid
   │   │   │   ├── nrrd_256_filter_div_64_solid
   │   │   │   ├── nrrd_256_filter_div_128_solid
   │   │   │   ├── processed_captions_train.p
   │   │   │   ├── processed_captions_val.p
   │   │   │   ├── processed_captions_test.p
   │   │   │   ├── shapenet.json
   ```

3. Preprocess the dataset
   ```shell
   python data/preprocess_all_data.py data=text2shape_chair_table +cpu_workers={num_processes}
   ```

4. Precache the CLIP embeddings (optional)
   ```shell
   python extract_clip_feats.py data=text2shape_chair_table data.image_size=224
   ```

### Text2Shape (C13)
1. Download [Text2Shape C13](https://aspis.cmpt.sfu.ca/projects/tricolo/data/c13.csv).

## Training, Inference and Evaluation
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.

```shell
# log in to WandB
wandb login

# train a model from scratch
# available voxel_encoder_name: SparseCNNEncoder, null
# available image_encoder_name: MVCNNEncoder, CLIPImageEncoder, null
# available text_encoder_name: BiGRUEncoder, CLIPTextEncoder
# available dataset_name: text2shape_chair_table, text2shape_c13
python train.py data={dataset_name} model.voxel_encoder={voxel_encoder_name} \
model.image_encoder={image_encoder_name} model.text_encoder={text_encoder_name} \
experiment_name={any_string}

# train a model from a checkpoint
python train.py data={dataset_name} model.voxel_encoder={voxel_encoder_name} \
model.image_encoder={image_encoder_name} model.text_encoder={text_encoder_name} \
experiment_name={checkpoint_experiment_name} ckpt_name={checkpoint_file_name}

# test a pretrained model
python test.py data={dataset_name} model.voxel_encoder={voxel_encoder_name} \
model.image_encoder={image_encoder_name} model.text_encoder={text_encoder_name} \
experiment_name={checkpoint_experiment_name} +ckpt_path={checkpoint_file_path}

# evaluate inference results
# currently unavailable
```
## Checkpoints

| Modality | Dataset                    | Split  | RR@1  | RR@5  | NDCG@5 | Download                                                                                              |
|:---------|:---------------------------|:-----|:------|:------|:-------|:------------------------------------------------------------------------------------------------------|
| Tri(I+V) | Text2Shape (Chair & Table) | Val | 12.60 | 33.34 | 23.30  | [chair_table_tri.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/chair_table_tri.ckpt)   |
| Bi(I)    | Text2Shape (Chair & Table) | Val | 11.67 | 30.63 | 21.49  | [chair_table_bi_i.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/chair_table_bi_i.ckpt) |
| Bi(V)    | Text2Shape (Chair & Table) | Val | 9.33  | 27.52 | 18.62  | [chair_table_bi_v.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/chair_table_bi_v.ckpt) |
| Tri(I+V) | Text2Shape (C13)           | Val | 12.96 | 34.87 | 24.19  | [c13_tri.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/c13_tri.ckpt)                   |
| Bi(I)    | Text2Shape (C13)           | Val | 11.89 | 33.48 | 22.96  | [c13_bi_i.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/c13_bi_i.ckpt)                 |
| Bi(V)    | Text2Shape (C13)           | Val | 9.73  | 29.24 | 19.69  | [c13_bi_v.ckpt](https://aspis.cmpt.sfu.ca/projects/tricolo/checkpoints/c13_bi_v.ckpt)                 |

## Acknowledgements
1. [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch): Our overall training framework is heavily based on the [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch) implementation. [*Paper*](https://arxiv.org/pdf/2010.00747.pdf)
2. [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) The MVCNN implementation we used is from [this](https://github.com/jongchyisu/mvcnn_pytorch) implementation. [*Paper*](https://arxiv.org/pdf/1505.00880.pdf)
3. [Text2Shape](https://github.com/kchen92/text2shape/): We download the dataset and modify the evaluation code from the original [Text2Shape dataset](http://text2shape.stanford.edu/). [*Paper*](https://arxiv.org/pdf/1803.08495.pdf)

We thank the authors for their work and the implementations.
