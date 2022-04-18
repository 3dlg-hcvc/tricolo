# TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval
This repo is contains the official implementation for the paper [TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval](https://arxiv.org/pdf/2201.07366.pdf) ([*Project Page*](https://3dlg-hcvc.github.io/tricolo/)).

## Environment Installation
1. Create conda environment
```
conda create -n tricolo python=3.6
conda activate tricolo
```
2. Install required packages and [CLIP](https://github.com/openai/CLIP)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c conda-forge tensorboard
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
3. Install [sparseconvolution](https://github.com/facebookresearch/SparseConvNet)
```
git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash develop.sh
```
<!-- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch -->

## Dataset Download
1. Create datasets folder
```
mkdir datasets
cd datasets
```
2. Download [Text2Shape](http://text2shape.stanford.edu/) Dataset and unzip
```
wget http://text2shape.stanford.edu/dataset/text2shape-data.zip
unzip text2shape-data.zip
```

* ShapeNet

    1. Download [Colored Voxels](http://text2shape.stanford.edu/) and unzip
    ```
    cd text2shape-data/shapenet
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_64_solid.zip
    wget http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_128_solid.zip
    unzip nrrd_256_filter_div_32_solid.zip
    unzip nrrd_256_filter_div_64_solid.zip
    unzip nrrd_256_filter_div_128_solid.zip
    ```
    2. Download [ShapeNet](https://shapenet.org/) v1 and v2 (for multi-view), and place in datasets folder.

* Primitives

    1. Download [Colored Voxels](http://text2shape.stanford.edu/) and unzip
    ```
    cd text2shape-data/primitives
    wget http://text2shape.stanford.edu/dataset/primitives/primitives.zip
    unzip primitives.zip
    ```
The directory structure should look like:
```
This Repository
|--datasets
    |--ShapeNetCore.v1
    |--ShapeNetCore.v2
    |--text2shape
        |--shapenet
            |--nrrd_256_filter_div_32_solid
            |--nrrd_256_filter_div_64_solid
            |--nrrd_256_filter_div_128_solid
        |--primitives
            |--primitives.v2
```

## Dataset Preprocessing
1. Save dataset into json files
```
cd preprocess
python create_modelid_caption_mapping.py --dataset shapenet
python create_modelid_caption_mapping.py --dataset primitives
```
2. Render multi-view images from ShapeNet
```
python run_render.py
```
3. Generate npz files for each model
```
python gen_all_npz.py
```

## Download Pretrained Model
1. Download the pretrained model [here](https://www.dropbox.com/sh/yfseeplx2u5zsbh/AAAonT5VO_DDmedyl6wtKdQsa?dl=0).
2. Place into the repository
```
This Repository
|--logs
```
3. Evaluate the pretrained models
```
python run_retrieval_val.py --exp Tri --split test
python run_retrieval_val.py --exp BiV --split test
python run_retrieval_val.py --exp BiI --split test
python run_retrieval_val.py --exp Primitives --split test
```

## Train Model
1. ShapeNet Dataset

    Train the Tri-modal model on ShapeNet
    ```
    python run_retrieval.py --config_file t3ds/configs/shapenet.yaml
    ```
    Train the Bi-modal Image model on ShapeNet
    ```
    python run_retrieval.py --config_file t3ds/configs/shapenet_I.yaml
    ```
    Train the Bi-modal Voxel model on ShapeNet
    ```
    python run_retrieval.py --config_file t3ds/configs/shapenet_V.yaml
    ```
2. Primitives Dataset

    Train the Bi-modal Voxel model on Primitives (only Bi-modal V with resolution 32 is supported)
    ```
    python run_retrieval.py --config_file t3ds/configs/primitives.yaml
    ```

## Evaluate Model
1. Evaluate trained model. Here enter the trained log folder name under ./logs/retrieval for the flag --exp. For the --split flag enter either 'valid' or 'test' to evaluate on that dataset split.
```
# Example
python run_retrieval_val.py --exp Apr14_15-12-21 --split test
```
2. Evaluate zero-shot [CLIP](https://github.com/openai/CLIP) on the Text2Shape retrieval.
```
python run_retrieval_val.py --clip --split test
```

## Notes
In our original paper we used a dense 3D CNN for the voxels. To save memory we also implement a sparse version using Facebook's [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). The Tri-modality model using sparse 3D convolutions is able to run on a RTX 2080 Ti with 11 GB of VRAM and the performance is on par with our original dense implementation. The sparse convolution model is turned off by default in the config files. However, if you would like to use the sparse model you can set the **sparse_model** flag in config files to True.

The memory usage for the models in the original paper are as listed (reported by torch.cuda.max_memory_reserved):
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>#Params</th>
            <th>Resolution</th>
            <th>BatchSize</th>
            <th>Memory</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5>Bi(V)</td>
            <td rowspan=5>8.7 M</td>
            <td>32<sup>3</sup></td>
            <td>128</td>
            <td>2.1 GB</td>
        </tr>
        <tr>
            <td rowspan=4>64<sup>3</sup></td>
             <td>32</td>
             <td>2.8 GB</td>
        </tr>
        <tr>
            <td >64</td>
            <td>5.7 GB</td>
        </tr>
        <tr>
            <td >128</td>
            <td>10.2 GB</td>
        </tr>
        <tr>
            <td >256</td>
            <td>21.6 GB</td>
        </tr>
        <tr>
            <td rowspan=7>Bi(I)</td>
            <td rowspan=7>13.3 M</td>
            <td>64<sup>2</sup></td>
            <td rowspan=3>128</td>
            <td>3.3 GB</td>
        </tr>
        <tr>
            <td>128<sup>2</sup></td>
            <td>9.5 GB</td>
        </tr>
        <tr>
            <td >224<sup>2</sup></td>
            <td>25.7 GB</td>
        </tr>
        <tr>
            <td rowspan=4>128<sup>2</sup></td>
            <td>32</td>
            <td>3.0 GB</td>
        </tr>
        <tr>
            <td >64</td>
            <td>6.6 GB</td>
        </tr>
        <tr>
            <td >128</td>
            <td>9.5 GB</td>
        </tr>
        <tr>
            <td >256</td>
            <td>15.3 GB</td>
        </tr>
        <tr>
            <td>Tri(I+V)</td>
            <td>20.6 M</td>
            <td>v64<sup>3</sup>i128<sup>2</sup></td>
            <td>128</td>
            <td>17.0 GB</td>
        </tr>
    </tbody>
</table>

The memory usage for the models using the sparse implementation (reported by torch.cuda.max_memory_reserved):
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Resolution</th>
            <th>BatchSize</th>
            <th>Memory</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Bi(V)</td>
            <td>64<sup>3</sup></td>
            <td>128</td>
            <td>9.9 GB</td>
        </tr>
        <tr>
            <td>Tri(I+V)</td>
            <td>v64<sup>3</sup>i128<sup>2</sup></td>
            <td>128</td>
            <td>9.8 GB</td>
        </tr>
    </tbody>
</table>


## Acknowledgements
1. [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch): Our overall training framework is heavily based on the [ConVIRT](https://github.com/edreisMD/ConVIRT-pytorch) implementation. [*Paper*](https://arxiv.org/pdf/2010.00747.pdf)
2. [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) The MVCNN implementation we used is from [this](https://github.com/jongchyisu/mvcnn_pytorch) implementation. [*Paper*](https://arxiv.org/pdf/1505.00880.pdf)
3. [Text2Shape](https://github.com/kchen92/text2shape/): We download the dataset and modify the evaluation code from the original [Text2Shape dataset](http://text2shape.stanford.edu/). [*Paper*](https://arxiv.org/pdf/1803.08495.pdf)
4. [ShapeNet Renderer](https://github.com/panmari/stanford-shapenet-renderer): We modified this script to render the multi-view images for the ShapeNet dataset.

We thank the authors for their work and the implementations.
