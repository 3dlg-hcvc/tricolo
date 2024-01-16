from setuptools import find_packages, setup

setup(
    name="tricolo",
    version="2.0",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/tricolo",
    description="Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval",
    packages=find_packages(include=("tricolo", "data")),
    install_requires=[
        "tqdm", "lightning", "wandb", "hydra-core", "pynrrd", "trimesh", "spconv-cu113", "pyrender",
        "CLIP @ git+https://github.com/openai/CLIP.git", "efficientnet_pytorch"
    ]
)
