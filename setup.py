from setuptools import setup

setup(
    name="cifar_10_pytorch",
    version="1.0",
    description="Using different models on CIFAR-10 dataset",
    author="MohammadReza Beygifard",
    author_email="samanbeygifard73@gmail.com",
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "tqdm==4.65.0",
        "beepy==1.0.7",
    ],
)
