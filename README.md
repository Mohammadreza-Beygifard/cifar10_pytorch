# cifar10_pytorch
In This Repo I will push my adventure on solving CIFAR 10

# Note Books

In the notebooks directory you can find the Jupyter notebook, which I use to explore the models and play around with decision factors.

As I have a macbook pro I use mps platform for training the model with GPU.
Windows and linux users should use cuda instead.

# Install all the dependencies

Install Conda: following the [Official Anaconda installation](https://docs.anaconda.com/free/anaconda/install/index.html)

Then install Numpy, Pandas, Matplotlib and beppy to play the successful sound when the training is finished launching:

```shell
pip install --verbose -e .
```

Then install the torch script with this command:

```shell
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

## Suggestion

To avoid messing up with your local envirnoment, I suggest you to create a conda virtual env and install the dependencies there and launch the Jupyter notbook with the kernel in the venv. You can follow [This guide by janakiev](https://janakiev.com/blog/jupyter-virtual-envs/) for more info.