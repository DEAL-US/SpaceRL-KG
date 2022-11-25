# RL-KG (Logo)

<p><img src="https://img.shields.io/badge/Python-3.8-blue"></p>

# Introduction

RLKG is an end-to-end framework which aids with generating new relations in knowledge graphs by leveraging on Reinforcement Learning techniques and algorithms.  
RLKG recieves any formatted dataset as input and provides embedding representations, usable agents, graphic information and an deployable API.  
With an easy to use GUI that helps users makes sense of the multitude of options that these types of frameworks usually present.  

RLKG was built and is mantained by [**DEAL research group**](https://deal.us.es/) as a tool for researchers which would want to further investigate these topics, or users which would like to use these agents for any purpose, such as inteligent recomendations, knowledge discovery, etc...

<br>  

# Contents


# Requirements
This project can be run in Windows (>= 7) or Linux (Ubuntu >=20.04)

The project is compatible with any version of **python 3.8** as in depends on [**DGL-KE**](https://aws-dglke.readthedocs.io/en/latest/) for embedding generation.  

If embedding generation is done beforehand, you can safely use **python >= 3.8.**

The software supports multiple GPUs for agent training and testing, to take advantage of this you'll need to install the latest version of your NVIDIA drivers, that can be obtained [**here**](https://www.nvidia.es/Download/index.aspx)


```
NOTE: we higly encourage the use of a virtual environment, you can get it with pip

python3 pip -m install --user virtualenv
virtualenv <env_name> 

To activate them run:
. <env_name>/bin/activate (Linux)
<env_name>/Scritps/activate (Windows)

```

**Pytorch** is required for embedding generation, they have CUDA compatibility out of the box and it can installed via pip with the following commands, or you can install it manually from their website, [**here**](https://pytorch.org/get-started/locally/)

`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`

**Tensorflow** 
new version of CUDA and 2.8 tf.

# Instalation

To use the elements of this work, first clone this repository anywhere in your system.