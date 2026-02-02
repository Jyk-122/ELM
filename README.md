# ELM (WIP)
This repository is the official implementation of [ICLR'26][Equilibrium Language Models](https://openreview.net/pdf?id=lqJT6xmuH3). 

![framework](./assets/overview.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Datasets
We recommend the following form to organize datasets:
```datasets_form
└─datasets
    ├─train_set_1
    |   └─train.json
    ├─train_set_2
    |   └─train.json
    ├─test_set_1
    |   └─train.json
    └─test_set_2
        └─test.json
```
Here we provide mathematical reasoning datasets for reference. As illustrated in paper, the training set is MetaMathQA and the evaluation dataset is GSM8K and MATH. In repository we provide example training datas as the demonstration and the full datasets can be achieved in corresponding [huggingface](https://huggingface.co/datasets/meta-math/MetaMathQA). The evaluation dataset is achieved from [github](https://github.com/meta-math/MetaMath/tree/main/data/test).

## Training



## Evaluation



## Inference



## Results

We compare ELM with other pruning methods, including Sheared-LLaMA, Shortened-Llama, ReplaceMe, LLM-Streamline. We prune 8 layers of Qwen2.5-1.5B/Qwen2.5-7B/Llama3.1-3B (28% non-embedding parameters) and finetune on different specific tasks. ELMs achieve siginificant improvements. Details are available in our paper.

![result](./assets/performance.png)