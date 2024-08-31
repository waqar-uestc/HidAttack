
### HidAttack: An Effective and Undetectable Model Poisoning Attack to Federated Recommenders
___

This repository contains the code, dataset, and related guidelines for the practical implementation of our work **HidAttack: An Effective and Undetectable Model Poisoning Attack to Federated Recommenders**. 

## Introduction
Our study introduces HidAttack, a novel undetectable poisoning strategy for Federated Recommender Systems (FedRS) that raises the exposure ratio of targeted items with minimal Byzantine clients. By leveraging a bandit model to generate diverse poisoned gradients, HidAttack evades existing defenses while significantly enhancing the exposure rate of targeted items without affecting recommendation accuracy. 

Here we provide a comprehensive overview of our PyTorch implementation for the proposed method, including model architecture, parameter settings, and experimental configurations. This includes details on the Neural Collaborative Filtering model in FL settings, training procedures, and specific configurations for our attack strategy. Here are the specific steps:

___

## Usage
First, You need to install the required dependencies:  
```
pip install -r requirements.txt 
```


To evaluate the HidAttack performance, run: 
```bash
python HidAttack.py 
```

## Support
If you have any questions, feel free to contact us for assistance !!


