# Multi-task Meta Deep Q-Network (M2DQN)

The main goal of this work is to introduce a DQN algorithm that integrates multi-task learning and meta-learning. The algorithm aims to learn how to initialize the DQN network under new tasks based on training from multiple source tasks, thereby enhancing the generalization ability of the DQN.

## System Setup
Make sure that you have Jupyter Notebook. The main Python packages and versions involved are as follows
- mxnet: 1.6.0
- matplotlib: 3.3.4
- numpy: 1.18.5
- pandas: 1.1.5
- gym: 0.21.0


## What is included
```
M2DQN/
├── lib/
    ├── agents/
        ├── dqn.py
        ├── mtdqn.py  (Multi-task DQN)
        ├── medqn.py  (Meta DQN)
        ├── m2dqn.py  (Multi-task Meta DQN)
    ├── envs/
        ├── slicing_env.py
    ├── plotting.py
    ├── utils.py
├── rl_agents_training/
    ├── non_accelerated.ipynb
    ├── TL-DQN_Training.ipynb
    ├── TL-DQN_Adaptation.ipynb
    ├── Multi-task_DQN.ipynb
    ├── Meta_DQN.ipynb
    ├── Multi-task_Meta_DQN.ipynb
├── saved_models/
├── visualization/
    ├── visualization.ipynb
├── README.md
```
## How to use
* Go to the ```rl_agents_training``` folder to run the training of agents:
    * Transfer Learning: 
        * TL-DQN_Training.ipynb
        * TL-DQN_Adaptation.ipynb
    * Multi-task Learning:
        * Multi-task_DQN.ipynb
    * Meta-Learning:
        * Meta_DQN.ipynb
    * base:
        non_accelerated.ipynb
* Go to the ```visualization``` folder to check some of the graphs generated from the training data. (**The result data needs to be trained to obtain**)

