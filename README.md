# Escaping-Stochastic-Traps-With-Aleatoric-Mapping-Agents

This repo presents the code used for the NeurIPS 2020 Biological and Artificial Reinforcement Learning Workshop paper "Escaping Stochastic Traps with Aleatoric Mapping Agents".

Authors: Augustine N. Mavor-Parker, Kimberly A. Young, Caswell Barry, Lewis D. Griffin

# Installing Dependencies

pip install -r requirements.txt

# Reproducing Results 

For the minigrid experiments, navigate to the rl-starter directory. Then:

```bash uncertainty_experiment.sh```

For the noisy MNIST results go to the MNIST directory and run the Noisy_MNIST.ipynb

# Code Acknowledgements 

The [rl-starter-files](https://github.com/lcswillems/rl-starter-files) files were used as a base for the algorithms in the minigrid experiment. Furthermore, the underlying RL code of the rl-starter files package [torch-ac](https://github.com/lcswillems/torch-ac) was added into this repo to be altered to add the intrinsic reward bonus. Thanks [Lucas Willems](https://github.com/lcswillems) for your fantastic open source contributions.

When developing the aleatoric uncertainty quantification code, the following repos were helpful:

https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision

https://github.com/pmorerio/dl-uncertainty 

https://github.com/hmi88/what
