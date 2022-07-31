# This paper got a rebrand (new title and some other changes for a resubmission), please use the new repo [here](https://github.com/self-supervisor/How_to_stay_curious_while_avoiding_noisy_TVs)

Official anonymised code submission for: 

*Escaping Stochastic Traps with Aleatoric Mapping Agents*

Each directory contains different experiment code and instructions 
to run the code.

Logging is performed with weights and biases, if you want to get 
logged data you need to either have a weights and biases account 
or comment out the wandb lines.

# Escaping-Stochastic-Traps-With-Aleatoric-Mapping-Agents

This repo presents the code used for the **NeurIPS 2020 Biological and Artificial Reinforcement Learning Workshop** paper "Escaping Stochastic Traps with Aleatoric Mapping Agents".

**Authors**: Augustine N. Mavor-Parker, Kimberly A. Young, Caswell Barry, Lewis D. Griffin

**Abstract**: Exploration in environments with sparse rewards is difficult for artificial agents. Curiosity driven learning --- using feed-forward prediction errors as intrinsic rewards --- has achieved some success in these scenarios, but fails when faced with action dependent noise sources. We present aleatoric mapping agents (AMAs), a neuroscience inspired solution modeled on the cholinergic system of the mammalian brain. AMAs aim to explicitly ascertain which dynamics of the environment are unpredictable, regardless of whether those dynamics are induced by the actions of the agent. This is achieved by generating separate forward predictions for the mean and variance of future states and reducing intrinsic rewards for those transitions with high aleatoric variance. We show AMAs are able to effectively circumvent action dependent stochastic traps that plague conventional curiosity driven agents.


# Code Acknowledgements

The [rl-starter-files](https://github.com/lcswillems/rl-starter-files) files were used as a base for the algorithms in the minigrid experiment. Furthermore, the underlying RL code of the rl-starter files package [torch-ac](https://github.com/lcswillems/torch-ac) was added into this repo to be altered to add the intrinsic reward bonus. Thanks [Lucas Willems](https://github.com/lcswillems) for your fantastic open source contributions.

The retro game experiments were built from: 

https://github.com/openai/large-scale-curiosity

When developing the aleatoric uncertainty quantification code, the following repos were helpful:

https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision

https://github.com/pmorerio/dl-uncertainty

https://github.com/hmi88/what

When developing the forward prediction architecture the following repos were helpful:

https://github.com/facebookresearch/impact-driven-exploration

https://github.com/L1aoXingyu/295pytorch-beginner/

Misc:

https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream

# If you find that this code/paper is useful enough for a citation use the following bibtex:

```
@article{mavor2021escaping,
  title={Escaping Stochastic Traps with Aleatoric Mapping Agents},
  author={Mavor-Parker, Augustine N and Young, Kimberly A and Barry, Caswell and Griffin, Lewis D},
  journal={arXiv preprint arXiv:2102.04399},
  year={2021}
}
```

