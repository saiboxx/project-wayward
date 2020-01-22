# Project Wayward

## Who are we?
We are computer science students at LMU Munich and this project is happening as part of the Autonomous Systems Practical.
Have fun checking out our stuff!

Cheers

## What's happening in this project?

Our task was to implement an reinforcement learning algorithm, which is able to handle one or more environments that are provided by the Unity ML-Agents package ([click](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md)).
Our goal was to succeed in the crawler environment and the proceed to take on the walker environment.
We first started using [DDPG](https://arxiv.org/pdf/1509.02971.pdf) but this ended in mediocre results at the best.
Subsequently another approach using [PPO](https://arxiv.org/pdf/1707.06347.pdf) with [GAE](https://arxiv.org/pdf/1506.02438.pdf) was implemented, which yields remarkable results and is able to beat all benchmarks in all Unity ML-Agents environments.
The algorithm seems relative insensible to parameter settings as we could achieve competetive results with a variety of configurations. We believe that the current settings in `config.yml` represent a good compromise between a fast-learning and also stable algorithm.

For the implementation of the neural architectures we decided to rely on PyTorch. The project provides a two or three layer network for training, where for PPO each layer is a fully-connected one with [ELU activation](https://arxiv.org/pdf/1511.07289.pdf) and [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf).


## How to use the project?

1. **Build an executable**
   
   The basis for training is an executable of an ML-Agents environment, which can be compiled using the Unity IDE.
   For convenience reasons please save the build in the directory `./executables/<your_env_name>` and name the file`<your_env_name>`.
   Alternatively a compiled build of the walker environment, which can also be used, is included in this repository.

2. **Adjust parameters in the config file**

   It is mandatory that you set the `EXECUTABLE` parameter in `config.yml` with `<your_env_name>`.
   Feel free to adjust the parameters of the algorithms as you please.

3. **Start training**

   To start the training execute `make train-ddpg` or `make train-ppo` according to the desired training algorithm.
   Our project also includes the possibility of utilizing the environments of the OpenAI Gym package. The standard game is Lunar Lander, which can be started by `make mock-ddpg` or `make mock-ppo`.

4. **Monitor training**

   The whole training process can be monitored using tensorboard. The specified log directory is `runs_ddpg` oder `runs_ppo` respectively.
   
5. **Watch trained agent succed**

   During training the models will be saved in `./models`. By specifying the correct path to the chosen model in the config file and using `make run-ddpg` or `make run-ppo` the environment will be booted in inference mode.
