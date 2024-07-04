# Robust Inverse Constrained Reinforcement Learning under Model Misspecification
This is the code for the paper [Robust Inverse Constrained Reinforcement Learning under Model Misspecification](https://openreview.net/pdf?id=pkUl39b0in) published at ICML 2024. Note that:
1. The experimental environment is mainly based on the [MuJoCo](https://mujoco.org/).
2. The implementation is based on the code from [ICRL-benchmark](https://github.com/Guiliang/ICRL-benchmarks-public/tree/main).

## Create Python Environment 
1. Please install the conda before proceeding.
2. Create a conda environment and install the packages:
   
```
mkdir save_model
mkdir evaluate_model
conda env create -n ricrl python=3.9 -f python_environment.yml
conda activate ricrl
```
You can also first install Python 3.9 with the torch (2.0.1+cu117) and then install the packages listed in `python_environment.yml`.

## Setup Experimental Environments 
### Setup MuJoCo Environment (you can also refer to [MuJoCo Setup](https://github.com/Guiliang/ICRL-benchmarks-public/blob/main/virtual_env_tutorial.md))
1. Download the MuJoCo version 2.1 binaries for Linux or OSX.
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use mujoco-py.
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment

export MUJOCO_PY_MUJOCO_PATH=YOUR_MUJOCO_DIR/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:YOUR_MUJOCO_DIR/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Generate Expert Demonstration
Note that we have generated the expert data for ease of usage, and you can download it through [expert_data]().

Alternatively, you can also generate your own dataset with different settings such as different constraints or noise levels through the following steps (here we use the `Blocked Half-Cheetah` environment with noise level 1e-3 as an example):
### 1. Train expert agents with ground-truth constraints.
Firstly we should train an expert agent (PPO-Lag) with ground-truth constraints:
```
# run PPO-Lag knowing the ground-truth constraints
python train_policy.py ../config/train/BlockedHalfCheetah/PPO-Lag_HCWithPos.yaml -n 5 -s 123
```

### 2. Sample trajectories of the expert
After training the expert agent, we can get the expert demonstration through sampling from it:
```
# run data generation
python generate_data_for_constraint_inference.py -n 5 -mn your_expert_file_path -tn PPO-Lag-HC -ct no-constraint -rn 0
```
Note that you need to replace the `your_expert_file_path` with the saved path of your trained expert. You can find it through `save_model/PPO-Lag-HC/your_expert_file_path`.

## Train ICRL Algorithms
We use the `Blocked Half-Cheetah` environment with seed 123 and opponent strength 0.05 (alpha=0.95) for AR-ICRL as an example. You can also use different seeds or modify the noise level using different configs.

```
# train BC2L
python train_icrl.py ...config/train/BlockedHalfCheetah/BC2L_HCWithPos.yaml -n 5 -s 123

# train MEICRL
python train_icrl.py ...config/train/BlockedHalfCheetah/MEICRL_HCWithPos.yaml -n 5 -s 123

# train VICRL
python train_icrl.py ...config/train/BlockedHalfCheetah/VICRL_HCWithPos.yaml -n 5 -s 123

# train IRCO
python train_icrl.py ...config/train/BlockedHalfCheetah/IRCO_HCWithPos.yaml -n 5 -s 123

# train ARICRL (Robust PPO-Lag Version)
python train_icrl.py ...config/train/BlockedHalfCheetah/ARICRL_HCWithPos-RPPO-Lag-op5e-2.yaml -n 5 -s 123

# train ARICRL (Robust Dual PPO Version)
python train_icrl.py ...config/train/BlockedHalfCheetah/ARICRL_HCWithPos-RDPPO-op5e-2.yaml -n 5 -s 123
```

## Evaluate Results
After training the algorithms, we can evaluate their performance under testing environments with different transition dynamics through the following steps (we use the `Blocked Half-Cheetah` environment as an example):

1. Modify the `config/evaluate/eval_BlockedHalfCheetah.yaml` to change the testing environment with different types of noises (i.e., full_random / partial_random / attack) and the noise scale (eval_noise_mean, eval_noise_std)
2. Fill in the 'model_paths' list in `interface/evaluate_halfcheetah.py` with the log path of the algorithm you want to evaluate, and then run it.

## Welcome to Cite and Star
If you have any questions, please contact me via shengxu1@link.cuhk.edu.cn.

If you feel the work helpful, please use the citation:
```
@inproceedings{xu2024ricrl,
  title={Robust Inverse Constrained Reinforcement Learning under Model Misspecification},
  author={Xu, Sheng and Liu, Guiliang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
  url={https://openreview.net/pdf?id=pkUl39b0in}
}
```
