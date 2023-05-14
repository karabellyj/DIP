### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```



#### Pre-Training

```
python ase/run.py --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --headless
```

#### Task-Training

```
python ase/run.py --task HumanoidHeading --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --headless
```
```
HumanoidHeading: ase/data/cfg/humanoid_sword_shield_heading.yaml
HumanoidLocation: ase/data/cfg/humanoid_sword_shield_location.yaml
```
To test a trained model, use the following command:
```
python ase/run.py --test --task HumanoidHeading --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --checkpoint [path_to_hlc_checkpoint]
```