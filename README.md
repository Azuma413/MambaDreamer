# MambaDreamer
DreamreV3をベースにGRUをMambaで置き換えようという試み。

### 動作テスト
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/runx
```
GPUは必須です。いくつか必要なライブラリがあるのでエラーを見て適宜対処してください。  
### 実行に必要なライブラリの一部
```
causal-conv1d>=1.1.0
mamba-ssm
setuptools==60.0.0
torch
torchvision
pandas
matplotlib
ruamel.yaml
moviepy
einops
protobuf
gym
mujoco
dm_control
scipy
atari-py
opencv-python
numpy
tensorboard
```
### コードについて
`models.py`の53行目をFalseにするとGRUが使われ，TrueにするとMambaが使われます。  
`mamba_simple.py`のMambaCellのコンストラクタからMambaのハイパーパラメータを変更できます。

### 注意
このリポジトリは[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)と[mamba](https://github.com/state-spaces/mamba/tree/main?tab=Apache-2.0-1-ov-file)のコードに強く影響されています。