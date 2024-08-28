# Graph Network Simulator (GNS) Paddle framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/geoelements/gns/main/license.md)

The development and operation of SPH algorithm are realized on the AI Studio of PaddlePaddle, and the flow field data of 3D complex calculation examples are obtained. The problem of missing radius_graph function in Paddle PGL module was solved by customizing model functions, and simulation data training and prediction based on graph neural network (GNS) model framework were successfully deployed and implemented. In response to the poor learning performance of GNS in 3D SPH data, physical constraints such as edge features and mass conservation constraints were introduced to improve the accuracy of the model. After optimization, the improved GNS model exhibits excellent generalization ability, is suitable for various particle algorithms, and shows better computational accuracy and stability than the GNS model based on PyTorch in the Paddle framework.

> Training GNS/MeshNet on simulation data
```shell
# For particulate domain,
python train.py --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" -ntraining_steps=100
```

> Resume training

To resume training specify `model_file` and `train_state_file`:

```shell
# For particulate domain,
python train.py --data_path="<input-training-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>"  --model_file="model.pt" --train_state_file="train_state.pt" -ntraining_steps=100
```

> Rollout prediction
```shell
# For particulate domain,
python train.py --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"
```

> Render
```shell
# For particulate domain,
python render_rollout.py --output_mode="gif" --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
```

--Output mode can choose gif or vtk, where `vtk` files to visualize in ParaView.


