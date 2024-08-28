# Graph Network Simulator (GNS) Paddle framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/geoelements/gns/main/license.md)

The development and operation of SPH algorithm are realized on the AI Studio of PaddlePaddle, and the flow field data of 3D complex calculation examples are obtained. The problem of missing radius_graph function in Paddle PGL module was solved by customizing model functions, and simulation data training and prediction based on graph neural network (GNS) model framework were successfully deployed and implemented. In response to the poor learning performance of GNS in 3D SPH data, physical constraints such as edge features and mass conservation constraints were introduced to improve the accuracy of the model. After optimization, the improved GNS model exhibits excellent generalization ability, is suitable for various particle algorithms, and shows better computational accuracy and stability than the GNS model based on PyTorch in the Paddle framework.

## Installation

Paddle_GNS uses [paddlepaddle-gpu](https://www.paddlepaddle.org.cn/install) and [CUDA 11.8](https://developer.nvidia.com/cuda-downloads). These packages have specific requirements, please see [paddlepaddle installation]((https://www.paddlepaddle.org.cn/install) for details. 

> Paddle CPU installation on Linux

```shell
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
> Paddle GPU installation on Linux

```shell
python -m pip install paddlepaddle-gpu==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
For more environmental requirements, please read the 'requirements.txt'.

After installation, you can use `python` to enter the python interpreter, enter import paddle, and then enter `paddle.utls.run_check()`
If `PaddlePaddle is installed successfully!`, You have successfully installed it.

You can use the WaterDrop dataset to check if your gns code is working correctly.
To test on the small waterdroplet sample:

```
python train.py --data_path=.../datasets/WaterDrop/dataset/ --output_path=../datasets/WaterDrop/models/ -ntraining_steps=10
```
## Run
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

![Sand rollout](GNS-Paddle/gif/fns_github1.gif)
> GNS prediction of Sand rollout after training for 2 million steps.

In mesh-based domain, the renderer writes `.gif` animation.

![Fluid flow rollout](docs/img/meshnet.gif)
> Meshnet GNS prediction of cylinder flow after training for 1 million steps.

## Datasets
### Particulate domain:
We use the numpy `.npz` format for storing positional data for GNS training.  The `.npz` format includes a list of tuples of arbitrary length where each tuple corresponds to a differenet training trajectory and is of the form `(position, particle_type)`.  The data loader provides `INPUT_SEQUENCE_LENGTH` positions, set equal to six by default, to provide the GNS with the last `INPUT_SEQUENCE_LENGTH` minus one positions as input to predict the position at the next time step.  The `position` is a 3-D tensor of shape `(n_time_steps, n_particles, n_dimensions)` and `particle_type` is a 1-D tensor of shape `(n_particles)`.  

The dataset contains:

* Metadata file with dataset information `(sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...)`:

```
{
  "bounds": [[0.1, 0.9], [0.1, 0.9]], 
  "sequence_length": 320, 
  "default_connectivity_radius": 0.015, 
  "dim": 2, 
  "dt": 0.0025, 
  "vel_mean": [5.123277536458455e-06, -0.0009965205918140803], 
  "vel_std": [0.0021978993231675805, 0.0026653552458701774], 
  "acc_mean": [5.237611158734309e-07, 2.3633027988858656e-07], 
  "acc_std": [0.0002582944917306106, 0.00029554531667679154]
}
```
* npz containing data for all trajectories `(particle types, positions, global context, ...)`:

### Inspiration
PyTorch version of Graph Network Simulator are based on:
* https://github.com/geoelements/gns/tree/main

