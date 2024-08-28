# Graph Network Simulator (GNS) Paddle framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/geoelements/gns/main/license.md)

The development and operation of SPH algorithm are realized on the AI Studio of PaddlePaddle, and the flow field data of 3D complex calculation examples are obtained. The problem of missing radius_graph function in Paddle PGL module was solved by customizing model functions, and simulation data training and prediction based on graph neural network (GNS) model framework were successfully deployed and implemented. In response to the poor learning performance of GNS in 3D SPH data, physical constraints such as edge features and mass conservation constraints were introduced to improve the accuracy of the model. After optimization, the improved GNS model exhibits excellent generalization ability, is suitable for various particle algorithms, and shows better computational accuracy and stability than the GNS model based on PyTorch in the Paddle framework.## Run GNS

