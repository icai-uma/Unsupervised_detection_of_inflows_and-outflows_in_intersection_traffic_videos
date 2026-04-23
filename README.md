# Unsupervised detection of inflows and outflows in intersection traffic videos

Repository for the code supporting of the paper titled **Unsupervised detection of inflows and outflows in intersection traffic videos** (DOI: []())

## Citation
Please use the following citation if you use our data or code:
````
@article{

}
````




## Requirements
The code is tested with the following requirements:
- Python 3.11.10
- Pytorch 2.5.1
- CUDA 12.4

Experiments were conducted with a GPU NVIDIA GeForce RTX 3090 Ti 24GB.

The ``src/main.py`` file has the code to run the YOLOv5 model (``src/yolov5x6.pt``) to detect and track vehicles using the [input/]() folder. In addition, versions with added noise are generated to validate the robustness of the approach. Finally, the code evaluates the tracked points using clustering methods and calculates performance metrics.

## Abstract 
As traffic cameras become prevalent, the automatic analysis of traffic scenes presents new opportunities and challenges. Advances in deep learning allow for automated characterization of traffic in such videos. This work aims to understand traffic flow without human supervision, focusing on the localization of road intersections. For this purpose, a three-stage method is proposed that uses a deep neural network for vehicle detection, an object tracker to recover vehicle trajectories, and unsupervised machine learning to detect potential incoming and outgoing traffic flows. The approach has been tested on a variety of real and synthetic videos, with satisfactory results across different camera positions, traffic patterns, and weather conditions. As a key part of the methodology, five options for clustering starting and ending track points were tested. These options included a basic strategy based on predefined spatially localized clusters, and the K-means algorithm with two methods to determine the optimal number of clusters: the Elbow method and the Silhouette score. Additionally, Mean Shift and the Density-Based Spatial Clustering of Applications with Noise were evaluated. An exhaustive analysis of the proposed clustering methods was conducted, including runtime at each stage, performance metrics , and the addition of noise to simulate tracker failures. The results demonstrated the feasibility of the proposed methodology and concluded that Mean Shift is the most suitable clustering method due to its balance of high performance, low runtime, and stable behaviour against abnormal trajectory points.

![Methodology pipeline](images/pipeline.pdf)