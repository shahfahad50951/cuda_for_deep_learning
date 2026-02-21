# CUDA FOR DEEP LEARNING

## GOALS
- Implement most frequently used operations in deep learning as CUDA kernels 
- Implement Deep Learning Models from highest level of abstraction (pytorch) down to the lowest level (C/CUDA) with each level adding more and more details to the implementation
- Start with naive implementation and then improve the implementation to match / beat the state of the art (pytorch) implementations

## TAKEAWAYS
- A complete understanding of how deep learning models implemented in pytorch are mapped to lower level concepts like CUDA kernel
- Understanding of the purpose of the autograd engine of pytorch and minimal replication of its behaviour required for our implementation
- Optimization concepts and step by step process of how a model (and CUDA kernels) are optimized to get the best performance