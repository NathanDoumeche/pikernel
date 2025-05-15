# Pikernel

**Pikernel** is a Python package for constructing **physics-informed kernels** as introduced in the paper  
[**Physics-Informed Kernel Learning**](https://arxiv.org/pdf/2409.13786) (2025) by Nathan Doumèche, Francis Bach, Claire Boyer,  and Gérard Biau.

It provides an easy-to-use framework for implementing physics-informed kernels in **1D and 2D**, for a wide class of **ODEs and PDEs with constant coefficients**.  
The package supports both **CPU and GPU** execution and automatically leverages available hardware for optimal performance.



##  Features

- Build kernels tailored to your differential equation constraints  
- Works with any linear ODE/PDE with constant coefficients in 1D or 2D  
- Compatible with NumPy and PyTorch backends  
- GPU support via PyTorch for accelerated computation  



## Installation

You can install the package via pip:

```bash
pip install pikernel
```

## Resources

* **Tutorial:** [https://github.com/claireBoyer/tutorial-piml](https://github.com/claireBoyer/tutorial-piml)
* **Source code:** [https://github.com/NathanDoumeche/pikernel](https://github.com/NathanDoumeche/pikernel)
* **Bug reports:** [https://github.com/NathanDoumeche/pikernel/issues](https://github.com/NathanDoumeche/pikernel/issues)



## Citation
To cite this package:

    @article{doumèche2024physicsinformedkernellearning,
      title={Physics-informed kernel learning},
      author={Nathan Doumèche and Francis Bach and Gérard Biau and Claire Boyer},
      journal={arXiv:2409.13786},
      volume={},
      pages={},
      year={2024},
      publisher={}
    }