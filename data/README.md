# File Description
`./data/` directory stores the `.npy` files of barrel cortex anatomical data required for building the model, the `.h5` files of the whisker sweep dataset, and the trained `.pth` parameter files. The following is the detailed description of these files.

* `Exc_ThtoAll_prob.py`: The connection probabilities from thalamic neurons to various neuronal subtypes in the barrel cortex, and all projections from the thalamus to the barrel cortex are excitatory. 
* `Exc_AlltoAll_prob.py`: The connection probability of excitatory neurons within the barrel cortex. 
* `Inh_AlltoAll_prob.py`: The connection probability of inhibitory neurons within the barrel cortex.
* `snn_train3.h5`: Whisker sweep training set of three types. 
* `snn_test3.h5`: Whisker sweep test set of three types.
* `RSNN_bfd_0.0xb_0.0xw.pth`: Model parameter files trained on the spiking whisker sweep dataset.

The `.h5` and `.pth` files can be downloaded [here](https://pan.quark.cn/s/887585c2ffcd).
