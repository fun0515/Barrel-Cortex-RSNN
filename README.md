# BarrelCortexRSNN  
Train and evaluate a recurrent spiking neural network that is biologically constrained by the mouse barrel cortex. The paper is [here](https://markdown.com.cn](https://openreview.net/forum?id=UvfI4grcM7&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions))).
# Dependencies
Python==3.8, h5py==3.10.0, pytorch==2.1.1. 
Install the dependencies from the requirements.txt file:   
`pip install -r requirements.txt`
# File Description
* `/data/`:  Three `.npy` files that store the anatomical information of the barrel cortex, two `.h5` files of datasets, and some additional trained `.pth` files. The `.h5` and `.pth` files can be downloaded from [here](https://pan.quark.cn/s/6d41efaccd6c).
* `whisker_dataset.py`:  Load the whisker sweep dataset, including the real-valued form and the spiking-based form.
* `RSNN_bfd_SpikingBased.py`: Train the model on the spiking whisker sweep dataset and plot the neural firing selectivity.
* `RSNN_bfd_RealValued.py`: Train the model on the real-valued whisker sweep dataset.
* `RSNN_bfd_whisker_deprivation.py`: Test the performance of the trained model in response to the whisker deprivation experiment.
* `SparseLinear.py`: Sample the connections between neural subtypes based on projection intensities.
* `utils.py`: Plot the neural raster, dynamic gradient, weighted degree distribution, and CV measure.
