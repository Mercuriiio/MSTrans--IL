## MSTrans-MIL: Multiple Instance Learning-based Multi-scale Transformer for Cancer Diagnosis and Prognosis

<!--**Summary:** We propose a Multiple Instance Learning-based Multi-scale Transformer (MSTrans-MIL) for cancer survival analysis and grade prediction. We investigate the application of the self-attention mechanism to multi-instance feature sampling for WSI to reduce the input redundancy of the model and improve its scalability. A multi-scale feature extractor based on self-supervised learning is also proposed to perform Convolution-Transformer based contextual feature extraction on WSI. The pathology instances at three scales comprehensively reflect the properties of the tumor microenvironment, structure of the tumor cells and polymorphism of the nuclei. Meanwhile, we proposed a new multi-scale feature aggregation module based on the gating mechanism. The utilization of sparse Transformer further reduces the redundancy of the features across scales.-->

![image](https://github.com/Mercuriiio/MSTrans-MIL/blob/main/model.jpg)

### Prerequisites
- NVIDIA GPU (Tested on Nvidia V100)
- CUDA + cuDNN (Tested on CUDA 10.2 and cuDNN 7.8.)
- torch>=1.1.0

## Code Base Structure
The code base structure is explained below: 
- **train.py**: Cross-validation script for training and testing multi-scale network.
- **Model.py**: Contains PyTorch model definitions for multi-scale network.
- **NegativeLogLikelihood.py**: The survival analysis loss function used in the method.
- **Loader.py**: Contains definitions for collating, data preprocessing, etc...

## Data Preprocess

### Obtain the Patches
Raw WSI images can be obtained from [TCGA](https://portal.gdc.cancer.gov/). Use histolab to slice the WSI to get the patches (Requires [histolab](https://github.com/histolab/histolab) to be installed).

### Patch Feature Extraction
Self-supervised learning of patches using the [UFormer network](https://github.com/ZhendongWang6/Uformer). [data](https://github.com/Mercuriiio/MSTrans-MIL/tree/main/data/gbmlgg) shows some of the multi-scale data collected. Please note that due to the automated ease of data processing, we only provide some of the train data for reference.

## Training and Evaluation

Train and test model with the following code. Validation and testing are included in [train.py](https://github.com/Mercuriiio/MSTrans-MIL/blob/main/train.py).

```
python train.py
```

The KM curve plots are implemented in R language. We first obtained and sorted the survival prediction output for each sample, and then categorized them into risk groups based on different quantiles (e.g., tertiles of TCGA-GBMLGG).
