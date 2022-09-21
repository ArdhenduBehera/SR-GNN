## SR-GNN: Spatial Relation-aware Graph Neural Network for Fine-Grained Image Categorization 
**Asish Bera, Zachary Wharton, Yonghuai Liu, Nik Bessis, and Ardhendu Behera**<br/>
**Department of Computer Science, Edge Hill University, United Kingdom**

### Abstract
Over the past few years, a significant progress has been made in deep convolutional neural networks (CNNs)-based image recognition. This is mainly due to the 
strong ability of such networks in mining discriminative object pose and parts information from texture and shape. This is often inappropriate for 
fine-grained visual classification (FGVC) since it exhibits high intra-class and low inter-class variances due to occlusions, deformation, illuminations, 
etc. Thus, an expressive feature representation describing global structural information is a key to characterize an object/ scene. To this end, we 
propose a method that effectively captures subtle changes by aggregating context-aware features from most relevant image-regions and their importance 
in discriminating fine-grained categories avoiding the bounding-box and/or distinguishable part annotations. Our approach is inspired by the recent 
advancement in self-attention and graph neural networks (GNNs) approaches to include a simple yet effective relation-aware feature transformation and 
its refinement using a context-aware attention mechanism to boost the discriminability of the transformed feature in an end-to-end learning process. Our 
model is evaluated on eight benchmark datasets consisting of fine-grained objects and human-object interactions. It outperforms the state-of-the-art 
approaches by a significant margin in recognition accuracy.

### Spatial Relation-aware Graph Neural Network (SR-GNN)
It is a simple yet effective connection between the image space and feature space to discriminate subtle variances in FGVC. These connections are captured 
using a novel relationaware feature transformation and its refinement via attentional context modeling. High-level CNN features are pooled using geometrically 
constrained regions of various sizes and positions. These pooled features are transformed using a GNN that captures the visual-spatial relationships via 
propagating information between regions represented as the nodes of a connected graph to enhance the disriminative power of features.
![Image](diagram.jpg)
**High-level illustration of our model (left). The detailed architecture of our novel CAP (right).**

![Image](diagram2.jpg)
**Learning pixel-level relationships from the convolutional feature map of size _W x H x C_. b) CAP using integral regions to capture both self and neighborhood contextual information. c) Encapsulating spatial structure of the integral regions using an LSTM. d) Classification by learnable aggregation of hidden states of the LSTM.**

### Paper and Supplementary Information
Extended version of the accepted paper in [ArXiv](https://arxiv.org/abs/2101.06635).

[Supplementary Document](AAAI_Supplementary.pdf)

[Source code](https://github.com/ArdhenduBehera/cap)

### Bibtex
```markdown
@inproceedings{behera2021context,
  title={Context-aware Attentional Pooling (CAP) for Fine-grained Visual Classification},
  author={Behera, Ardhendu and Wharton, Zachary and Hewage, Pradeep and Bera, Asish},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021},
  organization={AAAI}
}
```

### Acknowledgements

This research was supported by the UKIERI (CHARM) under grant DST UKIERI-2018-19-10. The GPU is kindly donated by the NVIDIA Corporation.
