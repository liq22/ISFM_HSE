# Heterogeneous Signal Embedding (HSE) Module

## Overview
This repository contains the official implementation of "HSE: A plug-and-play module for unified fault diagnosis foundation models". HSE serves as a fundamental component for processing heterogeneous industrial signals within the [ISFM framework (Private stage)](https://github.com/liq22/ISFM).


## Citation
```
If you find our work useful, please consider citing:
[@article{LI2025103277,
title = {HSE: A plug-and-play module for unified fault diagnosis foundation models},
journal = {Information Fusion},
volume = {123},
pages = {103277},
year = {2025},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103277},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525003501},
author = {Qi Li and Bojian Chen and Qitong Chen and Xuan Li and Zhaoye Qin and Fulei Chu},
keywords = {Intelligent fault diagnosis, Heterogeneous signal embedding, Signal fusion, Signal foundation model, Plug-and-play module},
abstract = {Intelligent Fault Diagnosis (IFD) plays a crucial role in industrial applications, where developing foundation models analogous to ChatGPT for comprehensive fault diagnosis remains a significant challenge. Current IFD methodologies are constrained by their inability to construct unified models capable of processing heterogeneous signal types, varying sampling rates, and diverse signal lengths across different equipment. To address these limitations, we propose a novel Heterogeneous Signal Embedding (HSE) module that projects heterogeneous signals into a unified signal space, offering seamless integration with existing IFD architectures as a plug-and-play solution. The HSE framework comprises two primary components: the Temporal-Aware Patching (TAP) module for embedding heterogeneous signals into a unified space, and the Cross-Dimensional Patch Fusion (CDPF) module for fusing embedded signals with temporal information into unified representations. We validate the efficacy of HSE through two comprehensive case studies: a simulation signal dataset and three distinct bearing datasets with heterogeneous features. Our experimental results demonstrate that HSE significantly enhances traditional fault diagnosis models, improving both diagnostic accuracy and generalization capability. While conventional approaches necessitate separate models for specific signal types, sampling frequencies, and signal lengths, HSE-enabled architectures successfully learn unified representations across diverse signal. The results from bearing fault diagnosis applications confirm substantial improvements in both diagnostic precision and cross-dataset generalization. As a pioneering contribution toward IFD foundation models, the proposed HSE framework establishes a fundamental architecture for advancing unified fault diagnosis systems.}
}]
```

## Contact
- Qi Li - [Personal Website](https://liq22.github.io/)
