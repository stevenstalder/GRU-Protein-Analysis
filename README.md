
[![MIT License][license-shield]][license-url]


<br />
<p align="center">
  <a href="https://github.com/stevenstalder/GRU-Protein-Analysis">
    <img src="https://miro.medium.com/max/680/1*b1zFnLOeM36CBsF4lA7pvQ.gif" alt="Logo" width="250">
  </a>

  <h3 align="center">Examining Gated Recurrent Network Architectures in Protein Analysis</h3>

  <p align="center">
    Semester Project for the Deep Learning 2020 Lecture at ETH Zürich
    <br />
    <a href="https://github.com/stevenstalder/GRU-Protein-Analysis/tree/main/src"><strong>Explore the Project »</strong></a>
    <br />
    <br />
    <a href="https://github.com/stevenstalder/GRU-Protein-Analysis/issues">Report Bug</a>
  </p>
</p>

## About the Project

A fundamental task in understanding evolution and fighting diseases is to gain knowledge about proteins and their
assembly. Proteins are chains of amino acids formed into a specific structure [Yanofsky et al., 1964]. Due to recent
research in deep learning and particularly the Natural Language Processing (NLP) area, certain entities of these structures
– e.g. where a chain meets again or how it is twisted – can be predicted more accurately. To achieve more
comparable results, Rao et al. [2019] produced five standardized datasets and specific tasks in the area of protein
prediction. The one tasks we want to take on is the Secondary Structure (SS) prediction.

## The Code 

## Table of Contents
* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Images](#images)
* [Getting Started](#getting-started)
  * [Download missing data](#download-missing-data)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproduce our results](#reproduce-our-results)
    * [Train and predict results](#train-and-predict-results)
    * [Predict using pretrained models](#predict-using-pretrained-models)
    * [Find the results](#find-the-results)

## About The Project
This repository contains the source code for the graded semester project for the [Computational Intelligence Lab 2020 lecture](http://da.inf.ethz.ch/teaching/2020/CIL/) at ETH Zurich.
Please follow the instructions below to get started and reproduce our results.
Read the [paper](https://github.com/winbergs/CILlitbang/blob/master/report.pdf) for more information about our experiments and design decisions.

Our final model is a dilated U-Net with transposed convolutional layers and has the following structure:

<div align="center">
<img src="plots/diagrams/unet_dilated_v2_transposed.png" alt="U-Net Dilated v2 Transposed" width="50%"/>
</div>

Furthermore, we constructed the following dedicated post-processing pipeline to further refine and cleanup the results.

<div align="center">
<img src="plots/diagrams/post_processing_pipeline.png" alt="Dedicated Post-Processing Pipeline" width="90%"/>
</div>

## Folder Structure
```
├── README.md
├── environment.yml                                   - YAML file for conda environment (only for CPU usage)
├── environment_gpu.yml                               - YAML file for GPU usage working on ETH's Leonhard cluster (recommended)
├── tape_data                                             
│   ├── test                                          - CB513 testing data (will be filled after first run)
│   ├── training                                      - TAPE training data (will be filled after first run)
│   └── validation                                    - TAPE validation data (will be filled after first run)
└── src
    ├── __init__.py                                   - Imports for main.py
    ├── main.py                                       - Main script to execute
    ├── data
    │   ├── dataimport.py                             - Dataloader imports for main.py
    │   ├── dataloader.py                             - Defines Dataloaders
    │   └── download.py                               - Downloads data if not present locally
    ├── models
    │   ├── classifier_autoregressive.py              - Autoregressive classifier
    │   ├── classifier_CNN.py                         - CNN classifier
    │   ├── encoder_GRU.py                            - GRU encoder
    │   ├── encoder_LSTM.py                           - LSTM encoder
    │   ├── model_GRU_autoregressive.py               - Full model with GRU encoder and autoregressive classifier
    │   ├── model_GRU_CNN.py                          - Full model with GRU encoder and CNN classifier
    │   └── model_LSTM_CNN.py                         - Full model with LSTM encoder and CNN classifer
    └── utils
        ├── argparser.py                              - Parser for command line arguments
        ├── accuracy.py                               - Custom accuracy metric
        └── tokenizer.py                              - TAPE Tokenizers
```

## Contact

Steven Stalder  - staldest@student.ethz.ch <br>
Michael Sommer  - sommemic@student.ethz.ch <br>
Donal Naylor  - dnaylor@student.ethz.ch <br>
Lukas Klein  - luklein@student.ethz.ch

<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square

[license-url]: https://github.com/


