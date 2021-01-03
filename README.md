
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

## Table of Contents
* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Protein Data](#protein-data)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproducing our results](#reproducing-our-results)
* [Contact](#contact)

## About the Project

A fundamental task in understanding evolution and fighting diseases is to gain knowledge about proteins and their
assembly. Proteins are chains of amino acids formed into a specific structure [[Yanofsky et al., 1964]](https://doi.org/10.1126/science.146.3651.1593). Due to recent
research in deep learning and particularly the Natural Language Processing (NLP) area, certain entities of these structures
– e.g. where a chain meets again or how it is twisted – can be predicted more accurately. To achieve more
comparable results, [[Rao et al. 2019]](https://www.biorxiv.org/content/early/2019/06/20/676825) produced five standardized datasets and specific tasks in the area of protein
prediction. In this project we take on the Secondary Structure (SS) prediction.

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

## Protein Data

We are using the predefined training, validation and test data as provided by the [TAPE group](https://github.com/songlab-cal/tape#data). As defined in TAPE's SS prediction task, we are mapping each amino acid of a protein sequence to one of three labels. Accuracy is reported on a per-amino acid basis on the [CB513 test dataset](https://onlinelibrary.wiley.com/doi/full/10.1002/%28SICI%291097-0134%2819990301%2934%3A4%3C508%3A%3AAID-PROT10%3E3.0.CO%3B2-4).

## Usage

All essential libraries for the execution of the code are provided in the environment_gpu.yml file from which a new conda environment can be created.

### Run the code

Once the virtual environment is activated you can run the code as follows:

- Go into the `src` directory.
  ```sh
  cd src/
  ```
- Run the program with any number of custom arguments as defined in src/utils/argparser.py. For example:
  ```sh
  python main.py --encoder_type="gru" --learning_rate=0.01
  ```
- If you want to run our code on ETH's Leonhard cluster, submit the same job as above as follows:
  ```sh
  bsub -W 24:00 -R "rusage[ngpus_excl_p=1,mem=8192]" "python main.py --encoder_type="gru" --learning_rate=0.01"
  ```

### Reproducing our results

For reproducibility, we have fixed a random seed which you can leave at its default value. However, we have performed a reasonable amount of hyperparameter tuning for all of our models and if you wish to reproduce our best results, we recommend running our code as follows (only showing arguments for which the default value needs change):

- For the unidirectional GRU model:
  ```sh
  python main.py --cnn_dilated=true --cnn_hidden_size=2048 --learning_rate=0.01
  ```
- For the unidirectional LSTM model:
  ```sh
  python main.py --cnn_dilated=true --cnn_hidden_size=2048 --learning_rate=0.01 --encoder_type="lstm"
  ```
- For the bidirectional GRU model:
  ```sh
  python main.py --cnn_dilated=true --enc_bidirectional=true --learning_rate=0.1
  ```
- For the bidirectional LSTM model:
  ```sh
  python main.py --cnn_dilated=true --enc_bidirectional=true --learning_rate=0.1 --encoder_type="lstm"
  ```

## Contact

Steven Stalder  - staldest@student.ethz.ch <br>
Michael Sommer  - sommemic@student.ethz.ch <br>
Donal Naylor  - dnaylor@student.ethz.ch <br>
Lukas Klein  - luklein@student.ethz.ch

<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square

[license-url]: https://github.com/


