# Visually-Grounded VideoQA for Mitigating Language Bias

This repository contains the source code for the CS5342 project on addressing language bias in video question and answering (VideoQA) using visual-grounding with self-supervision. The project code is extended from the original paper on [Self-Chained Image-Language Model for Video Localization and Question Answering (SeViLA)](https://github.com/Yui010206/SeViLA), which is in turn developed by using the repository [LAVIS - A Library for Language-Vision Intelligence](https://github.com/salesforce/LAVIS) by Salesforce as a base. As such, the primary logic for training, evaluation data loading and pre-processing as well as model initialization and loading are heavily borrowed from LAVIS.

## File Structure
The file structure of the codebase is as follows, focusing on files containing logic that are relevant to the project:

```
VGVideoQA
├── README.md
├── checkpoints
│   ├── sevila_finetuned.pth
│   └── sevila_pretrained.pth
├── datasets
│   └── nextqa
│       ├── map_vid_vidorID.json
│       ├── processed
│       ├── raw
│       └── videos
│               ├── ...
│               ├── 6994417868.mp4
│               └── ...
├── environment.yml
├── lavis
│   ├── configs
│   │   ├── datasets
│   │   │   └── nextqa
│   │   │       └── defaults.yaml
│   │   └── models
│   ├── datasets
│   │   ├── datasets
│   │   │   ├── video_vqa_datasets.py
│   │   │   └── ...
│   │   └── ...
│   ├── models
│   │   ├── sevila_models
│   │   │   └── sevila.py
│   │   └── ...
│   ├── processors
│   │   ├── blip_processors.py
│   │   └── ...
│   ├── projects
│   │   ├── sevila
│   │   │   ├── eval
│   │   │   │   └── nextqa.yaml
│   │   │   └── train
│   │   │       └── nextqa.yaml
│   │   └── ...
│   └── tasks
│       ├── vqa.py
│       └── ...
├── outputs
│   └── nextqa
│       ├── pretrained
│       │       └── evaluate.txt
│       ├── finetuned
│       │       └── evaluate.txt
│       └── refined
│               └── evaluate.txt
├── main.py
└── scripts
    └── sevila
        ├── evaluate.sh
        ├── finetune.sh
        └── refine.sh
```

We provide a brief overview of the essential files necessary for running our experiments in the project below.

### Model Architecture: 
`lavis/models/sevila_models/sevila.py` contains the VideoQA model used in the project. It is refactored from the original SeViLA repository model code in preparation for extention in the project, while retaining the model computation logic from the original repository for the assignment.

### Data Loading and Pre-processing:
The primary dataset used in our experiments is [NExT-QA](https://github.com/doc-doc/NExT-QA), which is contained under the `datasets/nextqa` directory.`lavis/datasets/datasets/video_vqa_datasets.py` contains the `NExTQADataset` class from PyTorch encapsulating dataset iteration logic for the NExT-QA dataset during training and inference, while `BlipVideoTrainProcessor`, `BlipVideoEvalProcessor` and `BlipQuestionProcessor` in `lavis/processors/blip_processors.py` handles data pre-processing for both video and text inputs respectively. The data loading and preparation logic in these files are largely adapted from SeViLa with minimal change, due to the simple and already clean code present in the original files.

### Training and Evaluation

`main.py` is the primary driver function for training and evaluation, which is executed by `scripts/sevila/finetune.sh`, `scripts/sevila/refine.sh` and `scripts/sevila/evaluate.sh` driver scripts. The configurations for both training and evaluation can be found under `lavis/projects/sevila/train/nextqa.yaml` and `lavis/projects/sevila/eval/nextqa.yaml` respectively. The configuration defined in both files are primarily used as per the SeViLa repository, and modified in the 3 driver scripts as needed.

The training logs for finetuning and localizer refinement can be found under `outputs/nextqa/finetuned/val_results.txt` and `outputs/nextqa/refined/val_results.txt` respectively. The predictions on the test set is also provided under `val_predictions.json` in the same subdirectory. 

## Setup

### Install dependencies
To install the packages required to run the repository, we use the Anaconda package manager. Ensure that Anaconda is installed on your machine, and run the following commands to download the necessary packages:
```
conda env create -f environment.yml
conda activate sevila
```
**NOTE**: The `decord` library may not be available if you are using Mac M1/M2 processors. If you're using Mac ARM machines, please change `decord==0.6.0` in the `environment.yml` file to `eva-decord` instead.

### Download pre-trained models

We initialize from pre-trained model weights `checkpoints/sevila_pretrained.pth` using moment retrieval pre-training, as performed by the authors in SeViLA for the first stage of training, answerer fine-tuning. The weights from answerer fine-tuning is `checkpoints/sevila_finetuned.pth` is then used for localizer refinement in the second stage of training. Both checkpoints can be found in this [link](https://drive.google.com/drive/folders/1l8Sll64JhbQxo2ZgiPLS7Mt754nANMAj?usp=drive_link), and should be placed under the `checkpoints` folder.

## Data Preparation

We train and evaluate the model on the NExT-QA dataset, which is present under the `datasets/nextqa` folder. The QA pairs and video mappings for each example are present in the JSON files under each split respectively. However, due to the large size of the videos, we omit them in the submissions. Please download the raw videos from the NExTVideo.zip in this [link](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view?usp=drive_link), and extract the contents under `datasets` as per the file structure shown above.

## Training and Inference

All experiments for training were run on compute cluster on the National SuperComputer Centre (NSCC) using GPUs with VRAM of 40GB. It is recommended to have at least GPUs of at least 24GB VRAM to minimally run the code using batch size of 1.

### Answerer finetuning
To perform the first stage of training for fine-tuning the answerer (QA) module, run the following command:
```
./scripts/sevila/finetune.sh
```

### Localizer refinement
To perform the second stage of training for refining the localizer (visual grounding) module, run the following command:
```
./scripts/sevila/refine.sh
```

### Evaluation
To run evaluation on the test data, execute the following command:
```
./scripts/sevila/evaluate.sh
```
The results of the implemented SeViLA approach can be found under `outputs/` for pre-trained, answerer fine-tuned and localizer refined respectively in their respective `val_results.txt` files.