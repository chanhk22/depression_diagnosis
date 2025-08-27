# Depression Diagnosis (DAIC-WOZ, E-DAIC, D-VLOG)

This repository implements a **multimodal depression detection pipeline** using three data

+ **DAIC-WOZ** : CLNF/OpenFace + COVAREP + audio + transcripts
+ **E-DAIC** : audio + transcripts + egemaps + CNN visual features
+ **D-VLOG** : In-the-wild dataset with acoustic + visual '.npy' features

Model supports **teacher-student training** with **domain adaption** and **explainability**.

# Data Preparation

## Download the datasets

+ D-VLOG datasets' features are extracted by authors and are publicly available by [here](https://sites.google.com/view/jeewoo-yoon/dataset). Fill out the data request form and contact the author by email.

+ DAIC-WOZ and E-DAIC datasets are available upon request [here](https://dcapswoz.ict.usc.edu)


## Setup

### 1. clone repo
bash
git clone https://github.com/chanhk22/depression_diagnosis.git
cd depression_diagnosis

If using PASE+ , install from tis official repo:

git clone https://github.com/santi-pdp/pase.git

## Pipeline

### Step 1. Preprocess audio
+ Cut Ellie's speech
+ Masks silence (VAD)
```
bash scripts/1_preprocess_audio.sh
```

### step 2. Extract features
+ Re-extract eGeMAPS (25 LLD) for DAIC/E-DAIC audio files.
+ Extract PASE+ (optional)
```
bash scripts/2_extract_features.sh
```

### step 3. Build cache
+ Window segmentation ( fixed length, stride)
+ Saves .npz files + *_index.csv
```
bash scripts/3_build_cache.sh
```

### step 4. Train Teacher
+ Teacher model (audio + privileged modalities)
```
bash scripts/4_train_teacher.sh
```

### step 5. Train Student
+ Student model (audio only or audio + landmarks)
+ Knowledge Distillation from teacher
```
bash scripts/5_train_student.sh
```

### step 6. Explainability
+ Compute SHAP values
+ Plot top-k important features (e.g. landmarks, LLDs)
```
bash scripts/6_run_shap.sh
```

## Features
+ Audio : eGeMAPS (25 LLD), MFCC , PASE+(optional)
+ Visual : CLNF/OpenFace landmarks, CNN(VGG16/DenseNet201) features
+ Fusion : cross-attention + domain adaption (MMD/GRL)
+ Explainability : SHAP + landmark heatmaps

## Losses
+ Knowledge Distillation (KD)
+ Maximum Mean Discrepancy (MMD) for domain adaption
+ Multitask Loss (classification + regression)

## Results


## 📜 Notes
+ contain original datasets in data_raw/
+ data/processed , data/cache generates during preprocessing
+ set paths and hyperparameter in configs/


## 📜 Citation
If you use this pipeline, please cite DAIC-WOZ, E-DAIC, D-VLOG datasets 