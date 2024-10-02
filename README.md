### Automation Lab, Sungkyunkwan University

# ETSS-06: Anomaly Detection

This is the official repository of 

**OpenAnomaly: An Open Source Implementation of Anomaly Detection Methods.**

## 1. Setup
### 1.1. Using environment.yml
```bash
conda env create -f environment.yml
conda activate anomaly
```

### 1.2. Using requirements.txt
```bash
conda create --name anomaly python=3.10.13
conda activate anomaly
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## 2. Dataset Preparation
### 2.1. Industry Anomaly Detection Datasets
For MVTec dataset, please download the images from this [link](https://www.mvtec.com/company/research/datasets/mvtec-ad)

For BTAD dataset, please download the images from this [repository](https://github.com/pankajmishra000/VT-ADL)

For VisA dataset, please download the images from this [repository](https://github.com/amazon-science/spot-diff)

For MVTec LOCO dataset, please download the images from this [link](https://www.mvtec.com/company/research/datasets/mvtec-loco)

For MPDD dataset, please download the images from this [repository](https://github.com/stepanje/MPDD)

For DTD dataset, please download the images from this [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

For DAGM dataset, please download the images from this [link](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)

For WFDD dataset, please download the images from this [repo](https://github.com/cqylunlun/GLASS)

For Real-IAD dataset, please download the images from this [link](https://huggingface.co/datasets/Real-IAD/Real-IAD)

For MVTec3D dataset, please download the images from this [link](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)

For Eyecandies dataset, please download the images from this [link](https://eyecan-ai.github.io/eyecandies/download)

For MadSim dataset, please download the images from this [repository](https://github.com/EricLee0224/PAD)

For ShapNet dataset, please download the images from this [repository](https://github.com/Chopper-233/Anomaly-ShapeNet)

For Real3D dataset, please download the images from this [repository](https://github.com/M-3LAB/Real3D-AD)

For PD-REAL dataset, please download the images from this [repository](https://github.com/Andy-cs008/PD-REAL)

For BrokenChairs dataset, please download the images from this [repository](https://github.com/VICO-UoE/Looking3D)

For ELPV dataset, please download the images from this [repository](https://github.com/zae-bayern/elpv-dataset)

For SDD dataset, please download the images from this [link](https://www.vicos.si/resources/kolektorsdd/)

For AITEX dataset, please download the images from this [link](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database/data)

For Brain MRI dataset, please download the images from this [link](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?select=yes)

For Head CT dataset, please download the images from this [link](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)

### 2.2. Unsupervised Anomaly Detection Datasets
For UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets, please download the dataset from this [repository](https://github.com/aseuteurideu/STEAL)

### 2.3. Video Anomaly Detection Datasets
For ShanghaiTech dataset, please download the extracted features from this [repository](https://github.com/tianyu0207/RTFM).

For UCF-Crime dataset, please download the extracted features from this [repository](https://github.com/carolchenyx/MGFN.) or [repo](https://github.com/Roc-Ng/DeepMIL).

For UCF-Traffic dataset, please download the extracted features from this [repository](https://github.com/VFWm614/TA-NET)

For XD-Violence dataset, please download the extracted features from this [link](https://roc-ng.github.io/XD-Violence/).

For TAD dataset, please download the extracted features from this [repository](https://github.com/ktr-hubrt/WSAL)

### 2.4. Dashcam Traffic Anomaly Detection Datasets
For ROL and DoTA datasets, please download the images from this [repository](https://github.com/monjurulkarim/risky_object)

For CCD, DAD, and A3D datasets, please download the images from this [repository](https://github.com/Cogito2012/UString)

## 3. Usage
### 3.1.1. Supported Models for 2D Industrial Anomaly Detection
| Models    | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD |
|-----------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|
| CutPaste  | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| SSAPS     | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| DRAEM     | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| SPR       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| ADShift   | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| FOD       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| PatchSVDD | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| CPR       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| FAIR      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| GLASS     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      |
| SimpleNet | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      |
| SegAD     | :x:                | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      |
| CDO       | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      |
| SPD       | :x:                | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      |
 
### 3.1.2. Supported Models for 3D Industrial Anomaly Detection
| Models       | MVTec3D            | Eyecandies         | MadSim             | ShapNet | Real3D | BrokenChairs | PD-Real |
|--------------|--------------------|--------------------|--------------------|---------|--------|--------------|---------|
| BTF          | :heavy_check_mark: | :x:                | :x:                | :x:     | :x:    | :x:          | :x:     |
| CFM          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:     | :x:    | :x:          | :x:     |
| CPMF         | :heavy_check_mark: | :x:                | :x:                | :x:     | :x:    | :x:          | :x:     |
| Looking3D    | :x:                | :x:                | :x:                | :x:     | :x:    | :x:          | :x:     |
| M3DM         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:     | :x:    | :x:          | :x:     |
| PAD          | :x:                | :x:                | :heavy_check_mark: | :x:     | :x:    | :x:          | :x:     |
| Shape-Guided | :heavy_check_mark: | :x:                | :x:                | :x:     | :x:    | :x:          | :x:     |
| SplatPose    | :x:                | :x:                | :heavy_check_mark: | :x:     | :x:    | :x:          | :x:     |

### 3.1.3. Supported Models for Diffusion Industrial Anomaly Detection
| Models           | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD | MVTec3D            |
|------------------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|--------------------|
| AnoDDPM          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| D3AD             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DDAD             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiffAD           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiffusionAD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| GLAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| RealNet          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| AnomalyDiffusion | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| TransFusion      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :heavy_check_mark: |

### 3.1.4. Supported Models for Knowledge Distillation Industrial Anomaly Detection
| Models       | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|--------------|--------------------|--------------------|------|------------|------|------|----------|
| DeSTSeg      | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| DMAD         | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| EfficientAD  | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |
| IKD          | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| MemKD        | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| MixedTeacher | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| MKD          | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| RD           | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| RD++         | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| STFPM        | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| DAF          | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.5. Supported Models for Memory Bank Industrial Anomaly Detection
| Models            | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|-------------------|--------------------|--------------------|------|------------|------|------|----------|
| CFA               | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| MemSeg            | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| PaDiM             | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| PatchCore         | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| SPADE             | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.6. Supported Models for Multi-class Industrial Anomaly Detection
| Models     | MVTec              | BTAD | VisA               | MVTec LOCO | MPDD | WFDD | Real-IAD           | MVTec3D            |
|------------|--------------------|------|--------------------|------------|------|------|--------------------|--------------------|
| CRAD       | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :x:                | :x:                |
| Real-IAD   | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :heavy_check_mark: | :x:                |
| UniAD      | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :heavy_check_mark: | :x:                |
| HVQ        | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :x:                | :x:                |
| MSTAD      | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :x:                | :x:                |
| MambaAD    | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:                | :heavy_check_mark: |
| ViTAD      | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:                | :heavy_check_mark: |
| InvAD      | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:                | :heavy_check_mark: |
| Dinomaly   | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :heavy_check_mark: | :x:                |
| RLR        | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:                | :x:                |
| UniFormaly | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :x:                | :x:                |
| OneNIP     | :heavy_check_mark: | :x:  | :x:                | :x:        | :x:  | :x:  | :x:                | :x:                |
| IUF        | :heavy_check_mark: | :x:  | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:                | :x:                |

### 3.1.7. Supported Models for Noisy Industrial Anomaly Detection
| Models    | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|-----------|--------------------|--------------------|------|------------|------|------|----------|
| SoftPatch | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.8. Supported Models for Normalizing Flows Industrial Anomaly Detection
| Models      | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD | WFDD | Real-IAD |
|-------------|--------------------|--------------------|--------------------|------------|------|------|----------|
| AST         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      |
| BGAD        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      |
| CFLOW-AD    | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      |
| DifferNet   | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      |
| HGAD        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:      |
| MSFlow      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      |
| PyramidFlow | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      |

### 3.1.9. Supported Models for Out-of-Distribution Industrial Anomaly Detection
| Models    | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD | WFDD | Real-IAD |
|-----------|--------------------|--------------------|--------------------|--------------------|------|------|----------|
| ADShift   | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      |
| GeneralAD | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:  | :x:      |

### 3.1.10. Our Industrial Anomaly Detection Models
| Models     | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD | WFDD               | Real-IAD |
|------------|--------------------|--------------------|--------------------|------------|------|--------------------|----------|
| **DM-GRD** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:  | :heavy_check_mark: | :x:      |

### 3.1.11. Supported Models for Segment Anything Industrial Anomaly Detection
| Models | MVTec              | BTAD | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|--------|--------------------|------|------|------------|------|------|----------|
| UCAD   | :heavy_check_mark: | :x:  | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.12. Supported Models for Supervised Industrial Anomaly Detection
| Models | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|--------|--------------------|--------------------|------|------------|------|------|----------|
| DevNet | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |
| PRNet  | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| DRA    | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.13. Supported Models for Few-Shot/Zero-Shot Industrial Anomaly Detection
| Models      | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD | ELPV               | SDD                | AITEX              | BrainMRI           | HeadCT             | DAGM               | DTD                | Br35H              | ISIC               | ColonDB            | ClinicDB           | TN3K               |
|-------------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| WinCLIP     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PromptAD    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| InCTRL      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyCLIP | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| APRIL-GAN   | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaCLIP     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.2. Supported Models for Dashcam Traffic Anomaly Detection
| Models       | ROL                | DoTA               | CCD                | DAD                | A3D                |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| AMNet        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Baseline GRU | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| DSTA         | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| UString      | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| XAI-Accident | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |

### 3.3. Supported Models for Unsupervised Anomaly Detection
| Models      | Avenue             | ShanghaiTech       | Ped2               | Arbitrary Video    |
|-------------|--------------------|--------------------|--------------------|--------------------|
| 3DNet       | :x:                | :x:                | :x:                | :heavy_check_mark: |
| GMM_DAE     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| Jigsaw-VAD  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| OGAM-MRAM   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| SwinAnomaly | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.4. Supported Models for Surveillance Traffic Anomaly Detection
| Models | UCF-Traffic        | TAD                |
|--------|--------------------|--------------------|
| TA-NET | :heavy_check_mark: | :heavy_check_mark: |

### 3.5. Supported Models for Weakly-supervised Anomaly Detection
| Models   | Ten crops          | Flatten crops      | UCF-Crime          | XD-Violence        | ShanghaiTech       |
|----------|--------------------|--------------------|--------------------|--------------------|--------------------|
| MIL      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| RTFM     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| WSAL     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GCN      | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| MGFN     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| HyperVD  | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| UR-DMU   | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| BN-WVAD  | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CMA-LA   | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| ARNet    | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| MyModel  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| MyModel1 | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.6. Supported Models for Feature Extraction
| Models              | RGB                | Point Cloud        | Depth              | Text               | Mask               | Optical Flow       |
|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| C3D                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| I3D                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Dense Trajectory    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Foreground Mask     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| HoF                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| HoG                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Motion Boundary     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Motion Magnitude    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| TVL1                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Depth Anything      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Depth Anything V2   | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| MiDaS               | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| CLIP                | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| LongCLIP            | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| MobileCLIP          | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| OpenCLIP            | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| Segment-Anything    | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Segment-Anything-v2 | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| RobustSAM           | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| DIS                 | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Point-E             | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| EVF-SAM             | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| AlphaCLIP           | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| Marigold            | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| ZoeDepth            | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| NeuFlow v2          | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| DPT                 | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                |
| FastSAM             | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| MobileSAM           | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |

### 3.7. Supported Models for AI City Challenge
| Models   | Iowa DOT           |
|----------|--------------------|
| CETCVLAB | :heavy_check_mark: |
| SIS_Lab  | :heavy_check_mark: |

### 3.7. Supported Models for Extension Research
<div align="center">
    <b>Architectures</b>
</div>
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <td>
                <b>Attention Modules</b>
            </td>
            <td>
                <b>Convolution Architectures</b>
            </td>
            <td>
                <b>Generative Models</b>
            </td>
            <td>
                <b>Geometric Models</b>
            </td>
            <td>
                <b>Losses</b>
            </td>
            <td>
                <b>MLP</b>
            </td>
            <td>
                <b>Optimizer</b>
            </td>
            <td>
                <b>Transformer Models</b>
            </td>
            <td>
                <b>UNet</b>
            </td>
            <td>
                <b>3D Reconstruction Models</b>
            </td>
            <td>
                <b>MoE</b>
            </td>
            <td>
                <b>Mamba</b>
            </td>
            <td>
                <b>KAN</b>
            </td>
            <td>
                <b>Implicit Neural Representations</b>
            </td>
        <tr valign="top">
        <td>
            <ul>
                <li><a href="extention_research/Attention_modules/A2">A2 Attention</a></li>
                <li><a href="extention_research/Attention_modules/ACmix">ACmix Attention</a></li>
                <li><a href="extention_research/Attention_modules/AFT">AFT Attention</a></li>
                <li><a href="extention_research/Attention_modules/Agent Attention">Agent Attention</a></li>
                <li><a href="extention_research/Attention_modules/AoA">Attention on Attention</a></li>
                <li><a href="extention_research/Attention_modules/Axial">Axial Attention</a></li>
                <li><a href="extention_research/Attention_modules/Axial Attention">Axial Attention</a></li>
                <li><a href="extention_research/Attention_modules/BAM">BAM Attention</a></li>
                <li><a href="extention_research/Attention_modules/BidirectionalCrossAttention">Bidirectional Cross Attention</a></li>
                <li><a href="extention_research/Attention_modules/CBAM">CBAM Attention</a></li>
                <li><a href="extention_research/Attention_modules/CoAtNet">CoAtNet Attention</a></li>
                <li><a href="extention_research/Attention_modules/CompositionalAttention">Compositional Attention</a></li>
                <li><a href="extention_research/Attention_modules/Coord">Coord Attention</a></li>
                <li><a href="extention_research/Attention_modules/CoT">CoT Attention</a></li>
                <li><a href="extention_research/Attention_modules/CrissCross">CrissCross Attention</a></li>
                <li><a href="extention_research/Attention_modules/Crossformer">Crossformer Attention</a></li>
                <li><a href="extention_research/Attention_modules/DANet">DANet Attention</a></li>
                <li><a href="extention_research/Attention_modules/DAT">DAT Attention</a></li>
                <li><a href="extention_research/Attention_modules/DeformableAttention">Deformable Attention</a></li>
                <li><a href="extention_research/Attention_modules/ECA">ECA Attention</a></li>
                <li><a href="extention_research/Attention_modules/EMSA">EMSA Attention</a></li>
                <li><a href="extention_research/Attention_modules/External">External Attention</a></li>
                <li><a href="extention_research/Attention_modules/FlashTransformer">Flash Transformer Attention</a></li>
                <li><a href="extention_research/Attention_modules/GFNet">GFNet Attention</a></li>
                <li><a href="extention_research/Attention_modules/GSA">GSA Attention</a></li>
                <li><a href="extention_research/Attention_modules/Halo">Halo Attention</a></li>
                <li><a href="extention_research/Attention_modules/Hamburger">Hamburger Attention</a></li>
                <li><a href="extention_research/Attention_modules/KroneckerAttention">Kronecker Attention</a></li>
                <li><a href="extention_research/Attention_modules/LambdaNet">LambdaNet Attention</a></li>
                <li><a href="extention_research/Attention_modules/Linformer">Linformer Attention</a></li>
                <li><a href="extention_research/Attention_modules/LocalAttention">Local Attention</a></li>
                <li><a href="extention_research/Attention_modules/MOATransformer">MOA Transformer</a></li>
                <li><a href="extention_research/Attention_modules/MobileViT">MobileViT Attention</a></li>
                <li><a href="extention_research/Attention_modules/MobileViTv2">MobileViTv2 Attention</a></li>
                <li><a href="extention_research/Attention_modules/MUSE">MUSE Attention</a></li>
                <li><a href="extention_research/Attention_modules/Nonlocal">Non-local Attention</a></li>
                <li><a href="extention_research/Attention_modules/NystromAttention">Nystrom Attention</a></li>
                <li><a href="extention_research/Attention_modules/Outlook">Outlook Attention</a></li>
                <li><a href="extention_research/Attention_modules/ParNet">ParNet Attention</a></li>
                <li><a href="extention_research/Attention_modules/PolarizedSelfAttention">PolarizedSelf Attention</a></li>
                <li><a href="extention_research/Attention_modules/PSA">PSA Attention</a></li>
                <li><a href="extention_research/Attention_modules/Residual">Residual Attention</a></li>
                <li><a href="extention_research/Attention_modules/S2">S2 Attention</a></li>
                <li><a href="extention_research/Attention_modules/SCSE">SCSE Attention</a></li>
                <li><a href="extention_research/Attention_modules/SelfAttention">Self-attention</a></li>
                <li><a href="extention_research/Attention_modules/SGE">SGE Attention</a></li>
                <li><a href="extention_research/Attention_modules/Shuffle">Shuffle Attention</a></li>
                <li><a href="extention_research/Attention_modules/SimAM">SimAM Attention</a></li>
                <li><a href="extention_research/Attention_modules/SimplifiedSelfAttention">Simplified Self-attention</a></li>
                <li><a href="extention_research/Attention_modules/SK">SK Attention</a></li>
                <li><a href="extention_research/Attention_modules/SlotAttention">Slot Attention</a></li>
                <li><a href="extention_research/Attention_modules/TaylorLinearAttention">Taylor Linear Attention</a></li>
                <li><a href="extention_research/Attention_modules/Triplet">Triplet Attention</a></li>
                <li><a href="extention_research/Attention_modules/UFO">UFO Attention</a></li>
                <li><a href="extention_research/Attention_modules/ViP">ViP Attention</a></li>
                <li><a href="extention_research/Attention_modules/RingAttention">Ring Attention</a></li>
                <li><a href="extention_research/Attention_modules/HierarchicalAttention">Hierarchical Attention</a></li>
                <li><a href="extention_research/Attention_modules/RectifiedLinearAttention">Rectified Linear Attention</a></li>
                <li><a href="extention_research/Attention_modules/STAM">Space Time Attention</a></li>
                <li><a href="extention_research/Attention_modules/HaloAttentiono">Halo Attention</a></li>
                <li><a href="extention_research/Attention_modules/CoordinateDescentAttention">Coordinate Descent Attention</a></li>
                <li><a href="extention_research/Attention_modules/MemoryCompressedAttention">Memory Compressed Attention</a></li>
                <li><a href="extention_research/Attention_modules/Mega">Mega Attention</a></li>
                <li><a href="extention_research/Attention_modules/Tranception">Tranception Attention</a></li>
                <li><a href="extention_research/Attention_modules/InfiniAttention">Infini Attention</a></li>
                <li><a href="extention_research/Attention_modules/SparseAttention">Sparse Attention</a></li>
                <li><a href="extention_research/Attention_modules/MultimodalCrossAttention">Multi-modal Cross Attention</a></li>
                <li><a href="extention_research/Attention_modules/AttentionConvolution">Attention with Convolution</a></li>
                <li><a href="extention_research/Attention_modules/MoA">Mixture of Attention</a></li>
                <li><a href="extention_research/Attention_modules/LinearAttentionTransformer">Linear Attention Transformer</a></li>
                <li><a href="extention_research/Attention_modules/Performer">Performer Attention</a></li>
                <li><a href="extention_research/Attention_modules/ISAB">Induced Set Attention Block</a></li>
                <li><a href="extention_research/Attention_modules/MemoryEfficientAttention">Memory Efficient Attention</a></li>
                <li><a href="extention_research/Attention_modules/Perceiver">Perceiver Attention</a></li>
                <li><a href="extention_research/Attention_modules/CoLT5-attention">CoLT5 Attention</a></li>
                <li><a href="extention_research/Attention_modules/GatedSlotAttention">Gated Slot Attention</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Convolution_architectures/DynamicConvolution">Dynamic Convolution</a></li>
                <li><a href="extention_research/Convolution_architectures/WTConv">Wavelet Convolution</a></li>
                <li><a href="extention_research/Convolution_architectures/InvertibleResidualNetworks">Invertible Residual Networks</a></li>
                <li><a href="extention_research/Convolution_architectures/FrEIA">FrEIA</a></li>
                <li><a href="extention_research/Convolution_architectures/FFF">Fast Feedforward Networks</a></li>
                <li><a href="extention_research/Convolution_architectures/GLOM">GLOM</a></li>
                <li><a href="extention_research/Convolution_architectures/Firefly">Firefly</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Generative_models/EMA">EMA</a></li>
                <li><a href="extention_research/Generative_models/VectorQuantization">Vector Quantization</a></li>
                <li><a href="extention_research/Generative_models/VQ-VAE">VQ-VAE</a></li>
                <li><a href="extention_research/Generative_models/TiTok">TiTok</a></li>
                <li><a href="extention_research/Generative_models/DiT">DiT</a></li>
                <li><a href="extention_research/Generative_models/Fast-DiT">Fast DiT</a></li>
                <li><a href="extention_research/Generative_models/DTR">DTR</a></li>
                <li><a href="extention_research/Generative_models/ANT">ANT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Geometric_models/geoopt">geoopt</a></li>
                <li><a href="extention_research/Geometric_models/HCL">HCL</a></li>
                <li><a href="extention_research/Geometric_models/HyperbolicCV">Hyperbolic CV</a></li>
                <li><a href="extention_research/Geometric_models/HyperbolicEmbedding">Hyperbolic Embedding</a></li>
                <li><a href="extention_research/Geometric_models/HyperbolicVisionTransformer">Hyperbolic Vision Transformer</a></li>
                <li><a href="extention_research/Geometric_models/meru">meru</a></li>
                <li><a href="extention_research/Geometric_models/HCL">HCL</a></li>
                <li><a href="extention_research/Geometric_models/PoincareResnet">Poincare Resnet</a></li>
                <li><a href="extention_research/Geometric_models/HypMix">HypMix</a></li>
                <li><a href="extention_research/Geometric_models/CurvatureGeneration">Curvature Generation</a></li>
                <li><a href="extention_research/Geometric_models/CO-SNE">CO-SNE</a></li>
                <li><a href="extention_research/Geometric_models/HypAD">HypAD</a></li>
                <li><a href="extention_research/Geometric_models/HVT">HVT</a></li>
                <li><a href="extention_research/Geometric_models/HiHPQ">HiHPQ</a></li>
                <li><a href="extention_research/Geometric_models/GM-VAE">GM-VAE</a></li>
                <li><a href="extention_research/Geometric_models/GHSW">GHSW</a></li>
                <li><a href="extention_research/Geometric_models/DOHSC-DO2HSC">DOHSC-DO2HSC</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Losses/OT-CLIP">OT CLIP</a></li>
                <li><a href="extention_research/Losses/GradNorm">GradNorm</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/MLP/gMLP">gMLP</a></li>
                <li><a href="extention_research/MLP/MLP-Mixer">MLP Mixer</a></li>
                <li><a href="extention_research/MLP/Segformer">Segformer</a></li>
                <li><a href="extention_research/MLP/ResMLP">ResMLP</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Optimizer/AdamAtan2">Adam Atan 2</a></li>
                <li><a href="extention_research/Optimizer/AdaMod">AdaMod</a></li>
                <li><a href="extention_research/Optimizer/Adan">Adan</a></li>
                <li><a href="extention_research/Optimizer/Lion">Lion</a></li>
                <li><a href="extention_research/Optimizer/GradientAscent">Gradient Ascent</a></li>
                <li><a href="extention_research/Optimizer/DecoupledLionW">Decoupled LionW</a></li>
                <li><a href="extention_research/Optimizer/GradientEquillibrum">Gradient Equillibrum</a></li>
                <li><a href="extention_research/Optimizer/CV-Model-Compression">Computer Vision Model Compression Techniques</a></li>
                <li><a href="extention_research/Optimizer/Grokfast">Grokfast</a></li>
                <li><a href="extention_research/Optimizer/AdEMAMix">AdEMAMix</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Transformer_models/RotaryEmbedding">Rotary Embedding</a></li>
                <li><a href="extention_research/Transformer_models/TNT">Transformer in Transformer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/UNet/Uformer">Uformer</a></li>
                <li><a href="extention_research/UNet/ReversibleUnet">Reversible U-Net</a></li>
                <li><a href="extention_research/UNet/X-UNet">X-UNet</a></li>
            </ul>
        </td>
         <td>
            <ul>
                <li><a href="extention_research/3d_reconstruction_models/2D_Gaussian_Splatting">2D Gaussian Splatting</a></li>
                <li><a href="extention_research/3d_reconstruction_models/3D_Gaussian_Splatting">3D Gaussian Splatting</a></li>
                <li><a href="extention_research/3d_reconstruction_models/iNeRF">iNeRF</a></li>
                <li><a href="extention_research/3d_reconstruction_models/NeRF">NeRF</a></li>
                <li><a href="extention_research/3d_reconstruction_models/SyncDreamer">SyncDreamer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/MoE/MoE-CNN">MoE CNN</a></li>
                <li><a href="extention_research/MoE/Soft-MoE">Soft MoE</a></li>
                <li><a href="extention_research/MoE/ViT-Soft-MoE">ViT Soft MoE</a></li>
                <li><a href="extention_research/MoE/PEER">PEER</a></li>
                <li><a href="extention_research/MoE/ST-MoE">ST MoE</a></li>
                <li><a href="extention_research/MoE/Sparse-MoE">Sparse MoE</a></li>
                <li><a href="extention_research/MoE/LIMoE">LIMoE</a></li>
                <li><a href="extention_research/MoE/MoD">MoD</a></li>
                <li><a href="extention_research/MoE/MoE-Mamba">MoE Mamba</a></li>
                <li><a href="extention_research/MoE/MHMoE">Multi-Head MoE</a></li>
                <li><a href="extention_research/MoE/MoNE">MoNE</a></li>
                <li><a href="extention_research/MoE/MoE-Fusion">MoE-Fusion</a></li>
                <li><a href="extention_research/MoE/MoE">MoE</a></li>
                <li><a href="extention_research/MoE/SinkhornRouter">Sinkhorn Router</a></li>
                <li><a href="extention_research/MoE/BlackMamba">BlackMamba</a></li>
                <li><a href="extention_research/MoE/MoU">MoU</a></li>
                <li><a href="extention_research/MoE/Switch-MoE">Switch MoE</a></li>
                <li><a href="extention_research/MoE/DiT-MoE">DiT MoE</a></li>
                <li><a href="extention_research/MoE/TC-MoA">TC-MoA</a></li>
                <li><a href="extention_research/MoE/tutel">tutel</a></li>
                <li><a href="extention_research/MoE/MoSE">MoSE</a></li>
                <li><a href="extention_research/MoE/M4oE">M4oE</a></li>
                <li><a href="extention_research/MoE/NID">NID</a></li>
                <li><a href="extention_research/MoE/FAME">FAME</a></li>
                <li><a href="extention_research/MoE/MVMoE">MVMoE</a></li>
                <li><a href="extention_research/MoE/Switch-DiT">Switch-DiT</a></li>
                <li><a href="extention_research/MoE/ScatterMoE">ScatterMoE</a></li>
                <li><a href="extention_research/MoE/SFAN">SFAN</a></li>
                <li><a href="extention_research/MoE/SMoE">SMoE</a></li>
                <li><a href="extention_research/MoE/DeepMoE">DeepMoE</a></li>
                <li><a href="extention_research/MoE/GMoE">GMoE</a></li>
                <li><a href="extention_research/MoE/MoFME">MoFME</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/Mamba/VisionMamba">Vision Mamba</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/KAN/EfficientKAN">Efficient KAN</a></li>
                <li><a href="extention_research/KAN/Vision-KAN">Vision KAN</a></li>
                <li><a href="extention_research/KAN/Fast-KAN">Fast KAN</a></li>
                <li><a href="extention_research/KAN/FourierKAN">Fourier KAN</a></li>
                <li><a href="extention_research/KAN/MoE-KAN">MoE KAN</a></li>
                <li><a href="extention_research/KAN/KAT">KAT</a></li>
                <li><a href="extention_research/KAN/GR_KAN">GR_KAN</a></li>
                <li><a href="extention_research/KAN/KAN-COIN">KAN COIN</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="extention_research/INR/siren">siren</a></li>
                <li><a href="extention_research/INR/CoLIE">CoLIE</a></li>
                <li><a href="extention_research/INR/WIRE">WIRE</a></li>
                <li><a href="extention_research/INR/NeRCo">NeRCo</a></li>
                <li><a href="extention_research/INR/GKAN">GKAN</a></li>
                <li><a href="extention_research/INR/SCONE">SCONE</a></li>
                <li><a href="extention_research/INR/LINR">LINR</a></li>
                <li><a href="extention_research/INR/FR-INR">FR-INR</a></li>
                <li><a href="extention_research/INR/NeRD">NeRD</a></li>
                <li><a href="extention_research/INR/INRAD">INRAD</a></li>
                <li><a href="extention_research/INR/SineKAN">SineKAN</a></li>
                <li><a href="extention_research/INR/NeRP">NeRP</a></li>
                <li><a href="extention_research/INR/MINER">MINER</a></li>
                <li><a href="extention_research/INR/O-INR">O-INR</a></li>
                <li><a href="extention_research/INR/ImpSq">ImpSq</a></li>
                <li><a href="extention_research/INR/ANR">ANR</a></li>
                <li><a href="extention_research/INR/Partition-INR">Partition INR</a></li>
                <li><a href="extention_research/INR/COIN">COIN</a></li>
                <li><a href="extention_research/INR/LoE">LoE</a></li>
            </ul>
        </td>
        </tr>
    </tbody>
</table>
 
## 4. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {OpenAnomaly: An Open Source Implementation of Anomaly Detection Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/SKKUAutoLab/ETSS-06-Anomaly},
  year         = {2023}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.