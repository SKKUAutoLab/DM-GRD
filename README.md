# DM-GRD

**💡 This is the official implementation of the paper "Dual Memory Networks Guided Reverse Distillation for Unsupervised Anomaly Detection (ACCV 2024)"**  

## 🔧 Installation
```
git clone https://github.com/SKKUAutoLab/DM-GRD
cd DM-GRD
conda env create --name anomaly --file=environment.yml
conda activate anomaly
```

## 🏆 Dataset preparation
For the MVTec dataset, please download it from this [link](https://www.mvtec.com/company/research/datasets/mvtec-ad).

For the BTAD dataset, please download it from this [repository](https://github.com/pankajmishra000/VT-ADL).

For the VisA dataset, please download it from this [repository](https://github.com/amazon-science/spot-diff).

For the DTD dataset, please download it from this [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

## 🚀 Experiments
### 🌞 Training and testing DM-GRD on the MVTec dataset
```shell
bash scripts/train_mvtec.sh
bash scripts/test_mvtec.sh
```

### 🌞 Training and testing DM-GRD on the BTAD dataset
```shell
bash scripts/train_btad.sh
bash scripts/test_btad.sh
```

### 🌞 Training and testing DM-GRD on the VisA dataset
```shell
bash scripts/train_visa.sh
bash scripts/test_visa.sh
```

## 🔗 Citation
If you find our work useful, please cite the following:
```
@inproceedings{tran2024dual,
  title={Dual memory networks guided reverse distillation for unsupervised anomaly detection},
  author={Tran, Chi Dai and Pham, Long Hoang and Tran, Duong Nguyen-Ngoc and Tran and Ho, Quoc Pham-Nam},
  booktitle={Proceedings of the asian conference on computer vision},
  pages={x--x},
  year={2024}
}
```

## ☎️ Contact
If you have any questions, feel free to contact `Chi D. Tran` ([ctran743@gmail.com](ctran743@gmail.com)).

## 🙏 Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [hq-deng/RD4AD](https://github.com/hq-deng/RD4AD)
* [tientrandinh/Revisiting-Reverse-Distillation](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
<!--te-->
