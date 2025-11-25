# MonoLDP
An official pytorch implemention of MonoLDP
## Introduction
MonoLDP is a two-stage network consists of a depth prediction module and a relative pose estimation module. 
![network](https://github.com/user-attachments/assets/ce961a01-387a-4415-a44d-3e96aed45f3d)


## Depth Prediction
### Preparation
1. Pytorch and conda environment
create conda environment
```
conda conda create -n monoldp_depth python=3.8
conda activate monoldp_depth
conda install pytorch=1.7.0 torchvision=0.8.0  cudatoolkit=11.0 -c pytorch
```
2. Requirements
```
cd MonoLDP/Depth_Estimation
pip install -r requirements.txt
```
3. Dataset
We have the processed NYU v2 dataset [here](https://drive.google.com/file/d/1AXUq0zHJQsWQ13DRSCUEiuAzeljOgefn/view?usp=drive_link). The prepared structure is as follows:
```
|-nyu_data
|--nyu2_train
|--nyu2_test
```
Or you can download the original sampled [dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view) here and run code

```
python python preprocess/extract_superpixel.py --data_path nyu_data/
python preprocess/extract_lineseg.py --data_path nyu_data/
```
Just notice that line segmentation only requires the installation of any version of opencv-python lower than 3.4.6, so you may have to reinstall the opencv.

### Training
You can modify the default settings in the options.py. For training just run
```
python train.py --data_path nyu_data/
```
### Single image prediction
The network predicts single RGB image by 
```
python inference_single_image.py --image_path $IMAGE_PATH --load_weights_folder $MODEL_PATH
```

### Evaluation
| Methods             | Supervision   |   AbsRel ↓ |   RMS ↓ |   δ₁ ↑ |   δ₂ ↑ |   δ₃ ↑ |
|:--------------------|:--------------|-----------:|--------:|-------:|-------:|-------:|
| Eigen et al. (2014) | ✓             |      0.158 |   0.641 |  0.769 |  0.95  |  0.988 |
| DORN (2018)         | ✓             |      0.115 |   0.509 |  0.828 |  0.965 |  0.992 |
| NewCRFs (2022)      | ✓             |      0.095 |   0.334 |  0.922 |  0.992 |  0.998 |
| DistDepth (2022)    | Δ             |      0.130 |   0.517 |  0.832 |  0.963 |  0.990 |
| SC-Depthv3 (2023)   | Δ             |      0.123 |   0.486 |  0.848 |  0.963 |  0.991 |
| GasMono (2023)      | Δ             |      0.113 |   0.459 |  0.871 |  0.973 |  0.992 |
| Monodepth2 (2019)   | ✗             |      0.161 |   0.600 |  0.771 |  0.948 |  0.987 |
| P²Net (2021)        | ✗             |      0.150 |   0.561 |  0.796 |  0.948 |  0.986 |
| PLNet (2021)        | ✗             |      0.144 |   0.540 |  0.807 |  0.957 |  0.990 |
| MonoIndoor (2021)   | ✗             |      0.142 |   0.581 |  0.802 |  0.952 |  0.990 |
| MonoIndoor++ (2021) | ✗             |      0.133 |   0.551 |  0.830 |  0.964 |  0.991 |
| IndoorDepth (2023)  | ✗             |      0.126 |   0.494 |  0.845 |  0.965 |  0.991 |
| Ours                | ✗             |      0.122 |   0.381 |  0.859 |  0.970 |  0.994 |

The pretrained model is provide [here](https://drive.google.com/file/d/1jh_RDYTmCKIGlIuon_i3qxirAYbN2HAX/view?usp=drive_link). And to test you can run this command:
```
python evaluate_nyu_depth.py --data_path your data path --load_weights_folder your model
```

### Acknowledgements
The project borrows codes from [Monodepth2](https://github.com/nianticlabs/monodepth2), [P^2Net](https://github.com/svip-lab/Indoor-SfMLearner), [PLNet](https://github.com/HalleyJiang/PLNet/tree/main) and [IndoorDepth](https://github.com/fcntes/IndoorDepth/tree/main). Many thanks to their authors.
