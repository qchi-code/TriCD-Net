# TriCD-Net

### Abstract
  This project is about research implementation of FSMIS.
### Dependencies
Please install following essential dependencies:
```
dcm2nii
json5==0.9.28
jupyter==1.1.1
nibabel==5.3.2
numpy==2.1.3
opencv-python==4.5.5.62
Pillow>=8.1.1
sacred==0.8.2
scikit-image==0.18.3
SimpleITK==1.2.3
torch==2.5
torchvision=0.20.1
tqdm==4.62.3
```

### Data sets and pre-processing
Download:
1) **CHAOS-MRI**: [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2) **Synapse-CT**: [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
3) **CMR**: [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

### Pretrained Model Weights
The pretrained checkpoints for all three datasets are provided via the following [Google Drive](https://drive.google.com/file/d/1d2-kjKGsKUd--ETPiCnDjmtNxfzfERnh/view?usp=drive_link) link for testing and evaluation. Please download the archive manually and extract it locally before use.

### Training
1. Compile `./data/supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./data/supervoxels/setup.py build_ext --inplace`) and run `./data/supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put your checkpoints folder, then replace the absolute path in the code `./models/encoder.py`.  
3. Run `./script/train_CHAOST2.sh` 
Please note that you need to modify the source domain yourself.

### Activation Map
We also provide visualization code for inspecting the activation maps. Using the same visualization pipeline, comparable results for the variant without FEBR can be obtained simply by disabling the FEBR module, since the difference mainly comes from the feature refinement introduced by FEBR itself.

### Citation
