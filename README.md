<p align="center">
<img src=./images/logo.png width=40% height=40%>
</p>

# GraspMixer
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GraspMixer ([IEEE T-ASE](https://ieeexplore.ieee.org/abstract/document/10843724)) is a combination of Contact Surface Pair Estimator (CSPE) and Grasp Quality Mapping Network (GQM-Net) for offline 6DOF grasp detection. The framework takes a complete object point cloud file to generate possible contact surface pairs and their robustness (a.k.a grasp dictionaries) for a parallel jaw gripper using both analytical and data-driven models.

![Example 0](./images/pipeline.png)

The project was done while [Hojun Lee](https://www.linkedin.com/in/hjunlee94/) was working for Barton Research Group ([BRG](https://brg.engin.umich.edu/)) at the University of Michigan.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Download Process](#download-process)
- [How to Run](#how-to-run)
    - [CSPE_v2](#cspe_v2)
    - [Extra Processing](#extra-processing)
    - [Quality Estimation (ML)](#quality-estimation)
- [Dataset](#dataset)
    - [Structure](#structure)
    - [Sample Generation](#sample-generation)
    - [Summary Table](#summary-table)
- [Citation](#citation)
- [ToDo Lists](#todo-lists)

---

## Repository Structure

    ├── ML_modules        # DL modules 
    │   ├── engine.py
    │   ├── eval.py
    │   ├── model.py      
    │   ├── quality.py
    │   ├── train.py     
    │   ├── transforms.py 
    │   └── utils.py    
    ├── dataset
    |   ├── train         # ML training data
    |   └── test          # ML test data
    ├── images              
    ├── objects
    │   ├── dicts         # .txt grasp files
    │   └── pcds          # .pcd files
    ├── CSPE_utils.py     
    ├── CSPE_v2.py        
    ├── gripper_config.py 
    └── helper.py     

## Download Process

    git clone https://github.com/kidpaul94/GraspMixer.git
    cd GraspMixer/
    pip3 install -r requirements.txt
    pip3 install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

## How to Run

### CPPE_v2:

> [!NOTE]
`CSPE_v2.py` receives several different arguments. Run the `--help` command to see everything it receives.

    python3 CSPE_v2.py --help

### Extra Processing:

> [!NOTE]
`helper.py` receives several different arguments to convert a CAD model to a point cloud or visualize grasp configurations. Run the `--help` command to see everything it receives.

    python3 helper.py --help

### Quality Estimation:

> [!NOTE]
`train.py` receives several different arguments. Run the `--help` command to see everything it receives.
    
    cd ML_modules/
    python3 train.py --help

> [!NOTE]
`eval.py` receives several different arguments. Run the `--help` command to see everything it receives.
    
    cd ML_modules/
    python3 eval.py --help

## Dataset

### Structure:

    └── dataset
        ├── train         
        |   ├── obj_01              # obj_01 grasp folder
        |   |   ├── 0000_misc_1.npy
        |   |   ├── 0000_misc_2.npy
        |   |   ├── 0000R_pts.npy
        |   |   ├── 0000L_pts.npy  
        |   |   └── prob.txt        
        |   ├── obj_02              # obj_02 grasp folder
        |   ├── obj_03              # obj_03 grasp folder
        |   └── summary.csv         # summary of file_paths & labels
        └── test          
        |   ├── obj_04              # obj_04 grasp folder
        |   |   ├── 0000_misc_1.npy
        |   |   ├── 0000_misc_2.npy
        |   |   └── 0000_pts.npy
        |   |   └── prob.txt           
        |   ├── obj_05              # obj_05 grasp folder
        |   ├── obj_06              # obj_06 grasp folder
        |   └── summary.csv         # summary of file_paths & labels

### Sample Generation:

For each object, we use a [Gazebo simulation](https://github.com/kidpaul94/grasp-factory) to simulate grasps generated by CSPE and record a success rate (a.k.a. *label*) of each sample. If you have your own simulation or a physical robot, you can use them as well to generate your own dataset. 

### Summary Table:

Once you clean up your training, testing samples, and their labels in the dataset folder, you can use [gen_csv](https://github.com/kidpaul94/GraspMixer/blob/00111ee931b5104e9e17908ef8151cdeea55d8f6/ML_modules/utils.py#L76) in `util.py` to generate summary.csv

## Citation
If you found GraspMixer useful in your research, please consider citing:

```plain
@article{lee2025graspmixer,
  title={GraspMixer: Hybrid of Contact Surface Sampling and Grasp Feature Mixing for Grasp Synthesis},
  author={Lee, Hojun and Toner, Tyler and Tilbury, Dawn M and Barton, Kira},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2025},
  publisher={IEEE}
}
```

## ToDo Lists

| **ML pipeline** | ![Progress](https://geps.dev/progress/100) |
| --- | --- |
| **Documentation** | ![Progress](https://geps.dev/progress/100) |
