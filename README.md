## Uninformed Teacher-Student for hard-sample distillation in weakly supervised mitosis localization

**Mitosis localzation** is a challenging tasks for computational pathology. One of the main challenges involves the use of
**centroid annotations**, and the presence of annotated **hard negatives** during training.

**How can you detect these hard negatives and avoid learning a biased solution?** Just don't tell the model where the 
mitosis are in the image! - follow a **weakly supervised strategy** -, and check **di-similarities** between annotations and
predictions for training a **distilled Student model**.

You can find more details on the following [manuscript]().

## Installation

* Install in your enviroment a compatible torch version with your GPU. For example:

```
conda create -n uts_env python=3.8 -y
conda activate uts_env
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

* Clone and install requirements.

```
git clone https://github.com/cvblab/Mitosis-UTS.git
cd Mitosis-UTS
pip install -r requirements.txt
```

## Datasets

For proper usage, training and validation of the proposed UTS method, please download and pre-process the following datasets as indicated:

1. Download the datasets.

* [TUPAC16-auxiliary](https://tupac.grand-challenge.org/Dataset/) - include the dataset at `./local_data/datasets/TUPAC16/`.
* [MITOS-ATYPIA14](https://mitos-atypia-14.grand-challenge.org/Donwload/) - include the dataset at `./local_data/datasets/MITOS14/`.
* [MIDOG21](https://imig.science/midog2021/download-dataset/) - include the dataset at `./local_data/datasets/MIDOG21/`.

2. Extract patches from high-power field views.

```
python data/preprocess.py --dataset TUPAC16 --extract_patches True --stain_norm False
python data/preprocess.py --dataset MIDOG21 --extract_patches True --stain_norm False
```

3. Stain normalization using Macenko's method - This process might take a while... be patient!

```
python data/preprocess.py --dataset TUPAC16 --extract_patches False --stain_norm True
python data/preprocess.py --dataset MIDOG21 --extract_patches False --stain_norm True
```

## UTS Training

1. Weakly supervised model training, corresponding to UTS-Teacher model.
```
python main.py --dataset TUPAC16 --experiment_name TUPAC16_UTS_teacher --strong_augmentation True
```
2. UTS model training - with noise injection during Teacher training.
```
python main.py --dataset TUPAC16 --experiment_name TUPAC16_UTS_student --distillation True --teacher_experiment TUPAC16_UTS_teacher
```

## Evaluation

```
python evaluate.py --dataset TUPAC16 --dir_model ./local_data/results/TUPAC16_UTS_teacher/
```

## Citation

If you find this repository useful, please consider citing this paper:
```
@article{UTS2023,
  title={Uninformed Teacher-Student for hard-sample distillation in weakly supervised mitosis localization},
  author={Claudio Fernandez-Martín and Julio Silva-Rodriguez and Umay Kiraz and Sandra Morales and Emiel A.M. Janssen and Valery Naranjo},
  journal={ArXiv Preprint},
  year={2023}
}
```

## TO-DOs

- [ ] Add paper link.
- [ ] Add pre-processing code for MITOS14.
- [ ] Release pre-trained weights.

## License

- **Code and Model Weights** are released under [Apache 2.0 license](LICENSE)
