# Latent Space Roadmap
This is the code for the paper: "[Latent Space Roadmap for Visual Action Planning of Deformable and Rigid Object Manipulation](https://arxiv.org/abs/2003.08974)"

Visit the dedicated website: [Latent Space Roadmap website](https://visual-action-planning.github.io/lsr/) for more information.

If you use this code in your work, please cite it as follows:

## Bibtex

```
@inproceedings{lippi2020latent,
  title={Latent Space Roadmap for Visual Action Planning of Deformable and Rigid Object Manipulation},
  author={Lippi, Martina and Poklukar, Petra and Welle, Michael C and Varava, Anastasiia and Yin, Hang and Marino, Alessandro and Kragic, Danica},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```

## Stacking example

### setup

```
pip install -r requirements.txt
```

### Datasets

Download LSR stacking datasets:
```
cd datasets/
python get_datasets.py
cd ..
```


### Train models
To make train and test split use:
```
python preprocess_dataset.py
```

To train the VAE use:
```
python train_VAE.py --exp_vae=VAE_UnityStacking_L1  --cuda=True
```

To train the APN use:
```
touch models/APN_UnityStacking_evaluation_results.txt

python train_APN_stacking.py \
                --exp_apn=APN_UnityStacking_L1  \
                --seed=98765 \
                --generate_new_splits=1 \
                --generate_apn_data=1 \
                --train_apn=1 \
                --eval_apn=1 \
                --cuda=True
```


### execute stacking LSR example

```
python lsr_stacking_example.py --seed=98765 --lable_ls=True --build_lsr=True --example=True
```

You should get a image like this in the root folder as a result: (depending on your random seed)

![Stacking example](stacking_example_98765.png)


