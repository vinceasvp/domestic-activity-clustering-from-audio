# Domestic Activity Clustering from Audio via Depthwise Separable Convolutional Autoencoder Network(DSCAN)

## Usage

1. Prepare datasets.  Download and unzip dataset of [DCASE 2018, Task 5: Monitoring of domestic activities based on multi-channel acoustics - Development dataset | Zenodo](https://zenodo.org/record/1247102#.YuyeJWNBztU).

2. Extract acoustic features.   
   `python extract_feature.py --dtpth DATASET_PATH`     

3. Run experiment on DCASE.   
   `python DCEC.py dcase --data_path FEATURE_PATH`   

## Reference

------------------------------------

The codes were adapted from

- [XifengGuo/DCEC](https://github.com/XifengGuo/DCEC)
