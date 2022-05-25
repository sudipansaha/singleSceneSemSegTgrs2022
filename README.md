# Unsupervised Single-Scene Semantic Segmentation for Earth Observation

**Instructions for Vaihingen dataset** 

Code can be run using following two commands:

For training the model on single scene (after running this command the model will be saved to ./trainedModels/)
$ python trainVaihingen.py --manualSeed 85 --nFeaturesIntermediateLayers 64  --nFeaturesFinalLayer 8 --numTrainingEpochs 2 --modelName Model5ChannelInitialToMiddleLayersDifferent

For obtaining segmentation maps from the test scenes (after running this command the model will be saved to ./results/vaihingen/)
$ python obtainSegMapVaihingen.py

Different manual seeds can be set in the above command.

Please download the Vaihingen dataset from appropriate source and save it in the directory (w.r.t the code) "../data/Vaihingen/"


### Citation
If you find this code or the multi-season dataset useful, please consider citing:
```[bibtex]
@article{saha2022unsupervised,
  title={Unsupervised Single-Scene Semantic Segmentation for Earth Observation},
  author={Saha, Sudipan and Shahzad, Muhammad and Mou, Lichao and Song, Qian and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```
