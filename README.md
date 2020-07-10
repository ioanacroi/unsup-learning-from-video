# Unsupervised learning from Video to detect foreground objects in single images

Paper: [ICCV 2017 Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Croitoru_Unsupervised_Learning_From_ICCV_2017_paper.pdf)


## Inference
Download model weights: [weights](https://drive.google.com/open?id=1e2-LEvSCIirFKt-iKZvVHu7QbGDbbE36)
`python get_segmentation.py --in_file path_to_file --model path_to_weights`


## Data preparation

1. Run [VideoPCA](https://sites.google.com/site/multipleframesmatching/) on the training videos. Please follow the instructions provided [here](https://sites.google.com/site/multipleframesmatching/)
2. Create tfrecords with the following [structure](https://github.com/ioanacroi/unsup-learning-from-video/blob/b97fbf82dc3d46c952e0ff1d6fac54230d6bc49b/train.py#L122). The `frame0` represents the RGB frame while the `softseg` represents the mask given by VideoPCA. To create tfrecords please follow the instructions provided [here](https://www.tensorflow.org/tutorials/load_data/tfrecord).

## Training
`python train.py`

Please make sure to update the tfrecords path in train.py beforehands. Note: the training code needs some updates, but as for now it is provided as a guideline and works with tensorflow 1.10.1.



For more details and qualitative results please visit our [project page](https://sites.google.com/view/unsupervisedlearningfromvideo).

Please consider citing the following paper if you use this model:
```
@InProceedings{Croitoru_2017_ICCV,
author = {Croitoru, Ioana and Bogolin, Simion-Vlad and Leordeanu, Marius},
title = {Unsupervised Learning From Video to Detect Foreground Objects in Single Images},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
