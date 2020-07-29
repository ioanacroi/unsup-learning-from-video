# Unsupervised learning from Video to detect foreground objects in single images

Papers: [ICCV 2017 Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Croitoru_Unsupervised_Learning_From_ICCV_2017_paper.pdf) and [IJCV 2019 Paper](https://link.springer.com/article/10.1007/s11263-019-01183-3)


## First iteration

### Data preparation

1. Run [VideoPCA](https://sites.google.com/site/multipleframesmatching/) on the training videos. Please follow the instructions provided [here](https://sites.google.com/site/multipleframesmatching/)
2. Selection - compute the mean of non-zero pixels for each mask obtained at 1. Keep only the top 10-20% masks based on the mean of the non-zero pixels.
3. Create tfrecords with the following [structure](https://github.com/ioanacroi/unsup-learning-from-video/blob/b97fbf82dc3d46c952e0ff1d6fac54230d6bc49b/train.py#L122). The `frame0` represents the RGB frame while the `softseg` represents the mask given by VideoPCA. To create tfrecords please follow the instructions provided [here](https://www.tensorflow.org/tutorials/load_data/tfrecord).

### Training
`python [model-name.py] (e.g. python lowRes-net.py)`

Please make sure to update the tfrecords path in [model-name.py] beforehands. Note: the training code needs some updates, but as for now it is provided as a guideline and works with tensorflow 1.10.1.

## Second iteration

### Data preparation
1. Extract the output off all the models trained in the First iteration. For each frame you will obtain several masks depending on the number of used models.
2. Run the confidence net on the masks extracted at point 1.

    2.1. Download the EvalSeg-Net weights from TODO

    2.2. Run the pre-trained EvalSeg-Net net `python evalSeg-Net-inference.py [weights-path]`

3. Selection - Keep only the top 10-20% masks based on the score given by the confidence net (usually keeping all the frames over 80 confidence score is a good practice - as it is set in the code).
4. Create tfrecords with the same structure as for the First iteration.
5. Re-train all/some of the models from scratch on the new data obtained at point 4. We have observed that DilateU-Net usually performs the best, so this would be a good candidate.

## Pre-trained models
You can also use our pre-trained ICCV model. In order to use it, download the model weights: [weights](https://drive.google.com/open?id=1e2-LEvSCIirFKt-iKZvVHu7QbGDbbE36) then
`python get_segmentation.py --in_file path_to_file --model path_to_weights`

For more details and qualitative results please visit our [project page](https://sites.google.com/view/unsupervisedlearningfromvideo).

Please consider citing the following papers if you use these models:
```
@InProceedings{Croitoru_2017_ICCV,
author = {Croitoru, Ioana and Bogolin, Simion-Vlad and Leordeanu, Marius},
title = {Unsupervised Learning From Video to Detect Foreground Objects in Single Images},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
and
```
@article{croitoru2019unsupervised,
  title={Unsupervised learning of foreground object segmentation},
  author={Croitoru, Ioana and Bogolin, Simion-Vlad and Leordeanu, Marius},
  journal={International Journal of Computer Vision},
  volume={127},
  number={9},
  pages={1279--1302},
  year={2019},
  publisher={Springer}
}
```
