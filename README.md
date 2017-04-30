# FusionNet_Pytorch

[FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics](https://arxiv.org/abs/1612.05360)

## Download code

~~~
git clone https://github.com/GunhoChoi/FusionNet_Pytorch
cd FusionNet_Pytorch
~~~

## Download Map data

~~~
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
tar -xzvf maps.tar.gz
~~~

## Make model directory

~~~
mkdir model
~~~

## Train Model
~~~
python3 main.py  -> medical image without augmentation
python3 main_augmented.py  -> map image with naive augmentation
~~~
Out of memory error -> change batch size or out_dim

## Result

### Medical Image
<img src="./result/original_image_185_0.png" width="20%"><img src="./result/label_image_185_0.png" width="20%"><img src="./result/gen_image_185_0.png" width="20%">

### Map Image
<img src="./result/satel_image_0_0.png" width="20%"><img src="./result/map_image_0_0.png" width="20%"><img src="./result/gen_image_0_0.png" width="20%">

Original Image / Label Image / Generated Image
