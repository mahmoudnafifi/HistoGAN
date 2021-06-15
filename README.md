# HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms

[Mahmoud Afifi](https://sites.google.com/view/mafifi), 
[Marcus A. Brubaker](https://mbrubake.github.io/), 
and [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)

York University  &nbsp;&nbsp; 

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Afifi_HistoGAN_Controlling_Colors_of_GAN-Generated_and_Real_Images_via_Color_CVPR_2021_paper.pdf) | [Supplementary Materials](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Afifi_HistoGAN_Controlling_Colors_CVPR_2021_supplemental.pdf) | [Video](https://www.youtube.com/watch?v=uMN85KBV4Rw) | [Poster](https://drive.google.com/file/d/1AQNnodTUFOtTKSaXPbNEUAEbLbQNZDvE/view) | [PPT](https://drive.google.com/file/d/167rFIDyUS368yecMSXPKQYzj73pY282i/view)

![teaser](https://user-images.githubusercontent.com/37669469/100063951-e497dc00-2dff-11eb-8454-98f720fe7d04.jpg)

Reference code for the paper [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms.](https://arxiv.org/abs/2011.11731) Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. In CVPR, 2021. If you use this code or our dataset, please cite our paper:
```
@inproceedings{afifi2021histogan,
  title={HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Abstract
*<p align="justify">
In this paper, we present HistoGAN, a color histogram-based method for controlling GAN-generated images' colors. We focus on color histograms as they provide an intuitive way to describe image color while remaining decoupled from domain-specific semantics. Specifically, we introduce an effective modification of the recent StyleGAN architecture to control the colors of GAN-generated images specified by a target color histogram feature. We then describe how to expand HistoGAN to recolor real images. For image recoloring, we jointly train an encoder network along with HistoGAN. The recoloring model, ReHistoGAN, is an unsupervised approach trained to encourage the network to keep the original image's content while changing the colors based on the given target histogram. We show that this histogram-based approach offers a better way to control GAN-generated and real images' colors while producing more compelling results compared to existing alternative strategies.</p>*

<p align="center">
  <img width = 95% src="https://user-images.githubusercontent.com/37669469/100063841-c500b380-2dff-11eb-8c4a-15fc57bb9caf.gif">
 </p>


## Code


### Prerequisite
* Pytorch
* numpy
* tqdm
* pillow
* linear-attention-transformer (optional)
* vector-quantize-pytorch (optional)
* torch-optimizer
* retry

Conda commands:
```
conda create -n histoGAN python=3.6 numpy=1.13.3 scipy 
conda activate histoGAN
conda install pytorch torchvision -c python
conda install -c conda-forge tqdm
conda install -c anaconda pillow
pip install linear-attention-transformer
pip install vector-quantize-pytorch
pip install torch-optimizer
pip install retry
```

* * *

### Histogram loss 
We provide a [Colab notebook example](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) code to compute our histogram loss. This histogram loss is differentiable and can be easily integrated into any deep learning optimization.

* * *

### HistoGAN
To train/test a histoGAN model, use `histoGAN.py`. Trained models should be located in the `models` directory and each trained model's name should be a subdirectory in the `models` directory. For example, to test a model named `test_histoGAN`, you should have `models/test_histoGAN/model_X.pt` exists (where `X` refers to the epoch number). 


#### Training
To train a histoGAN model on a dataset located at `./datasets/faces/` for example:

```python histoGAN.py --name histoGAN_model --data ./datasets/faces/ --num_train_steps XX --gpu 0```

`XX` should be replaced with the number of iterations. 

#### Testing
Here is an example of how to generate new samples of a trained histoGAN models named *Faces_histoGAN*:

```python histoGAN.py --name Faces_histoGAN --generate True --target_his ./target_images/1.jpg --gpu 0```

The shown figure below illustrates what this command does. First, we generate a histogram feature of this input image. Then, this feature is fed into our HistoGAN to generate face image samples. 

![histogan_exmaple1](https://user-images.githubusercontent.com/37669469/120030862-fd0ecc00-bfc5-11eb-9b10-f9e0ba1f143f.gif)

Generated samples will be located in `./results_HistoGAN/Faces_histoGAN`.

Another example is given below, where we use a fixed input noise and style vectors for the first blocks of the generator network, while we change the input histograms. In this example, we first use `--save_noise_latent = True` to save the noise and latent data for the first blocks. Then, we load the saved noise and latent files, using `--target_noise_file` and `--target_latent_file`, to generate the same samples but with different color histograms. 

```python histoGAN.py --name Faces_histoGAN --generate True --target_his ./target_images/1.jpg --save_noise_latent True --gpu 0```


```python histoGAN.py --name Faces_histoGAN --generate True --target_his ./target_images/ --target_noise_file ./temp/Face_histoGAN/noise.py --target_latent_file ./temp/Faces_histoGAN/latents.npy --gpu 0```


![histogan_example2](https://user-images.githubusercontent.com/37669469/120036046-88d82680-bfcd-11eb-9109-4f56dbc800a9.gif)


Additional useful parameters are given below. 

#### Parameters
* `--name`: Model name.
* `--models_dir`: Models directory (to save or load models).
* `--data`: Dataset directory (for training).
* `--new`: Set to True to train a new model. If `--new = False`, it will start training/evaluation from the last saved model. 
* `--image_size`: Output image size (should be a power of 2). 
* `--network_capacity`: To control network capacity.
* `--results_dir`: Results directory (for testing and evaluation during training).
* `--target_hist`: Target histogram (image, npy file of target histogram, or directory of either images or histogram files). To generate a histogram of images, check `create_hist_sample.py`. 
* `--generate`: Set to True for testing. 
* `--save_noise_latent`: To save the noise and latent of current generated samples in `temp` directory (for testing).
* `--target_noise_file`: To load noise from a saved file (for testing) 
* `--target_latent_file`: To load latent from a saved file (for testing).
* `--num_image_tiles`: Number of image tiles to generate. 
* `--gpu`: CUDA device ID. 
* `--hist_bin`: Number of bins in the histogram feature. 
* `--hist_insz`: Maximum size of the image before computing the histogram feature. 
* `--hist_method`: "Counting" method used to construct histograms. Options include: `inverse-quadratic` kernel, `RBF` kernel, or `thresholding`.  
* `--hist_resizing`: If `--hist_insz` doesn't match the input image size, the image is resized based on the resizing method. Resizing options are: `interpolation` or `sampling`. 
* `--hist_sigma`: If one of the kernel methods used to compute the histogram feature (specified in `--hist_method`), this is the kernel sigma parameter. 
* `--alpha`: histogram loss scale factor (for training). 


#### Trained models
As mentioned in the paper, we trained HistoGAN on several datasets. Our pre-trained models trained using `--network_capacity = 16` and `--image_size = 256` due to hardware limitations. **Better results can be achieved by increasing the network capacity.** Here are examples of our trained models (note: these models include both generator and discriminator nets):

* [Faces](https://ln3.sync.com/dl/ef2cce1a0/c7frehvr-kaexbw44-tvupizhg-xyqjvp98) | [Google Drive mirror](https://drive.google.com/file/d/1jkJBzXsakEtuVEwQqTCBb_R_Kefa2XJ1/view?usp=sharing)
* [Cars](https://ln3.sync.com/dl/5c2bc1a60/y3sx9dnq-m5gcspa5-6zz9d4fd-7j3yemca)
* [Flowers](https://ln3.sync.com/dl/e869e6d50/pnqiaccc-vtszcvs4-tt7x2y9v-gubvyymm)
* [Anime](https://ln3.sync.com/dl/13bacb830/u4hm9bcw-hbnfeg8j-pmrwsdjp-iwar8ee8)
* [Landscape](https://ln3.sync.com/dl/dfdbb5600/r2gzhwtk-5guyb8s8-j6a99c72-excf5kjy)

* * *

### ReHistoGAN
The ReHistoGAN code and models will be available soon.



## Landscape Dataset
Our collected set of 4K landscape images is available [here](https://ln2.sync.com/dl/1891becc0/uhsxtprq-33wfwmyq-dhhqeb3s-mtstuqw7).
<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/100063922-dba70a80-2dff-11eb-9b2d-288f76122e27.jpg">
</p>



## Acknowledgement
A significant part of this code was was built on top of the [PyTorch implementation](https://github.com/lucidrains/stylegan2-pytorch) of StyleGAN by [Phil Wang](https://github.com/lucidrains).
