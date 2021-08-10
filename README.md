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
* dlib (optional)

Conda & pip commands:
```
conda create -n histoGAN python=3.6 numpy=1.13.3 scipy 
conda activate histoGAN
conda install pytorch torchvision -c python
conda install -c conda-forge tqdm
conda install -c anaconda pillow
pip install CMake
pip install dlib
pip install linear-attention-transformer
pip install vector-quantize-pytorch
pip install torch-optimizer
pip install retry
```

You may face some problems in installing `dlib` on Windows via `pip`. It is required only for the face pre-processing option (see below for more details). In order to install `dlib` for Windows, please follow this [link](https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10). If couldn't install `dlib`, you can comment this [line](https://github.com/mahmoudnafifi/HistoGAN/blob/1fc4a0e0f3908ca67ea53c7b0996c28b41414e0d/utils/face_preprocessing.py#L2) and do not use the `--face_extraction` option for ReHistoGAN. 

* * *

### Histogram loss 
We provide a [Colab notebook example](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) code to compute our histogram loss. This histogram loss is differentiable and can be easily integrated into any deep learning optimization. 

In the [Colab](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) tutorial, we provide different versions of the histogram class to compute histogram loss for different color spaces: RGB-uv, rg-chroma, and CIE Lab. For CIE Lab, input images are supposed to be already in CIE LAB space before computing the loss. The code of these histogram classes is also provided in `./histogram_classes`. In HistoGAN and ReHistoGAN, we trained using RGB-uv histogram features. To use rg-chroma or CIE Lab, you simply replace this import `from histogram_classes.RGBuvHistBlock import RGBuvHistBlock` with `from histogram_classes.X import X as RGBuvHistBlock`, where `X` is the name of the histogram class. Note that for the CIE LAB histograms, you need to first convert loaded images into the CIE LAB space in the `Dataset` class in both histoGAN and ReHistoGAN codes. That also requires converting the generated images back to sRGB space before saving them.


If you faced issues with memory, please check this [issue](https://github.com/mahmoudnafifi/HistoGAN/issues/8) for potential solutions. 

* * *

### HistoGAN
To train/test a histoGAN model, use `histoGAN.py`. Trained models should be located in the `models` directory and each trained model's name should be a subdirectory in the `models` directory. For example, to test a model named `test_histoGAN`, you should have `models/test_histoGAN/model_X.pt` exists (where `X` refers to the epoch number). 


#### Training
To train a histoGAN model on a dataset located at `./datasets/faces/` for example:

```python histoGAN.py --name histoGAN_model --data ./datasets/faces/ --num_train_steps XX --gpu 0```

`XX` should be replaced with the number of iterations. 

During training, you can watch example samples generated by the generator network in the results directory (specified by `--results_dir`). Each column in the generated sample images shares the same training histogram feature. Shown below is the training progress of a HistoGAN trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset using `--network_capacity 16` and `--image_size 256`.

<p align="center">
  <img width = 25% src="https://user-images.githubusercontent.com/37669469/128811430-ce7c58d1-d439-4b05-a774-488f4d2e7bfd.gif">
</p>

There is no clear criterion to stop training, so watching generated samples will help to detect when the generator network starts diverging. Also reporting the FID score after each checkpoint may help. 

You may need to increase the number of training steps (specified by `--num_train_steps`), if the generator didn't diverge by the end of training. If the network starts generating degraded results after a short period of training, you may either want to reduce the network capacity (specified by `--network_capacity`) or use data augmentation by using `--aug_prob` value that is higher than 0. 

Shown below is the training progress of HistoGAN when trained on portrait images with and without augmentation applied. As shown, the generator starts to generate degraded images after a short training period, while it keeps generating reasonable results when augmentation is applied. 


<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/128811547-8f33511f-fe29-492d-bb64-ea9c4ceeabe1.gif">
</p>

  

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
* `--batch_size` and `--gradient_accumulate_every`: To control the size of mini-batch and the accumulation in computing the gradient. 
* `--network_capacity`: To control network capacity.
* `--attn_layers`: To add a self-attention to the designated layer(s) of the discriminator (and the corresponding layer(s) of the generator). For example, if you would like to add a self-attention layer after the output of the 1st and 2nd layers, use `--attn_layers 1,2`. In our training, we did not use any attention layers, but it could improve the results if added. 
* `--results_dir`: Results directory (for testing and evaluation during training).
* `--target_hist`: Target histogram (image, npy file of target histogram, or directory of either images or histogram files). To generate a histogram of images, check `create_hist_sample.py`. 
* `--generate`: Set to True for testing. 
* `--save_noise_latent`: To save the noise and latent of current generated samples in `temp` directory (for testing).
* `--target_noise_file`: To load noise from a saved file (for testing) 
* `--target_latent_file`: To load latent from a saved file (for testing).
* `--num_image_tiles`: Number of image tiles to generate. 
* `--gpu`: CUDA device ID.
* `--aug_types`: Options include: `translation`, `cutout`, and `color`. Example: `--aug_types translation cutout`.
* `--dataset_aug_prob`: Probability of dataset augmentation: applies random cropping
* `--aug_prob`: Probability of discriminator augmentation. It applies operations specified in `--aug_types`. Note that if you use `--aug_prob > 0.0` to train the model, you should use `--aug_prob > 0.0` in testing as well to work properly.
* `--hist_bin`: Number of bins in the histogram feature. 
* `--hist_insz`: Maximum size of the image before computing the histogram feature. 
* `--hist_method`: "Counting" method used to construct histograms. Options include: `inverse-quadratic` kernel, `RBF` kernel, or `thresholding`.  
* `--hist_resizing`: If `--hist_insz` doesn't match the input image size, the image is resized based on the resizing method. Resizing options are: `interpolation` or `sampling`. 
* `--hist_sigma`: If one of the kernel methods used to compute the histogram feature (specified in `--hist_method`), this is the kernel sigma parameter. 
* `--alpha`: histogram loss scale factor (for training).


#### Projection
Very soon!

#### Trained models
As mentioned in the paper, we trained HistoGAN on several datasets. Our pre-trained models were trained using `--network_capacity = 16` and `--image_size = 256` due to hardware limitations. **Better results can be achieved by increasing the network capacity and using attention layers (`--attn_layers`).** Here are examples of our trained models (note: these models include both generator and discriminator nets):

* [Faces](https://ln3.sync.com/dl/ef2cce1a0/c7frehvr-kaexbw44-tvupizhg-xyqjvp98) | [Google Drive mirror](https://drive.google.com/file/d/1jkJBzXsakEtuVEwQqTCBb_R_Kefa2XJ1/view?usp=sharing)
* [Faces_20](https://ln4.sync.com/dl/49c229e50/fr2stver-g36e3km7-qj3vwwtf-qwqa5ufb)
* [Cars](https://ln3.sync.com/dl/5c2bc1a60/y3sx9dnq-m5gcspa5-6zz9d4fd-7j3yemca)
* [Flowers](https://ln3.sync.com/dl/e869e6d50/pnqiaccc-vtszcvs4-tt7x2y9v-gubvyymm)
* [Anime](https://ln3.sync.com/dl/13bacb830/u4hm9bcw-hbnfeg8j-pmrwsdjp-iwar8ee8)
* [Landscape](https://ln3.sync.com/dl/dfdbb5600/r2gzhwtk-5guyb8s8-j6a99c72-excf5kjy)
* [PortraitFaces](https://ln4.sync.com/dl/f3b08fed0/zdqb9qxe-4wrasq8r-7ihvp73m-zv82vat5)
* [PortraitFaces_aug](https://ln4.sync.com/dl/f061af1b0/t7ukam9v-nk6napgw-krpn2wa9-h2agy8in)
* [PortraitFaces_20_aug](https://ln4.sync.com/dl/f8e934db0/9bddn4cf-fnhtvkwi-jg4zmpb4-dq8ypv4r)


Note that for model names that include `_20`, use  `--network_capacity 20` in testing. If the model name includes `_aug`, make sure to set `--aug_prob` to any value higher than zero. Below are examples of generated samples from each model. Each shown group of generated images share the same histogram feature. 


![pre-trained](https://user-images.githubusercontent.com/37669469/128905351-83b145a9-d045-47b0-835d-05974ef4106d.gif)


* * *



### ReHistoGAN

#### Training
Very soon!

#### Testing
Very soon!

#### Universal ReHistoGAN
Very soon!

#### Trained models
* [Faces model-0](https://ln4.sync.com/dl/521ea0f30/gtcq4jy4-736tvuxw-zti2s43q-wyv5wftx)
* [Faces model-1](https://ln4.sync.com/dl/de915df90/nfcy46i3-9xhzvtjn-x3barman-gbupfn3g)
* [Faces model-2](https://ln4.sync.com/dl/05a243a50/6sun3qv9-arcjqy3f-74acs92n-vcfwjpms)
* [Faces model-3](https://ln4.sync.com/dl/cf5b1f5f0/tynvcfey-v5fz9uk3-7reugdyx-h8y93rwg)
* [Universal model-0](https://ln4.sync.com/dl/66e969d90/b2sd68zx-bhrbj7ws-cjxq36u6-mbzbrg24)
* [Universal model-1](https://ln4.sync.com/dl/3bd657670/t5hfd86w-grnb794m-izhsvgyv-zwwzjei2)
* [Universal model-2](https://ln4.sync.com/dl/7d31a84c0/9na2sp3y-dt4n55eq-3k84ddvs-zd37eeh9)


The ReHistoGAN code will be available very soon.



## Landscape Dataset
Our collected set of 4K landscape images is available [here](https://ln2.sync.com/dl/1891becc0/uhsxtprq-33wfwmyq-dhhqeb3s-mtstuqw7).
<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/100063922-dba70a80-2dff-11eb-9b2d-288f76122e27.jpg">
</p>


## Portrait Dataset
We have extracted face images from the [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) dataset. This set includes ~7,000 portrait face images. You can download our processed portrait set from [here](https://ln4.sync.com/dl/d7addacf0/b978wvm4-9dndxvh6-hc4ss39y-5hpck6si). If you use this dataset, please cite our paper in addition to the [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) dataset. This set is provided only for non-commercial research purpose.
The images in the WikiArt dataset were obtained from [WikiArt.org](https://www.wikiart.org/en/terms-of-use). By using this set, you agree to obey the terms and conditions of [WikiArt.org](https://www.wikiart.org/en/terms-of-use).

![preprocessed_faces](https://user-images.githubusercontent.com/37669469/128927098-242d7301-cefa-4225-a380-0f8579828c39.jpg)


## Acknowledgement
A significant part of this code was was built on top of the [PyTorch implementation](https://github.com/lucidrains/stylegan2-pytorch) of StyleGAN by [Phil Wang](https://github.com/lucidrains).
