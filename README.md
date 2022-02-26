# HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms

[Mahmoud Afifi](https://sites.google.com/view/mafifi), 
[Marcus A. Brubaker](https://mbrubake.github.io/), 
and [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)

York University  &nbsp;&nbsp; 

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Afifi_HistoGAN_Controlling_Colors_of_GAN-Generated_and_Real_Images_via_Color_CVPR_2021_paper.pdf) | [Supplementary Materials](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Afifi_HistoGAN_Controlling_Colors_CVPR_2021_supplemental.pdf) | [Video](https://www.youtube.com/watch?v=uMN85KBV4Rw) | [Poster](https://drive.google.com/file/d/1AQNnodTUFOtTKSaXPbNEUAEbLbQNZDvE/view) | [PPT](https://drive.google.com/file/d/167rFIDyUS368yecMSXPKQYzj73pY282i/view)

![teaser](https://user-images.githubusercontent.com/37669469/100063951-e497dc00-2dff-11eb-8454-98f720fe7d04.jpg)

Reference code for the paper [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms.](https://arxiv.org/abs/2011.11731) Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. In CVPR, 2021. If you use this code or our datasets, please cite our paper:
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
* openCV
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
conda install -c conda-forge opencv
pip install CMake
pip install dlib
pip install linear-attention-transformer
pip install vector-quantize-pytorch
pip install torch-optimizer
pip install retry
```

You may face some problems in installing `dlib` on Windows via `pip`. It is required only for the face pre-processing option (see below for more details). In order to install `dlib` for Windows, please follow this [link](https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10). If couldn't install `dlib`, you can comment this [line](https://github.com/mahmoudnafifi/HistoGAN/blob/1fc4a0e0f3908ca67ea53c7b0996c28b41414e0d/utils/face_preprocessing.py#L2) and *do not* use the `--face_extraction` option for reHistoGAN. 

* * *

### Histogram loss 
We provide a [Colab notebook example](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) code to compute our histogram loss. This histogram loss is differentiable and can be easily integrated into any deep learning optimization. 

In the [Colab](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) tutorial, we provide different versions of the histogram class to compute the histogram loss for different color spaces: RGB-uv, rg-chroma, and CIE Lab. For CIE Lab, input images are supposed to be already in the CIE LAB space before computing the histogram loss. The code of these histogram classes is also provided in `./histogram_classes`. In HistoGAN and ReHistoGAN, we trained using RGB-uv histogram features. To use rg-chroma or CIE Lab, you can simply replace `from histogram_classes.RGBuvHistBlock import RGBuvHistBlock` with `from histogram_classes.X import X as RGBuvHistBlock`, where `X` is the name of the histogram class (i.e., `rgChromaHistBlock` or `LabHistBlock`). This change should be applied to all source code files that use the histogram feature. Note that for the CIE LAB histograms, you need to first convert loaded images into the CIE LAB space in the `Dataset` class in both histoGAN and ReHistoGAN codes. That also requires converting the generated images back to sRGB space before saving them.


If you faced issues with memory, please check this [issue](https://github.com/mahmoudnafifi/HistoGAN/issues/8) for potential solutions. 

* * *

### HistoGAN
To train/test a histoGAN model, use `histoGAN.py`. Trained models should be located in the `models` directory (can be changed from `--models_dir`) and each trained model's name should be a subdirectory in the `models` directory. For example, to test a model named `test_histoGAN`, you should have `models/test_histoGAN/model_X.pt` exists (where `X` refers to the last epoch number). 


#### Training
To train a histoGAN model on a dataset located at `./datasets/faces/`, use the following command:

```python histoGAN.py --name histoGAN_model --data ./datasets/faces/ --num_train_steps XX --gpu 0```

`XX` should be replaced with the number of iterations. There is no ideal number of training iterations. You may need to keep training until finds the model started to generate degraded images.


During training, you can watch example samples generated by the generator network in the results directory (specified by `--results_dir`). Each column in the generated sample images shares the same training histogram feature. Shown below is the training progress of a HistoGAN trained on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset using `--network_capacity 16` and `--image_size 256`.

<p align="center">
  <img width = 25% src="https://user-images.githubusercontent.com/37669469/128811430-ce7c58d1-d439-4b05-a774-488f4d2e7bfd.gif">
</p>

There is no clear criterion to stop training, so watching generated samples will help to detect when the generator network starts diverging. Also reporting the FID score after each checkpoint may help. 

You may need to increase the number of training steps (specified by `--num_train_steps`), if the generator didn't diverge by the end of training. If the network starts generating degraded results after a short period of training, you may need to apply some data augmentation by using `--aug_prob X` and `--dataset_aug_prob Y`, where `X` is a float number representing the probability of discriminator augmentation and `Y` float number to set the probability of dataset augmentation. 

Shown below is the training progress of HistoGAN when trained on portrait images with and without augmentation applied. As shown, the generator starts to generate degraded images after a short period of training, while it keeps generating reasonable results when data augmentation is applied (here, we use `--aug_prob 0.5`). 


<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/128811547-8f33511f-fe29-492d-bb64-ea9c4ceeabe1.gif">
</p>

  

#### Testing
Here is an example of how to generate new samples of a trained histoGAN models named *Faces_histoGAN*:

```python histoGAN.py --name Faces_histoGAN --generate True --target_his ./target_images/2.jpg --gpu 0```

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
* `--new`: Set to `True` to train a new model. If `--new = False`, it will start training/evaluation from the last saved model. 
* `--image_size`: Output image size (should be a power of 2). 
* `--batch_size` and `--gradient_accumulate_every`: To control the size of mini-batch and the accumulation in computing the gradient. 
* `--network_capacity`: To control network capacity.
* `--attn_layers`: To add a self-attention to the designated layer(s) of the discriminator. For example, if you would like to add a self-attention layer after the output of the 1st and 2nd layers, use `--attn_layers 1,2`. In our training, we did not use any attention layers, but it could improve the results if added. 
* `--results_dir`: Results directory (for testing and evaluation during training).
* `--target_hist`: Target histogram (image, npy file of target histogram, or directory of either images or histogram files). To generate a histogram of images, check [create_hist_sample.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/create_hist_sample.py). 
* `--generate`: Set to `True` for testing. 
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
* `--alpha`: Histogram loss scale factor (training).


#### Projection
Recoloring a given input image could be achieved if we can find a matching input, in the latent feature space, that can produce a similar image to the given input image. That is, we can optimize the input to our generator network, such that the generated image looks similar to the given input image. Once we have that done, we can manipulate this image by feeding our network a different histogram feature. 

<p align="center">
  <img width = 45% src="https://user-images.githubusercontent.com/37669469/128960458-247bd33f-02c4-4dc9-ad4b-afdb619d5db8.gif">
</p>

Here, we provide two options for this instance-optimization: (1) optimizing input Gaussian style vectors and (2) optimizing style vectors after the "to_latent" projection. The figure below shows the details of HistoGAN's first and last two blocks. The first optimization option optimizes input noise style vectors that feed the first blocks of HistoGAN from the left side. This optimization aims to minimize the differences between generated and the given input images (i.e., target image), while the second optimization option optimizes the style input in the latent space of each of the first blocks in HistoGAN. In both options, we do not touch the style input of the last two blocks of HistoGAN as these blocks get their input from the histogram feature, which is generated from the target image's colors. 

<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/128960884-aa62ee68-6413-490f-8381-e3f7e40f9b3c.jpg">
</p>

The provided code allows you also to optimize input noise (non-style noise), that feeds the right part of each block in the shown figure above, either in Gaussian space or in the latent space (i.e., after the "to_latent" projection). 

Let's suppose that our input image is `./input_images/41.jpg` and our pre-trained HistoGAN models for faces is named `histoGAN_model`. 

To optimize input Gaussian style vectors, use the following command:

`python projection_gaussian.py --name histoGAN_model --input_image ./input_images/41.jpg --gpu 0`


<p align="center">
  <img width = 35% src="https://user-images.githubusercontent.com/37669469/129124979-6a0711f8-2ecb-441e-9113-dab00ac09556.gif">
</p>

The final projected image and optimized style will be saved in `XX/histoGAN_model/41`, where `XX` is the result directory specified by `--results_dir`. To recolor the image after optimization with the colors of the image in `./target_images/1.jpg`, you can use this command:


`python projection_gaussian.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --gpu 0`


<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129124486-b84dc9c2-1838-4e41-97c2-e46fe6f9f8fd.jpg">
</p>


The generated image share a similar appearance with the input image, but it is for a different person! We can apply a simple post-processing upsampling step to pass the colors from the generated image to our input image:


`python projection_gaussian.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --upsampling_output True --gpu 0`



<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129124497-c5759ff7-6dcd-4c2f-adb8-69544b151ac7.jpg">
</p>



To adjust other styles than image colors (for example, the style vector of the fourth and fifth block in the generator), use this command:

`python projection_gaussian.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --random_styles 4 5 --gpu 0`



<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129124501-9e476d87-5ed0-41fa-a676-94a63121a047.jpg">
</p>


To optimize after the "to_latent" projection, use:

`python projection_to_latent.py --name histoGAN_model --input_image ./input_images/41.jpg --gpu 0`

<p align="center">
  <img width = 35% src="https://user-images.githubusercontent.com/37669469/129122422-7e7105e6-de4f-4c7b-be62-8701a1330be7.gif">
</p>


Similarly, the final projected image and optimized style will be saved in `XX/histoGAN_model/41`, where `XX` is the result directory specified by `--results_dir`. For recoloring, use:

`python projection_to_latent.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --upsampling_output True --gpu 0`

<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129122856-751ed483-2c5d-42a6-a3f4-02874457a2cb.jpg">
</p>


To apply a post-processing upsampling, use this command:

`python projection_to_latent.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --upsampling_output True --gpu 0`

<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129122954-c24d4964-3e4b-435b-a3af-1d4a4af33529.jpg">
</p>

To adjust other styles than image colors (for example, the style vector of the fifth block in the generator), use this command:

`python projection_to_latent.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --random_styles 5 --gpu 0`

<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129123027-db81ce52-9473-4f17-ab39-2f6f9ab55b93.jpg">
</p>


Here we randomize new styles for the fourth and fifth blocks in the generator network:

`python projection_to_latent.py --name histoGAN_model --input_image ./input_images/41.jpg --generate True --target_hist ./target_images/1.jpg --random_styles 4 5 --gpu 0`

<p align="center">
  <img width = 50% src="https://user-images.githubusercontent.com/37669469/129123100-0c0181aa-c94a-4c74-9786-ff4459600dcb.jpg">
</p>







In addition to HistoGAN parameters mentioned above, there are some additional parameters for the optimization code:

* `--input_image`: Path of the input image to optimize for.
* `--latent_noise`: To optimize the input noise (non-style noise) after the "to_latent" projection. The default value is `False`.
* `--optimize_noise`: To optimize the input noise (non-style noise) in Gaussian space. At inference time (i.e., after finishing the optimization step), you can set this parameter to `True` (even if wasn't used during optimization) to load the same input noise used during optimization. The default value is `False`.
* `--pixel_loss`: Reconstruction loss; this can be either: `L1` or `L2`. 
* `--pixel_loss_weight`: Scale factor of reconstruction loss to control the contribution of this loss term. 
* `--vgg_loss_weight`: In addition to the reconstruction loss, you can use VGG loss (AKA perceptual loss) by setting the value of this parameter to any value larger than zero. 
* `--generate`: To generate an output of the optimized style/noise input. This can be set to `True` *only* after finishing the optimization. 
* `--target_hist`: To use a new target histogram after optimization. This could be: an image, a npy file of the target histogram, or a directory of either images or npy histogram files.
* `--add_noise`: At inference time (i.e., with `--generate = True`), this option to add random noise to the saved/optimized non-style noise. The default value is `False`.
* `--random_styles`: A list of histoGAN's blocks that you would like to ignore the values of their optimized style vectors and randomize new style vectors for those blocks. For example, for the first three blocks in the generator network, use `--random_styles 1 2 3`. This is only used in the testing phase. The default value is `[]`. 
* `--style_reg_weight`: L2 regularization factor for the optimized style vectors. 
* `--noise_reg_weight`: If optimizing either Gaussian or the `to_latent` non-style noise, this is a scale factor of L2 regularization for the optimized noise.
* `--save_every`: To specify number of optimization steps for saving the output of current input during optimization. 
* `--post_recoloring`: To apply a post-processing color transfer that maps colors of the original input image to those in the generated image. Here, we use the [colour transfer algorithm based on linear Monge-Kantorovitch solution](https://ieeexplore.ieee.org/abstract/document/4454269). This option is recommended if the recolored images have some artifacts. Also, this is helpful to get the output image in the same resolution as the input image. Comparing with transferring colors of target histogram colors directly to the input image, this post recoloring was found to give better results as mentioned in the [supplementary materials](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Afifi_HistoGAN_Controlling_Colors_CVPR_2021_supplemental.pdf). You can replace the code of this method to use a recent color transfer method. 
* `--upsampling_output`: If the network produces a low-resolution image, while the input image is a high-resolution one, you may need to upsampling the output image at inference. We have two options for upsampling that can be specified by `--upsampling_method`.
* `--upsampling_method`: If `--upsampling_output` is set to `True`, then this argument specifies the upsampling method. We have two options, which are: `BGU` or `pyramid`. The `BGU` option applies the [bilateral guided upsampling method](https://github.com/google/bgu) for upsampling. The Matlab source code of BGU was cloned in this repo in the `./upsampling` directory. You can either build `./upsampling/BGU.m` or use the provided `BGU.exe`, which was built for Windows (make sure that you have [Matlab Runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html) installed for Matlab 2019b or higher). If you are going to rebuild `BGU.exe`, make sure to locate the new exe file in the root before selecting `BGU` as the `--upsampling_method`. The second option is `pyramid`, which simply swaps some of the lowest layers of the Laplacian pyramid of input and generated images.
* `--pyramid_levels`: If `--upsampling_method` is `pyramid`, then this is the number of levels in the Laplacian pyramid.
* `--swapping_levels`: The number of lowest levels to swap. For instance, if `--swapping_levels 2` is used, then the last two layers in the pyramid of input and generated images will get swapped.
* `--level_blending`: If `--upsampling_method` is `pyramid`, setting `--level_blending` to `True` will blend between the remaining pyramid levels. 


Here is a qualitative comparison of using the BGU vs the pyramid swapping. 

![histGAN_upsampling](https://user-images.githubusercontent.com/37669469/128967412-858533ba-5178-4798-96ba-e05c719fb3cf.jpg)



Note that you may need to play with the optimization settings (e.g., `--learning_rate`, `--style_reg_weight`, etc.) to get better results. For face images, it is preferred to use the same settings that were used to prepare the training data ([FFHQ](https://github.com/NVlabs/ffhq-dataset)). To crop face region, you can use [face_preprocessing.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/utils/face_preprocessing.py). If the recolored image suffers from artifacts, you may try to use `--post_recoloring` or use reHistoGAN. 





#### Trained models
As mentioned in the paper, we trained HistoGAN on several datasets. Most of our pre-trained models were trained using `--network_capacity = 16` and `--image_size = 256` due to hardware limitations. **Better results can be achieved by increasing the network capacity and using attention layers (`--attn_layers`).** Here are examples of our trained models (note: these models include both generator and discriminator nets):

* [Faces](https://ln3.sync.com/dl/ef2cce1a0/c7frehvr-kaexbw44-tvupizhg-xyqjvp98) | [Google Drive mirror](https://drive.google.com/file/d/1jkJBzXsakEtuVEwQqTCBb_R_Kefa2XJ1/view?usp=sharing)
* [Faces_20](https://ln4.sync.com/dl/49c229e50/fr2stver-g36e3km7-qj3vwwtf-qwqa5ufb)
* [Cars](https://ln3.sync.com/dl/5c2bc1a60/y3sx9dnq-m5gcspa5-6zz9d4fd-7j3yemca)
* [Flowers](https://ln3.sync.com/dl/e869e6d50/pnqiaccc-vtszcvs4-tt7x2y9v-gubvyymm)
* [Anime](https://ln3.sync.com/dl/13bacb830/u4hm9bcw-hbnfeg8j-pmrwsdjp-iwar8ee8)
* [Landscape](https://ln3.sync.com/dl/dfdbb5600/r2gzhwtk-5guyb8s8-j6a99c72-excf5kjy)
* [PortraitFaces](https://ln4.sync.com/dl/f3b08fed0/zdqb9qxe-4wrasq8r-7ihvp73m-zv82vat5)
* [PortraitFaces_aug](https://ln4.sync.com/dl/f061af1b0/t7ukam9v-nk6napgw-krpn2wa9-h2agy8in)
* [PortraitFaces_20_aug](https://ln4.sync.com/dl/f8e934db0/9bddn4cf-fnhtvkwi-jg4zmpb4-dq8ypv4r)


For model names that include `_20`, use  `--network_capacity 20` in testing. If the model name includes `_aug`, make sure to set `--aug_prob` to any value that is greater than zero. Below are examples of generated samples from each model. Each shown group of generated images share the same histogram feature. 


![pre-trained](https://user-images.githubusercontent.com/37669469/128978234-ffad52f7-3fd6-42e2-8269-65e17f7e05a0.gif)


* * *



### ReHistoGAN

ReHistoGAN is an extension of our HistoGAN to recolor an input image through an encoder-decoder network. This network employs our histoGAN's head (i.e., the last two blocks) in its decoder. 

<p align="center">
  <img width = 60% src="https://user-images.githubusercontent.com/37669469/129070204-7131375c-43ed-4466-a8b0-d7b1e2b491e6.jpg">
</p>


#### Training
To train a reHistoGAN model on a dataset located at `./datasets/faces/`, use the following command:

```python rehistoGAN.py --name reHistoGAN_model --data ./datasets/faces/ --num_train_steps XX --gpu 0```

`XX` should be replaced with the number of iterations. There is no ideal number of training iterations. You may need to keep training until the model starts to produce degraded images.

To use the weights of a pre-trained HistoGAN model located in `./models/histoGAN_model` to initialize the histoGAN's head in the reHistoGAN model, use the following command: 

```python rehistoGAN.py --name reHistoGAN_model --data ./datasets/faces/ --num_train_steps XX --histGAN_models_dir ./models --histoGAN_model_name histoGAN_model --load_histoGAN_weights True --gpu 0```



During training, you can watch example samples generated by the generator network in the results directory (specified by `--results_dir`).


#### Testing
To use a pre-trained reHistoGAN model (named, for example, `reHistoGAN_model`) to recolor an input image located in `./input_images/1.jpg` using the histogram of a target image located in `./target_images/1.jpg`, use the following command:

```python rehistoGAN.py --name reHistoGAN_model --generate True --input_image ./input_images/1.jpg --target_hist ./target_images/1.jpg --gpu 0```

Note that you can specify a target histogram feature instead of the image itself by first generating the histogram feature using [create_hist_sample.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/create_hist_sample.py), then you can set `--target_hist` to the path of the generated histogram feature.

To get an output image in the same resolution as the input image, we provide two options for upsampling (see the below parameters for more information). Here is an example of using the post-processing upsampling:


```python rehistoGAN.py --name reHistoGAN_model --generate True --input_image ./input_images/1.jpg --target_hist ./target_images/1.jpg --upsampling_output True --gpu 0```


<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/129094018-ebc695b3-0882-468b-9345-ebdb785a9a71.jpg">
</p>





Instead of processing a single input image, you can process all images located in a directory. Let's assume input images are located in `./input_images`, use the following command to process all images in this directory. 

```python rehistoGAN.py --name reHistoGAN_model --generate True --input_image ./input_images/ --target_hist ./target_images/1.jpg --upsampling_output True --gpu 0```

Similarly, you can specify a directory of target images (or npy histogram files) as shown below:

```python rehistoGAN.py --name reHistoGAN_model --generate True --input_image ./input_images/ --target_hist ./target_images/ --upsampling_output True --gpu 0```


![rehistogan](https://user-images.githubusercontent.com/37669469/128979200-b2b7441f-1cf8-4ade-b138-d0489a843920.gif)


For auto recoloring, you should first generate the target histogram set to sample from. This can be done using [create_hist_data.py.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/create_hist_data.py), which generate this set for you from all images located in `./histogram_data/`. Then, to generate new recolored images of an input image located in `./input_images/55.jpg`, for example, use this command:

```python rehistoGAN.py --name reHistoGAN_model --generate True --input_image ./input_images/55.jpg --upsampling_output True --sampling True --gpu 0```

This will generate `XX` recolored images for you, where `XX` can be specify using `--target_number`. 


<p align="center">
  <img width = 40% src="https://user-images.githubusercontent.com/37669469/129952110-44c3508f-c6d5-4860-bf51-34e72d3409e1.gif">
</p>


For face images, it is preferred to use the same settings that were used to prepare the training data ([FFHQ](https://github.com/NVlabs/ffhq-dataset)). To crop face region, you can use `--face_extraction True`.

#### Universal ReHistoGAN
As the case of most GAN methods, our reHistoGAN targets a specific object domain to achieve the image recoloring task. This restriction may hinder the generalization
of our method to deal with images taken from arbitrary domains. To deal with that, we collected images from a different domain, aiming to represent the "universal" object
domain (see the [supplemental materials](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Afifi_HistoGAN_Controlling_Colors_CVPR_2021_supplemental.pdf) for more details). To train our reHistoGAN on this "universal" object domain, we used `--network_capacity 18` without any further changes in the original architecture. 

To test one of our pre-trained "universal" reHistoGAN models (for example, `Universal_rehistoGAN_v0`), use the following command:

```python rehistoGAN.py --name Universal_rehistoGAN_v0 --generate True --input_image ./input_images/other-objects/ --target_hist ./target_images/ --upsampling_output True --network_capacity 18 --gpu 0```

<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/129090905-f274b247-5298-47c8-87cd-1b22f6680b1a.gif">
</p>

Again, you can either recolor all images in a directory or apply the recoloring to a single image. You also can recolor input image(s) with a single target image (or a histogram feature) or apply auto recoloring by sampling from a pre-defined set of histograms. 

If recolored images by any model have undesirable artifacts or color bleeding, you can try the `--post_recoloring` option to mitigate such artifacts.


#### Parameters
ReHistoGAN shares the same parameters of HistoGAN in addition to some extra parameters, such as:
* `--load_histoGAN_weights`: To use pre-trained HistoGAN weights instead of training from scratch. This is only for the weights of the histoGAN's head. 
* `--histoGAN_model_name`: If `--load_histoGAN_weights` is `True`, then this is the name of the pre-trained HistoGAN model.
* `--histGAN_models_dir`: If a pre-trained weights used for the histoGAN's head, then this is the directory of the pre-trained HistoGAN model. 
* `--sampling`: To auto recolor input image(s). If `--sampling` is set to `True`, make sure to set `--target_hist` to `None`. Then, it is supposed to have a pre-computed set of target histograms to sample from. This set should be located in `histogram_data/histograms.npy`. To generate this histogram set from a new image set, copy your images into `./histogram_data`, then run [create_hist_data.py.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/create_hist_data.py).
* `--target_number`: If `--sampling` is `True`, then this is the number of output recolored images for each single input image. 
* `--alpha`: Histogram loss scale factor (training).
* `--beta`: Reconstruction loss scale factor (training).
* `--gamma`: Discriminator loss scale factor (training).
* `--change_hyperparameters`: To change the value of `--alpha`, `--beta`, `--gamma` after `X` training steps, where `X` can be specified using `--change_hyperparameters_after`. If `--change_hyperparameters` is `True`, the new values of  `--alpha`, `--beta`, `--gamma` (after the first `X` training steps) can be specificed from [here](https://github.com/mahmoudnafifi/HistoGAN/blob/43ed585dc147309812bc6c4656caff1df5916a63/ReHistoGAN/rehistoGAN.py#L900).
* `--rec_loss`: Reconstruction loss options, which includes: `sobel` or `laplacian` (default). 
* `--variance_loss`: To use variance loss (Equation 9 in the [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Afifi_HistoGAN_Controlling_Colors_of_GAN-Generated_and_Real_Images_via_Color_CVPR_2021_paper.pdf)). 
* `--internal_hist`: Internal histogram injection. This was an ablation on a different design of reHistoGAN, but we did not use it the official reHistoGAN. The default value is `False`.
* `--skip_conn_to_GAN`: To use skip connections in reHistoGAN (see Figures 4 and 6 in the [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Afifi_HistoGAN_Controlling_Colors_of_GAN-Generated_and_Real_Images_via_Color_CVPR_2021_paper.pdf)). The default value is `True`. 
* `--fixed_gan_weights`: To *not* update the weights of the histoGAN's head during training. The default value is `False`.
* `--initialize_gan`: To initialize weights of reHistoGAN. This does not affect loading pre-trained histoGAN's weights if `--load_histoGAN_weights` is `True`.
* `--post_recoloring`: To apply a post-processing color transfer that maps colors of the original input image to those in the recolored image. Here, we use the [colour transfer algorithm based on linear Monge-Kantorovitch solution](https://ieeexplore.ieee.org/abstract/document/4454269). This option is recommended if the recolored images have some artifacts. Also, this is helpful to get the output image in the same resolution as the input image. Comparing with transferring colors of target histogram colors directly to the input image, this post recoloring was found to give better results as mentioned in the [supplementary materials](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Afifi_HistoGAN_Controlling_Colors_CVPR_2021_supplemental.pdf). You can replace the code of this method to use a recent color transfer method. 
* `--upsampling_output`: If the network produces a low-resolution image, while the input image is a high-resolution one, you may need to upsampling the output image at inference. We have two options for upsampling that can be specified by `--upsampling_method`.
* `--upsampling_method`: If `--upsampling_output` is set to `True`, then this argument specifies the upsampling method. We have two options, which are: `BGU` or `pyramid`. The `BGU` option applies the [bilateral guided upsampling method](https://github.com/google/bgu) for upsampling. The Matlab source code of BGU was cloned in this repo in the `./upsampling` directory. You can either build `./upsampling/BGU.m` or use the provided `BGU.exe`, which was built for Windows (make sure that you have [Matlab Runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html) installed for Matlab 2019b or higher). If you are going to rebuild `BGU.exe`, make sure to locate the new exe file in the root before selecting `BGU` as the `--upsampling_method`. The second option is `pyramid`, which simply swaps some of the lowest layers of the Laplacian pyramid of input and generated images.
* `--pyramid_levels`: If `--upsampling_method` is `pyramid`, then this is the number of levels in the Laplacian pyramid.
* `--swapping_levels`: The number of lowest levels to swap. For instance, if `--swapping_levels 2` is used, then the last two layers in the pyramid of input and generated images will get swapped.
* `--level_blending`: If `--upsampling_method` is `pyramid`, setting `--level_blending` to `True` will blend between the remaining pyramid levels. 
* `--face_extraction`: In testing, to pre-process input face images in the same way used to prepare the training data ([FFHQ](https://github.com/NVlabs/ffhq-dataset)), use `--face_extraction True`. Make sure that `dlib` is successfully installed. 


Here are some useful parameters similar to those for HistoGAN: 

* `--models_dir`: Directory to save rehistoGAN models.
* `--new`: Set to `True` to train a new model. If `--new = False`, it will start training/evaluation from the last saved model. 
* `--target_hist`: Target histogram (image, npy file of target histogram, or directory of either images or histogram files). To generate a histogram of images, check [create_hist_sample.py](https://github.com/mahmoudnafifi/HistoGAN/blob/master/create_hist_sample.py). 
* `--attn_layers`: To add a self-attention to the designated layer(s) of the discriminator. For example, if you would like to add a self-attention layer after the output of the 1st and 2nd layers, use `--attn_layers 1,2`. In our training, we did not use any attention layers, but it could improve the results if added. 
* `--generate`: Set to `True` for recoloring input image(s).
* `--network_capacity`: To control network capacity. In our pre-trained models, we used `--network_capacity 16`. For "universal" reHistoGAN models, we used `--network_capacity 18`. The default value is 16. 
* `--image_size`: Image size (should be a power of 2). 



#### Trained models
Here are some of our pre-trained models for face image recoloring in addition to our `universal` reHistoGAN. As it is totally subjective, we provide different versions of our trained models (each of which was either trained using a different number of iterations, different loss weights, or `--rec_loss` options). 

* [Faces model-0](https://ln4.sync.com/dl/521ea0f30/gtcq4jy4-736tvuxw-zti2s43q-wyv5wftx)
* [Faces model-1](https://ln4.sync.com/dl/de915df90/nfcy46i3-9xhzvtjn-x3barman-gbupfn3g)
* [Faces model-2](https://ln4.sync.com/dl/05a243a50/6sun3qv9-arcjqy3f-74acs92n-vcfwjpms)
* [Faces model-3](https://ln4.sync.com/dl/cf5b1f5f0/tynvcfey-v5fz9uk3-7reugdyx-h8y93rwg)
* [Universal model-0](https://ln4.sync.com/dl/66e969d90/b2sd68zx-bhrbj7ws-cjxq36u6-mbzbrg24)
* [Universal model-1](https://ln4.sync.com/dl/3bd657670/t5hfd86w-grnb794m-izhsvgyv-zwwzjei2)
* [Universal model-2](https://ln4.sync.com/dl/7d31a84c0/9na2sp3y-dt4n55eq-3k84ddvs-zd37eeh9)

Here we show qualitative comparisons between recoloring results of [Faces model-0](https://ln4.sync.com/dl/521ea0f30/gtcq4jy4-736tvuxw-zti2s43q-wyv5wftx) and [Faces model-1](https://ln4.sync.com/dl/de915df90/nfcy46i3-9xhzvtjn-x3barman-gbupfn3g). As shown, [Faces model-0](https://ln4.sync.com/dl/521ea0f30/gtcq4jy4-736tvuxw-zti2s43q-wyv5wftx) tends to produce less artifacts compared to [Faces model-1](https://ln4.sync.com/dl/de915df90/nfcy46i3-9xhzvtjn-x3barman-gbupfn3g), but [Faces model-1](https://ln4.sync.com/dl/de915df90/nfcy46i3-9xhzvtjn-x3barman-gbupfn3g) catches the colors in the target histogram in a better way than [Faces model-0](https://ln4.sync.com/dl/521ea0f30/gtcq4jy4-736tvuxw-zti2s43q-wyv5wftx).

![rehistogan_2](https://user-images.githubusercontent.com/37669469/129077281-d7637b11-1c20-4c4d-927d-915167f561f0.gif)



## Landscape Dataset
Our collected set of 4K landscape images is available [here](https://ln2.sync.com/dl/1891becc0/uhsxtprq-33wfwmyq-dhhqeb3s-mtstuqw7).
<p align="center">
  <img width = 100% src="https://user-images.githubusercontent.com/37669469/100063922-dba70a80-2dff-11eb-9b2d-288f76122e27.jpg">
</p>


## Portrait Dataset
We have extracted ~7,000 portrait face images from the [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) dataset. You can download this portrait set from [here](https://ln4.sync.com/dl/d7addacf0/b978wvm4-9dndxvh6-hc4ss39y-5hpck6si). If you use this dataset, please cite our paper in addition to the [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset) dataset. This set is provided only for non-commercial research purpose.
The images in the WikiArt dataset were obtained from [WikiArt.org](https://www.wikiart.org). By using this set, you agree to obey the [terms and conditions of WikiArt.org](https://www.wikiart.org/en/terms-of-use).

![preprocessed_faces](https://user-images.githubusercontent.com/37669469/128927098-242d7301-cefa-4225-a380-0f8579828c39.jpg)


## Acknowledgement
A significant part of this code was was built on top of the [PyTorch implementation](https://github.com/lucidrains/stylegan2-pytorch) of StyleGAN by [Phil Wang](https://github.com/lucidrains).



## Related Research Projects
- [Image Recoloring Based on Object Color Distributions](https://github.com/mahmoudnafifi/Image_recoloring): A method to perform automatic image recoloring based on the distribution of colors associated with objects present in an image.
- [CAMS](https://github.com/mahmoudnafifi/color-aware-style-transfer): A color-aware multi-style transfer method.
