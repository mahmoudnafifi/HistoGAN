# HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms

[Mahmoud Afifi](https://sites.google.com/view/mafifi), 
[Marcus A. Brubaker](https://mbrubake.github.io/), 
and [Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)

York University  &nbsp;&nbsp; 

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

### Abstract
*<p align="justify">
In this paper, we present HistoGAN, a color histogram-based method for controlling GAN-generated images' colors. We focus on color histograms as they provide an intuitive way to describe image color while remaining decoupled from domain-specific semantics. Specifically, we introduce an effective modification of the recent StyleGAN architecture to control the colors of GAN-generated images specified by a target color histogram feature. We then describe how to expand HistoGAN to recolor real images. For image recoloring, we jointly train an encoder network along with HistoGAN. The recoloring model, ReHistoGAN, is an unsupervised approach trained to encourage the network to keep the original image's content while changing the colors based on the given target histogram. We show that this histogram-based approach offers a better way to control GAN-generated and real images' colors while producing more compelling results compared to existing alternative strategies.</p>*

<p align="center">
  <img width = 95% src="https://user-images.githubusercontent.com/37669469/100063841-c500b380-2dff-11eb-8c4a-15fc57bb9caf.gif">
 </p>


### Code
We provide a [Colab notebook example](https://colab.research.google.com/drive/1dAF1_oAQ1c8OMLqlYA5V878pmpcnQ6_9?usp=sharing) code to compute our histogram loss. 
The rest of our code will be available soon!


### Landscape Dataset
Our collected set of 4K landscape images is available [here](https://ln2.sync.com/dl/1891becc0/uhsxtprq-33wfwmyq-dhhqeb3s-mtstuqw7).
<p align="center">
  <img width = 80% src="https://user-images.githubusercontent.com/37669469/100063922-dba70a80-2dff-11eb-9b2d-288f76122e27.jpg">
</p>



