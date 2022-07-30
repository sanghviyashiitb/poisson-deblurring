<div style='width:100%;'>
<div style='float:left;width:25%;'> 
<a style="text-align:center" href="https://arxiv.org/pdf/2110.15314.pdf">
      <image src="img/arxiv.png" height="60px">
      <h4>Paper</h4>
</a>
</div>
<div style='float:left;width:25%;'> 
<a style="text-align:center" href="https://ieeexplore.ieee.org/document/9746543/">
      <image src="img/pdf.png" height="60px">
      <h4>PDF</h4>
</a>
</div>
<div style='float:left;width:20%;'> 
<a style="text-align:center" href="https://github.com/sanghviyashiitb/poisson-deblurring/">
      <image src="img/github.png" height="60px">  
      <h4>Code</h4>
</a>
</div>
<div style='float:left;width:20%;'> 
<a style="text-align:center" href="https://aaaakshat.github.io/pldd/">
      <image src="img/dataset3.jpg" height="60px">
      <h4>Dataset</h4>
</a>
</div>
</div>            
    
    
<div style='float:left;'>
<br><br>
<h2>Abstract:</h2>
Image deblurring in photon-limited conditions is ubiquitous in a variety of low-light applications such as photography, microscopy and astronomy. However, the presence of the photon shot noise due to the low illumination and/or short exposure makes the deblurring task substantially more challenging than the conventional deblurring problems. In this paper, we present an algorithm unrolling approach for the photon-limited deblurring problem by unrolling a Plug-and-Play algorithm for a fixed number of iterations. By introducing a three-operator splitting formation of the  Plug-and-Play framework, we obtain a series of differentiable steps which allows the fixed iteration unrolled network to be trained end-to-end. The proposed algorithm demonstrates significantly better image recovery compared to existing state-of-the-art deblurring approaches. We also present a new photon-limited deblurring dataset for evaluating the performance of algorithms. 
<br>
<img src="https://user-images.githubusercontent.com/20774419/177592703-52f38ad4-1750-4157-841d-b8610173576e.png"  class="center" width="800">
<br>
</div>

<div style:'float:left;width:100%;'>
<h2>Unrolled Network Architecture using 3-operator Plug-and-Play</h2>
<br>
<img src="https://user-images.githubusercontent.com/20774419/177593608-9b5ccba2-ca3d-485a-9542-5f08df8e081a.png" width="800">
</div>

Pretrained model here: 
      [OneDrive](https://1drv.ms/u/s!AjMYTt_aGQ9-hH2aIaReD3DG_ITF)
      [Google-Drive](https://drive.google.com/file/d/1n2_RkgZ0z9rhS2r4rZ2lr2AZn_B5_vbZ/view?usp=sharing)

## ICASSP Video Presentation
[![ICASSP](http://img.youtube.com/vi/bJHiUKzjaCI/0.jpg)](http://www.youtube.com/watch?v=bJHiUKzjaCI "Non-Blind Photon-Limited Deblurring")

## Photon-Limited Deblurring Dataset
Real world dataset for evaluation of non-blind deblurring algorithms in the presence of photon shot noise. Contains 30 images at different light levels and blurred by different motion kernels - ground truth kernel captured using a point source.

<img src="docs/imaging_setup.png" width=300/> <img src="docs/imaging_setup.jpg" width=200/>

Feel free to ask your questions/share your feedback at ysanghvi@purdue.edu
