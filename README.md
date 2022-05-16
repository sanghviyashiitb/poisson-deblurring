# Photon-Limited Deblurring Dataset
Real world dataset for evaluation of deblurring algorithms (both non-blind and blind) in the presence of photon shot noise. 

<img src="docs/imaging_setup.png" width=400/> <img src="docs/imaging_setup.jpg" width=250/>

### Contains 
<ul> 
      <li>30 low-light photon shot noise corrputed, blurred images in .RAW format</li>
      <li>Corresponding blur kernel captured using a 30um pinhole</li>
      <li>Ground truth for each image.</li> 
</ul>

### [Download Link](https://1drv.ms/u/s!AjMYTt_aGQ9-hH_myp4irQREzX3K?e=NwARXc)

### Current Benchmarks 
| Method      | PSNR / SSIM |
| -----------  | ----------- |
| **Unrolled-Poisson PnP  [1]** |    **23.48 / 0.566** |
| Deep-Wiener Deconvolution [2]  | 22.85 / 0.561 |
| Deep PnP Image Restoration [3]  | 22.09 / 0.548 |
| PURE-LET [4]  | 20.88 / 0.501 |
| RGDN [5]  | 19.80 / 0.476 |


[1] Sanghvi, Yash, Abhiram Gnanasambandam, and Stanley H. Chan. "Photon Limited Non-Blind Deblurring Using Algorithm Unrolling." arXiv preprint arXiv:2110.15314 2021

[2] J. Dong, S. Roth, and B. Schiele, “Deep Wiener deconvolution: Wiener meets deep learning for image deblurring,” in 34th Conference on Neural Information Processing Systems, Curran Associates, Inc., 2020

[3] K. Zhang, W. Zuo, S. Gu, and L. Zhang, “Learning deep CNN denoiser prior for image restoration,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3929–3938, 2017.

[4] J. Li, F. Luisier, and T. Blu, “Pure-let image deconvolution,” IEEE Transactions on Image Processing, vol. 27, no. 1, pp. 92–105, 2017.

[5] D. Gong, Z. Zhang, Q. Shi, A. van den Hengel, C. Shen, and Y. Zhang, “Learning deep gradient descent optimization for image deconvolution,”
IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5468–5482, 2020

Feel free to ask your questions/share your feedback at sanghviyash95@gmail.com
