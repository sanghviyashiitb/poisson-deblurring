# Photon Limited Non-Blind Deblurring Using Algorithm Unrolling
Pytorch code for [Photon Limited Non-Blind Deblurring Using Algorithm Unrolling](https://arxiv.org/abs/2110.15314) - currently under review at TCI

<img src="https://user-images.githubusercontent.com/20774419/177593608-9b5ccba2-ca3d-485a-9542-5f08df8e081a.png" width="800">

Pretrained model here: 
      [OneDrive](https://1drv.ms/u/s!AjMYTt_aGQ9-hH2aIaReD3DG_ITF)
      [Google-Drive](https://drive.google.com/file/d/1n2_RkgZ0z9rhS2r4rZ2lr2AZn_B5_vbZ/view?usp=sharing)



## Instructions
1. Create a local copy of repository using the following commands
      ```console
      foor@bar:~$ git clone https://github.com/sanghviyashiitb/poisson-deblurring.git
      foor@bar:~$ cd poisson-deblurring
      foor@bar:~/poisson-deblurring$ 
      
      ```
3. Download the pretrained model into ```model_zoo``` from the link [here](https://1drv.ms/u/s!AjMYTt_aGQ9-hH2aIaReD3DG_ITF)
4. To test the network using synthetic data, run the file
      ```console
      foo@bar:~/poisson-deblurring$ python3 demo_synthetic.py  
      ```
      
      ### Output:
      
    <img src="results/demo_synthetic.png" alt="demo_synthetic" width="400"/>

4. Download the zip file containing [real dataset](https://1drv.ms/u/s!AjMYTt_aGQ9-hH_myp4irQREzX3K?e=NwARXc) into the main directory and unzip using the following command:
      ```console
      foo@bar:~/poisson-deblurring$ unzip real_data.zip -d data/ 
      ```
      
5. To test the network using real data, run the file 
      ```console
      foo@bar:~/poisson-deblurring$ python3 demo_synthetic.py  --idx=11
      ```
      (Variable ```idx``` represents the file index and can be any integer from [0,29] )
      

      
      ![demo_real](results/demo_real.png)
      
      ##### Output: PSNR: 29.08, SSIM: 0.696

      
      
Feel free to ask your questions/share your feedback at sanghviyash95@gmail.com
