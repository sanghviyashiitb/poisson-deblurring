# Photon-Limited Deblurring Dataset
Real world dataset for evaluation of deblurring algorithms (both non-blind and blind) in the presence of photon shot noise. 

<img src=docs/imaging_setup.png width=500> <img src=docs/imaging_setup.jpg width=300>


Contains 
<ul> 
      <li>30 low-light photon shot noise corrputed, blurred images in .RAW format</li>
      <li>Corresponding blur kernel captured using a 30um pinhole</li>
      <li>Ground truth for each image.</li> 
</ul>

### [Download Link](https://1drv.ms/u/s!AjMYTt_aGQ9-hH_myp4irQREzX3K?e=NwARXc)

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
      
      ### Output:
      
      ![demo_real](results/demo_real.png)

Feel free to ask your questions/share your feedback at sanghviyash95@gmail.com
