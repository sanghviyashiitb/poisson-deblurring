# Photon Limited Non-Blind Deblurring Using Algorithm Unrolling


## Instructions
1. Download the pretrained model into ```model_zoo``` from the link [here](https://1drv.ms/u/s!AjMYTt_aGQ9-hH2aIaReD3DG_ITF)
2. Download the zip file containing real dataset ([link](https://1drv.ms/u/s!AjMYTt_aGQ9-hH_myp4irQREzX3K?e=NwARXc) into the main directory and unzip using the following command:
      ```console
      foo@bar:~$ unzip real_data.zip -d data/ 
      ```
 3. To test the network using synthetic data, run the file
      ```console
      foo@bar:~$ python3 demo_synthetic.py  
      ```
      
      ### Output:
      
    <img src="results/demo_synthetic.png" alt="demo_synthetic" width="400"/>
      
   3. To test the network using real data, run the file 
      ```console
      foo@bar:~$ python3 demo_synthetic.py  --idx=11
      ```
      (Variable ```idx``` represents the file index and can be any integer from [0,29] )
      
      ### Output:
      
      ![demo_real](results/demo_real.png)
