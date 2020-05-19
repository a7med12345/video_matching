## Dependencies:

Code Based on CycleGan: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- pytorch = 1.0.1
- python = 3.7.5
- visdom = 0.1.8.9
- imaug = 0.3.0
- moviepy = 1.0.1

## Datasets :

Download Dataset From: https://drive.google.com/file/d/19x8RnikjTNtMedbdyR077v42sPthOMbA/view?usp=sharing
and Unzip in current directory ./

### Training Data: 

Organized in: 
- ./core_dataset/trainA; For Videos
- ./core_dataset/trainannotA; For Annotations

### Testing Data: 

Similary organized in: 
- ./core_dataset/testA; For Videos
- ./core_dataset/testannotA; For Annotations


## Network Definition

Network Definition is found in file ./models/networks.py


## Training

Make sure to run visodm server in another terminal.

To train the network use the following command:

* python train.py --name siamese --model siamese --gpu_ids 0 --batch_size 1

The results will be saved in folder called ./checkpoints/siamese; after every epoch.

## Testing

Link for Pretrained network: https://drive.google.com/file/d/1VSMo2_LwqP20GN1LijptIO3y0YhjRo_Z/view?usp=sharing
Save at ./checkpoints/siamese

To test the network use the following command:

* python test.py --name siamese --model siamese --gpu_ids 0 --batch_size 1 --epoch #epoch

The results will be saved in folder called ./results/siamese/test_#.

We have the output of groups of 4 video sequences:

* Two Similar: Ak1.png and Ak1.npy (for embedding) && Ak2.png and Ak2.npy (for embedding)
* Two Different: Bk1.png and Bk1.npy (for embedding) && Bk2.png and Bk2.npy (for embedding)

k between 1 and N:
