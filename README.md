# MD-DAN

##  Introduction

Speeding up Magnetic Resonance Imaging (MRI) is an inevitable task in capturing multi-contrast MR images for medical diagnosis. In MRI, some sequences, e.g., in T2 weighted imaging, require long
scanning time, while T1 weighted images are captured by short-time sequences. To accelerate MRI, in this paper, we propose a model-driven
deep attention network, dubbed as MD-DAN, to reconstruct highly undersampled long-time sequence MR image with the guidance of a certain
short-time sequence MR image. MD-DAN is a novel deep architecture
inspired by the iterative algorithm optimizing a novel MRI reconstruction model regularized by cross-contrast prior using a guided contrast image. The network is designed to automatically learn cross-contrast prior
by learning corresponding proximal operator. The backbone network to
model the proximal operator is designed as a dual-path convolutional
network with channel and spatial attention modules. Experimental results on a brain MRI dataset substantiate the superiority of our method
with significantly improved accuracy. For example, MD-DAN achieves
PSNR up to 35.04 dB at the ultra-fast 1/32 sampling rate.

##  Model

![image](https://user-images.githubusercontent.com/50322361/123506696-2b043080-d698-11eb-87db-7eb4d572d669.png)
![image](https://user-images.githubusercontent.com/50322361/123506701-33f50200-d698-11eb-9ca6-2b1e27f8db43.png)


##  Result

![image](https://user-images.githubusercontent.com/50322361/123506716-45d6a500-d698-11eb-8795-c4021bf2c98f.png)
![image](https://user-images.githubusercontent.com/50322361/123506725-4ff8a380-d698-11eb-8d7d-d4a0cbdb3883.png)
![image](https://user-images.githubusercontent.com/50322361/123506732-55ee8480-d698-11eb-934f-da605d10e9d4.png)
