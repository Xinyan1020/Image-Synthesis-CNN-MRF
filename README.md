# Image-Synthesis-CNN-MRF
This is the code and the results of our project for the course Computational Imaging with the subject : Combination of Convolutional Network and Markov Random Field for image synthesis. 
In this project, we replaced the VGG-19 network, which was used to extract the featres, with a pretrained Inception-V3 network. We tested the effect of some parameters on the results : We have changed the content and loss layers, the content loss weight, the style loss weight, the patch size and we have also changed the content image and the style image . The implementation is based on: https://github.com/jonzhaocn/cnnmrf-pytorch.
# Running the program on your own images.
python main.py /path/to/content/image/ /path/to/style/image/
