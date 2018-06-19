# AstroGAN
Generating galaxies with controlled morphologies using GANs.

The idea is to take a morphological feature and restrict the generation to that morphology.
Below are some examples from two datasets. For each example, the row on the far left is a
real image from a held out test set, and the remaining columns are the generated images.
Each row corresponds to the same morphological feature.

### GalaxyZoo
![zoo](https://i.imgur.com/5IxzM81.png)

### EFIGI
![efigi](https://i.imgur.com/nEWQDpO.png)


### Latent Space Interpolation
We can explore the latent space by picking two random z vectors and two random attributes,
then interpolating between the two. Then, simply pick another set of random vectors and interpolate
again, and so on. Here, [Gaussian interpolation](https://arxiv.org/abs/1609.04468) was used as
opposed to linear interpolation.

#### With Cropping
![cropinterpolate](https://i.imgur.com/oDWfTXG.mp4)


#### Without Cropping
![interpolate](https://i.imgur.com/oDiRaZc.gifv)

