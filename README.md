# On-Demand Learning for Deep Image Restoration

This project can be used to train a neural network in pytorch to perform image restoration tasks. 
Currently, only pixel interpolation is implemented.

The model is a DCGAN with skip connections. 

The training procedure uses on-demand learning to achieve a better generalization over a range of corruption levels. 






## References
- Gao and Grauman, 2017: On-Demand Learning for Deep Image Restoration
- Mao et al., 2016: Image restoration using very deep convolutional encoder-decoder networks with symmetric skip connections
- Radford et al., 2016: Unsupervised representation learning with deep convolutional generative adversarial networks
- Pathak et al., 2016: Context Encoders: Feature Learning by Inpainting


