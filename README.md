# Animal classification and detection-ViT
IFT 3710/6759 H23 - Projets (avancés) en apprentissage automatique

This project proposes a set of general ideas and data sets all around the topic of animal classification and detection from photos. Images of animals are interesting because they constitute a valuable test bed and benchmark for computer vision models and, at the same time, improving the performance of such models on the task of animal classification and detection has the potential of facilitating the work of life scientists.

Using data:
Caltech Camera Traps (CCT): images taken by motion- or heat-triggered cameras used by biologists to monitor animal populations and behaviour. 

We aim to take advantage of this unique opportunity to learn about and experiment with Vision Transformers. We hope to achieve good classification results on this dataset and better understand how to efficiently approach this type of dataset and classification problem through Vision Transformers and other analogous algorithms.

More specifically, our current research question (subject to change) is : 
Since Vision Transformers were designed for very large datasets, are they still able to achieve good Classification results on a smaller dataset?

To answer this question, we will train ViT and three other models (CNN, Alexnet, Resnet), and allow each model a maximum training time budget of 6 hours. The VIT results will then be compared to those of the three other models (ideas to enforce the 6 hours budget might be subsampling the dataset for example). 
This goal is in line with our limited Computing resources and will allow us to test ViT in a different context than usual (smaller dataset and training time). We will then aim to answer related sub-questions such as : 
On what metrics did ViT do better than the other architectures and vice-versa?
What is shared between the images that were Classified accurately and what is common between those that were not?
