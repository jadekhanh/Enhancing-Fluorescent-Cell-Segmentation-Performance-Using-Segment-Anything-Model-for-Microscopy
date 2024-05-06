# Enhancing Fluorescent Cell Segmentation Performance Using Segment Anything Model for Microscopy

# Collaborators
* Phuong Khanh Tran (Master of Science in Electrical and Computer Engineering, Boston University College of Engineering)
* Dr. Lei Tian - Advisor (Assistant Professor in Electrical and Computer Engineering, Boston University College of Engineering)

# Description
In 2023, Meta AI introduced the Segment Anything Model (SAM), a groundbreaking AI model capable of seamlessly segmenting any object within an image. SAM is a promptable segmentation solution with exceptional zero-shot generalization, eliminating the need for additional training [1]. Building upon SAM's innovation, researchers have introduced an extended version, Segment Anything for Microscopy (micro_sam), designed specifically for microscopy data [2]. This project is centered on the micro_sam framework, with a focus on enhancing fluorescent cell segmentation. Given SAM's architecture, comprising an image encoder, prompt encoder, and mask decoder, the primary objective is to fine-tune these components strategically using micro-sam's fine-tuning technique. This refinement aims to elevate micro_sam's capabilities and significantly improve its performance in the targeted domain of cell segmentation. This advancement holds promise for more accurate and efficient analysis of microscopy images.

# Tasks
The project aims to replicate the fine-tuning studies outlined in the micro_sam paper, with a focus on two primary objectives: firstly, enhancing the performance of SAM and micro_sam models on a fluorescent imaging dataset through sequential component fine-tuning, and secondly, conducting a comparative analysis of the fine-tuned models with their original versions.

# Dataset
For this project, DeepCell's  DynamicNuclearNet dataset is utilized (https://datasets.deepcell.org/), consisting of two subsets: one for tracking and another for segmentation. Given the focus on fluorescent nuclear cell segmentation, the segmentation subset is exclusively utilized. The segmentation subset comprises over 7000 images and more than 700,000 unique annotations, divided into training, validation, and testing sets. The training set consists of 4950 images with 684,037 annotations. The validation set comprises 1417 images with 152,782 annotations, while the testing set encompasses 717 images with 76,300 annotations. All images are standardized to dimensions of 512 x 512 pixels.

# References
[1] A. Kirillov et al., “Segment Anything.” arXiv, Apr. 05, 2023. doi: 10.48550/arXiv.2304.02643.  
[2] A. Archit et al., “Segment Anything for Microscopy.” bioRxiv, p. 2023.08.21.554208, Aug. 22, 2023. doi: 10.1101/2023.08.21.554208.
