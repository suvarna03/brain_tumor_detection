*****Observations:*****

1. Total number of classes are 4 (pituitary, meningioma,glioma, no_tumor)

2. The number of total images are varies for each class

   a. pituitary : 1457
   b. meningioma : 1329
   c. glioma : 1147
   d. no_tumor : 1067
3. The dataset is slightly imbalanced. The largest class (pituitary) has 1457 images, while the smallest class (no_tumor) has 1067 images. The imbalance is not severe enough to prevent training, but evaluation metrics beyond accuracy (such as Precision, Recall, and F1-score) should be monitored.
4. The dataset contains a mixture of RGB and Grayscale images. Since CNN models require a consistent number of input channels, all images must be converted into the same channel format during preprocessing.
5. we need to convert image mode before giving to tensorflow.
6. Image format JPEG is same over 4 classes
7. Images have varying spatial resolutions (e.g., 202×202, 512×512, 1280×1280). Deep learning models require fixed-size tensors. therefore, all images must be resized to a common resolution before training.
8. Pixel intensities are stored as 8-bit unsigned integers (uint8) ranging from 0 to 255. These values will be normalized to the range [0, 1] before training to improve optimization stability.