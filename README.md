# **BREAST CANCER PREDICTION FROM MAMMOGRAMS** 

|  Anveshak Rathore  |  Akshay Kolli  |  Prathyusha Reddy Nandikonda |  Lalitha Spoorthy Dantuluri |
|-----------------------------|--------------------------|------------------------------|------------------------------|
|  Miner School of Computer & Information Sciences, Kennedy College of Sciences, University of Massachusetts, Lowell |  Miner School of Computer & Information Sciences, Kennedy College of Sciences, University of Massachusetts, Lowell |  Miner School of Computer & Information Sciences, Kennedy College of Sciences, University of Massachusetts, Lowell Lowell |  Miner School of Computer & Information Sciences, Kennedy College of Sciences, University of Massachusetts, Lowell |
|  anveshak_rathore@student.uml.edu |  akshay_kolli@student.uml.edu |  prathyushareddy_nandikonda@student.uml.edu |  LalithaSpoorthy_Dantuluri@student.uml.edu |


## *Abstract:*
**Early detection of diseases has recently become an important concern in medical research due to rapid growth in the population. Breast cancer is the second most fatal of the cancers that have already been identified. An automatic disease detection system assists healthcare professionals in disease  diagnosis,  provides  reliable,  efficient,  and  rapid action, and reduces the risk of death.** 

**In this study, we are proposing a breast cancer prediction model  using  mammograms  using  a  convolutional autoencoder. The autoencoder is trained on exclusively non- cancer images, such that it perfectly reconstructs a non- cancer image with minimal loss. This way, when the model encounters a cancer positive image, it will give high loss on reconstruction  which  can  be  interpreted  as  a  possible positive case. We aim to create a tool that will help the medical community in the pre-screening phase, to minimize human  error  and  effort  in  recognizing  possible  positive cases.** **We analyze the outcomes of the stated approaches using measures such as Accuracy, F1 score, and Confusion Matrix.** 

## 1. **INTRODUCTION:** 

Breast  cancer  is  the  largest  cause  of  cancer-related deaths among women around the world. 1in 8 women is affected by breast cancer in  their lifetime as per American  Cancer  Society  Survey  Report  in  2013.  Early detection and precise diagnosis are crucial for improving  the  health  of  patients  and  decreasing mortality rates. Mammography is the most widely used imaging  modality  for  screening  breast  cancer. However,  mammography  interpretation  might  be difficult.  

In this project, we propose a breast cancer prediction model based on mammograms that use a convolutional autoencoder. An autoencoder is a neural network design that can be trained to compress input data into a lower- dimensional latent space and then recover the original data from this space. The autoencoder is trained solely on non-cancer images, allowing it to rebuild a non- cancer  image  with  minimum  loss.  When  the  model encounters a cancer positive image, it will return a large loss on reconstruction, indicating a likely positive case. 

Class  imbalance  is  a  major  problem  in  disease prediction in machine learning. It refers to a situation in which the number of examples in one class is much less than the number of instances in another class. Class imbalance is a common issue in the field of disease classification  using  medical  imaging.  Images  from affected patients are significantly harder to come by in the case of rare medical disorders than images from non- affected  patients,  resulting  in  an  undesired  class imbalance.  For beneficial results, the class distribution must be balanced while training a model. This problem serves as the foundation for our approach to using an autoencoder for mammogram classification.  

## 2. **RELATED WORK:** 

1. Aminul  Huq,  Md  Tanzim  Reza,  Shahriar  Hossain, Shakib  Mahmud  Dipto,  AnoMalNet:  Outlier  Detection based  Malaria  Cell  Image  Classification  Method Leveraging Deep Autoencoder. International Journal of Reconfigurable and Embedded Systems (IJRES), Vol. 99, No. 1, Month 2099, pp. 1∼1x. 
2. Kim,  H.E.,  Cosa-Linan,  A.,  Santhanam,  N.  et  al. Transfer  learning  for  medical  image  classification:  a literature review. BMC Med Imaging 22, 69 (2022). 
3. S. Guan and M. Loew, "Breast Cancer Detection Using Transfer  Learning  in  Convolutional  Neural  Networks," 2017  IEEE  Applied  Imagery  Pattern  Recognition Workshop (AIPR), Washington, DC, USA, 2017, pp. 1-8. 
4. Q.  A.  Al-Haija  and  A.  Adebanjo,  "Breast  Cancer Diagnosis in Histopathological Images Using ResNet-50 Convolutional Neural Network," 2020 IEEE International IOT,  Electronics  and  Mechatronics  Conference (IEMTRONICS), Vancouver, BC, Canada, 2020, pp. 1-7. 
5. S.  Kumar,  A.  Narang,  M.  Parihar  and  V.  Sawant, "Breast Tumour Classification using MobileNetV2," 2021 2nd Global Conference for Advancement in Technology (GCAT), Bangalore, India, 2021, pp. 1-6. 

## 3. **METHDOLOGY:** 

### **Dataset Description:**

We are using the RSNA Screening Mammography Breast Cancer  detection  Dataset.  The  Dataset  consists  of  4 images  for  each  person  with  a  whole  size  of  54707 images. In this Dataset we have 1158 of Positive samples and 53549 are of negative samples. 

| **Cancerous images** | **Non-cancerous images** |
| --- | --- |
| ![](/images/cancer1.jpeg) | ![](/images/noncancer1.jpeg) |
| ![](/images/cancer2.jpeg) | ![](/images/noncancer2.jpeg) |
| ![](/images/cancer3.jpeg) | ![](/images/noncancer3.jpeg) |
| ![](/images/cancer4.jpeg) | ![](/images/noncancer4.jpeg) | 

### **Dataset Preprocessing:**

Re-organize: The original data is organized based on the patient. This is reorganized into images with and without cancer.  

Resize: The image resolution is lowered from 5000 \* 5000 to 500 \* 500.  

Reformat: The mammograms in the original dataset are in dicom format. These are converted to PNG format. 

![](/images/comparision1.png)
![](/images/comparision2.png)
![](/images/comparision3.png)
![](/images/comparision4.png)

**Fig 1: Original Vs Reconstructed Images** 

### *Approach:* 

#### Autoencoder

A  custom  encoder  with  the  decoder  will  be  trained using  mammograms  free  of  cancer  images.  The proposed approach is based on the intuition that, during testing, this trained auto-encoder will achieve a loss score for mammograms free of cancer. However, in the case  of  mammograms  with  cancer,  the  model  will output a significantly higher loss value. 

With the help of simple statistics, a cut-off point can be established to label unknown mammograms as the one with cancer or cancer free. An unknown mammogram is  determined  as  an  outlier if the loss value of the unknown image which we get after passing through the model is more than the mean plus three times of the standard deviation of train loss. 

![](/images/AE_train_noncancer.png)

**Fig 2: Auto Encoder- Training with Cancer free Mammograms** 

![](/images/AE_test_noncancer.png)

**Fig 3: Auto-encoder: Testing with Cancer free Mammograms** 

![](/images/AE_test_cancer.png)

**Fig 4: Auto Encoder- Testing with cancer positive Mammogram** 

#### Transfer  Learning

Transfer  learning  is  a  machine learning technique in which a model that has been trained on one task is re-purposed for another related task. The use of a pre-trained convolutional neural network (CNN) as a feature  extractor  is  a  common  approach  for  transfer learning in breast cancer prediction. Another technique is to train a pre-trained CNN to predict breast cancer. Fine- tuning entails retraining the last couple of layers of a pre- trained  convolutional  neural  network  (CNN)  using  a smaller  dataset  of  mammograms.  This  enables  the network  to  customize  features  acquired  from  a  bigger dataset to the specific task of breast cancer prediction.

Fine-tuning  often  involves  freezing  the  weights  of  the previously  learned  CNN  layers  and  just  training  the weights  of  the  last  few  layers  connected  to  a  new classifier. To avoid overfitting, the learning rate is often set to a lower value than during pre-training. The number of epochs used for fine-tuning is often less than that used for pre-training. 

![](/images/finetuning_transfer_learning.png)

**Fig 5: Fine Tuning Architecture** 

- *MobilenetV2:* MobileNetV2  is  a  convolutional  neural network with fewer operations than other architectures, allowing for faster model training. Because of its small size,  high  computing  speed,  and  competitive performance,  MobileNetV2  is  also  suited  for  mobile devices. 

![](/images/mobilenetv2.jpeg)

**Fig 6: Architecture of MobileNetV2** 

- *ResNet50:* CNNs have at least one Convolution layer, wherein instead of matrix multiplication, a convolution operation is performed on the input matrix in order to learn  distinct  low-level  and  high-level  features  of  the image. Deep CNNs are able to learn more features by increasing the depth of the network. However, increasing the depth of the network results in problems of vanishing gradients and degradation. Residual  neural  networks  (ResNet)  address  these challenges  by  introducing  a  "Residual  block",  which features a "skip connection", that adds the output from the previous layer to the layer ahead as illustrated in Fig. 6. If x and F(x) below do not have the same dimension, x is multiplied  by  a  linear  projection  W  to  equalize  the dimensions  of  the  short-cut  connection  and  the  output layer. 

![](/images/resnet50.jpeg)

**Fig 7: Residual Network building block** 

- *VGG16:* A CNN design known as the VGG16 is made up of a stack of convolutional layers with tiny 3x3 filters, max-pooling layers, and fully linked layers at the very end. There are 13 convolutional layers in all, the first two of which have 64 filters each, and the following 11 of which have 128 filters. The feature maps created by the convolutional layers are down sampled using the max-pooling layers, which keeps all  of  the  crucial  characteristics  while  lowering  their spatial  dimensions.  With  the  features  taken  from  the convolutional layers, the fully connected layers at the end of the VGG16 architecture are used for classification to get a class prediction. A probability distribution over the various classes is output by the softmax layer, which is the network's last layer. 

![](/images/vgg16.png)

**Fig 10: VGG16-CNN Model** 

## 4. **RESULTS:** 

We applied the trained models to the images in the test dataset to assess the outcome, i.e. whether a person has breast  cancer  or  not.  To  determine  how  accurate  the training models are, the outcome prediction was compared to the corresponding target feature in the testing set. To compare  the  models,  we  use  a  few  model  evaluation criteria from the scikit-learn Python module. 

**Autoencoder Accuracy:** 99.218%

**Transfer Learning:**

Train images = 49235 images in 2 classes
Validation images = 5471 images in 2 classes
Input image size = 500x500x3

Validation Accuracies:
- VGG16: 98.92%
- Mobilenet V2: 97.92%
- ResNet50: 97.22%


Final autoencoder MSE loss = 0.0011

## 5.  **REFERENCES:** 

1. Dataset: RSNA Screening Mammography Breast Cancer Detection | Kaggle https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data
2. Aminul Huq, Md Tanzim Reza, Shahriar Hossain, Shakib Mahmud Dipto, AnoMalNet: Outlier Detection based Malaria Cell Image Classification Method Leveraging Deep Autoencoder. International Journal of Reconfigurable and Embedded Systems (IJRES), Vol. 99, No. 1, Month 2099, pp. 1∼1x. https://arxiv.org/pdf/2303.05789.pdf
3. Kim, H.E., Cosa-Linan, A., Santhanam, N. et al. Transfer learning for medical image classification: a literature review. BMC Med Imaging 22, 69 (2022). https://link.springer.com/article/10.1186/s12880-022-00793-7#citeas
4. S. Guan and M. Loew, "Breast Cancer Detection Using Transfer Learning in Convolutional Neural Networks," 2017 IEEE Applied Imagery Pattern Recognition Workshop (AIPR), Washington, DC, USA, 2017, pp. 1-8. https://ieeexplore.ieee.org/abstract/document/8457948
5. Q. A. Al-Haija and A. Adebanjo, "Breast Cancer Diagnosis in Histopathological Images Using ResNet-50 Convolutional Neural Network," 2020 IEEE International IOT, Electronics and Mechatronics Conference (IEMTRONICS), Vancouver, BC, Canada, 2020, pp. 1-7. https://ieeexplore.ieee.org/abstract/document/9216455
6. S. Kumar, A. Narang, M. Parihar and V. Sawant,
"Breast Tumour Classification using MobileNetV2," 2021 2nd Global Conference for Advancement in Technology (GCAT), Bangalore, India, 2021, pp. 1-6. https://ieeexplore.ieee.org/abstract/document/9587501
7. https://www.sciencedirect.com/science/article/abs/pii/B9780128181461000040
