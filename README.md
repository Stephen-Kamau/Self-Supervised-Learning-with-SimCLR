# Self-Supervised-Learning-with-SimCLR
In this task, a similar approach as described above was used. At first, it was data preparation. Cifar10 was used to train a self supervised model while cifar100 images was used for downstream tasks using a logistic regression method.




## Introduction.
- Self supervised learning is also called unsupervised learning. It normally describes a process through which a machine model is able to learn from data without having expected output. With a given input data, the machine trains and can be able to classify labels into classical groups as a supervised model would do. The model is able to learn information which can distinguish certain images like their descriptive patterns. These unsupervised learning methods are able to learn as much as possible from data alone and can be fine tuned for a specific task.
- One of the unsupervised methods that was proposed is SimCLR. It accepts images without any labels and trains the model quickly to adapt to any imaging task that can be done downstream for specific purposes. During the training iterations, a sample batch of images are sampled and for each image, two of its versions are created by applying some imaging transformation techniques. We then pass the image through a network and then apply the results into a predefined model head which is often an MLP network. The output features are trained and the task of the model is to be able to recognize content of the image that remained unchanged from the changed version through transformation.



## Task Approach
- In this task, a similar approach as described above was used. At first, it was data preparation. Cifar10 was used to train a self supervised model while cifar100 images was used for downstream tasks using a logistic regression method. In preparation of the dataset for training, all images were transformed through cropping and ensuring that their sizes remain 32, altering its colour through distortion and finally applying a gaussian blurring randomly. These transformations returned two views of the same image input. These image views are the one used for training the self supervised model. These transformations were only applied to the self supervised training tasks. Another transformation that was a must for all images was normalising images pixels between 0 and 1 and converting the image into tensor for fast processing during training and evaluation phases.



#### During training, the pipeline worked as follows. 
- Through each Iteration, each of the image views is used i.e x and x`. Both are then encoded into a 1D feature vector where they are used to maximise similarities between them. The encoder network is divided into two parts i.e Base encoder network (backbone) and Projection Head which consist of the added custom layers. The base network used for this task is a CNN model which is preatrained with imagenet weights using RESNET18 model. The head of the resnet18 model was changed and a Sequential layer with RELU (activation layer) and Linear Layers added on to it as Project Head for the self supervised network. For evaluation during the self training process, Cosine similarity was used to evaluate the indifferences between two same images passed through the network. It is evaluated using a temperature value to determine the peaked distribution by allowing to balance the influence of dissimilar images patches against the similar ones. Since the max cosine similarity is 1 and -1 for minimum. Features of two images converge when the value is around zero (total not similar) while 1 will be similar and -1 is similar but in opposite direction. This is the evaluation metric used during training and was considered as the model’s accuracy.
- After Training the network, the top projection head is removed and a new projection head is added that will be used to perform a specific task at hand. In this case, a projection head that was added was a linear layer for downstream tasks. Model Parameters Used include the following.
`
    ● Batch size : 256
    ● Hidden Dimension : 128
    ● Learning Rate: 0.0002
    ● Weight Decay: 0001
    ● Epochs: 25
`

### Downstream Model
- After training a self supervised model, a logistic regression approach was used to replace its projection head in order to create a downstream model for cifar100 classification task. The model assumed that all images are encoded into feature vectors (for this case, feature vectors are trained by first passing the images through the unchanged model network ). The image's features are passed through the self supervised network and its output is then passed through to a linear layer (acts as a projection head) and output probabilities of the expected classes as predictions . For this part, the evaluation metric was accuracy which determines how many labels were accurately predicted as corrected from actual outputs and expected outputs.

- The model trained on class labels from 0 to 9 had an accuracy of about 60% on the test dataset which was slightly lower by 8% from the training accuracy. This model was trained on the following hyperparameters;
`
      ● Batch size = 32,
      ● Number of epochs = 100
      ● Learning Rate = 0.001
      ● Weight Decay = 0.001
      ● Number of classes =10 (Only the selected one)
`

- Training curves for both loss and accuracy were always shifting and did not have a constant movement either increase for accuracy and reduction for loss
