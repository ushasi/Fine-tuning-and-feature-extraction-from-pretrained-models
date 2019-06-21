# Fine-tuning-and-feature-extraction-from-pretrained-models
In this example, we use the pre-trained ResNet50 model, which is pretrained on the ImageNet dataset. The implementation is in TensorFlow-Keras. 

1. We download the weights from "https://github.com/fchollet/deep-learning-models/releases"  (resnet50_weights_tf_dim_ordering_tf_kernels_notop_updated.h5).
2. We set the paths to the train, validation and test files, with respect to our local system path.
3. Using the pretrained Resnet50 model, we fine-tune the model according to our own dataset, and then extract the features from the entire dataset.
4. We save the features as a .Mat file.

# Requirements-

1. Tensorflow
2. Numpy
3. Scipy.io

Happy coding!!
