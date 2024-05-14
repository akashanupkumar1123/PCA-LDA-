Linear Discriminant Analysis(LDA) is a supervised learning algorithm used as a classifier and a dimensionality reduction algorithm. We will look at LDA’s theoretical concepts and look at its implementation from scratch using NumPy.



In some cases, the dataset’s non-linearity forbids a linear classifier from coming up with an accurate decision boundary. Therefore, one of the approaches taken is to project the lower-dimensional data into a higher-dimension to find a linear decision boundary. Consider the following example taken from Christopher Olah’s blog.






The other approach is to consider features that add maximum value to the process of modeling and prediction. If any feature is redundant, then it is dropped, and hence the dimensionality reduces. LDA is one such example.

It’s a supervised learning algorithm that finds a new feature space that maximizes the class’s distance. The higher the distance between the classes, the higher the confidence of the algorithm’s prediction.

The purpose for dimensionality reduction is to:

Obtain the most critical features from the dataset.
Visualize the dataset

Have efficient computation with a lesser but essential set of features: Combats the “curse of dimensionality”.

Let’s say we are given a dataset with n-rows and m-columns. Where n represents the number of data-points, and m represents the number of features. m is the data point’s dimensionality.

Assuming the target variable has K output classes, the LDA algorithm reduces the number of features to K-1. Hence, the number of features change from m to K-1.

The aim of LDA is:

Minimize the Inter-Class Variability: Inter-class variability refers to including as many similar points as possible in one class. This ensures less number of misclassifications.

Maximize the Distance Between the Mean of Classes: The classes’ mean is placed as far as possible to ensure high confidence during prediction.





The data-points are projected onto a lower-dimensional hyper-plane, where the above two objectives are met. In the example given above, the number of features required is 2. The scoring metric used to satisfy the goal is called Fischer’s discriminant.

The Fischer score is given as:

Fischer Score f(x) = (difference of means)^2/ (sum of variances).

We’re maximizing the Fischer score, thereby maximizing the distance between means and minimizing the inter-class variability.

Code
Let’s consider the code needed to implement LDA from scratch.

We’ll begin by defining a class LDA with two methods:

__init__: In the __init__ method, we initialize the number of components desired in the final output and an attribute to store the eigenvectors.

transform: We’ll consider Fischer’s score to reduce the dimensions of the input data. The Fischer score is computed using covariance matrices. The formula mentioned above is limited to two dimensions. We’ll be coding a multi-dimensional solution. Therefore, we’ll use the covariance matrices. The matrices scatter_t, scatter_b, and scatter_w are the covariance matrices. scatter_w matrix denotes the intra-class covariance and scatter_b is the inter-class covariance matrix. scatter_t covariance matrix represents a temporary matrix that’s used to compute the scatter_b matrix.

Using the scatter matrices computed above, we can efficiently compute the eigenvectors. The eigenvectors obtained are then sorted in descending order. The first n_components are selected using the slicing operation. If n_components is equal to 2, we plot the two components, considering each vector as one axis.

Finally, we load the iris dataset and perform dimensionality reduction on the input data. Then, we use the plot method to visualize the results.


---The output of the code should look like the image given below. The iris dataset has 3 classes. Observe the 3 classes and their relative positioning in a lower dimension.






