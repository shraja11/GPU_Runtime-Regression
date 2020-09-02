# GPU_Runtime-Regression
This is a project that implements simple-linear regression using a Deep Learning Model of three layers using a 3 layer Neural Network that is based on PyTorch Framework.

The Dataset can be found at - https://www.kaggle.com/rupals/gpu-runtime?select=sgemm_product.csv

Our Goal:
Implement a linear regression model on the dataset to predict the GPU run time. Use the average of four runs as the target variable. Use the sum of squared error normalized by 2*number of samples [J(β0, β1) = (1/2m)[Σ(yᶺ(i) – y(i))2] as your cost and error measures, where m is number of samples.

Dataset :
18 rows have been provided to us. And all of those are features on which the runtime depends upon.

Approach :
3 layers were used. ADAM optimizer was useed to do Gradient Descent with a Learning Rate of 0.01.
