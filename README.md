# cs559-project
Final project for CS-559 with Prof. Xubo Song

## Convolutional Neural Networks

## Backpropagation
### Mathematical background

$\frac{\partial E_n}{\partial w_{ji}} &=

### Programmatic Interface




* Taking inspiration from experience with PyTorch and Keras, I wanted a flexible interface so my network architecture could be easily reshaped at the top level of my program.

```python
    cnn = Sequential(
        Reshape((1, -1, 26)),
        Convolution2d(kernel=(11, 25), in_channels=1, out_channels=32),
        ReLU(),
        Convolution2d(kernel=(11, 13), in_channels=32, out_channels=32),
        ReLU(),
        Flatten(),
        Linear(62400, len(dictionary)),
    )
```

* One point to notice is that the activation functions are distinct modules from the layers they activate.



# Weight initialization
$\lambda$