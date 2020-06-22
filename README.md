# cs559-project
Final project for CS-559 with Prof. Xubo Song

The baseline experiment is in `run_mnist_baseline_softmax.py` and the convolutional experiment is in `run_mnist_cnn.py`.

They don't need any arguments so you should be able to do

```python
pip install -r requirements.txt
python run_mnist_baseline_softmax.py
```

Look for implementation details here:

* `data/*` Some pretty unremarkable things for loading and reshaping data.
* `dnn/activations.py` ReLU and Sigmoid
* `dnn/backprop.py` The recursive function is done in BackpropWrapper.backward(). It's called a BackpropWrapper because this is actually a decorator applied to each module's [numpy] numerical output. 
* `dnn/convolution.py` Conv2D and the (unused) Max Pooling
* `dnn/layers.py` Linear layer and some utility layers/modules
* `dnn/loss.py` Cross Entropy, which has softmax built in
* `dnn/train.py` The high level logic walking through epochs and minibatches
* `tests/test_gradient.py` Tests, mostly around the numerical gradient test

 