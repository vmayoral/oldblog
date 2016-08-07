---
layout:     post
title:      Deep Convolutional Q-Learning
date:       2016-08-07 
summary:    Convolutional Neurral What's reinforcement lerarning? how's reinfocement learning relevant for robotics? how do we apply reinforcement learning in real problems?.
categories: robots, ai, deep learning, rl, reinforcement learning
mathjax: true
---

- [Understanding Convolutional Neural Networks](#convnets)
  - [Motivation](#motivation)
  - [Convolutional Neural Network Architectures](#architectures)
- [Connecting Convolutional Neural Networks](#convnets)  
- [Deep Convolutional Reinforcement Learning, an example](#example)
  - [Code explained](#code)
- [Resources](#resources)  


<div id='convnets'/>
### Understanding Convolutional Neural Networks

Convolutional Neural Networks (ConvNets or CNNs), similar to ordinary Neural Networks, are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function however ConvNet architectures **make the explicit assumption that the inputs are images**, which *allows us to encode certain properties into the architecture*. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

<p style="border: 2px solid #000000; padding: 10px; background-color: #E5E5E5; color: black; font-weight: light;">
It is worth noting that the only difference between Fully Connected layers and Convolutional layers is that the neurons in the convolutional layer are connected only to a local region in the input, and that many of the neurons in a convolutional volume share parameters. However, the neurons in both layers still compute dot products, so their functional form is identical. Therefore, it turns out that **it’s possible to convert between Fully Connected and Convolutional layers**.
</p>


<div id='motivation'/>
#### Motivation

As described by [Andrej Karpathy](https://github.com/karpathy) in *[CS231n Convolutional Neural Networks for Visual Recognition
](http://cs231n.github.io/convolutional-networks/)*:
*Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32*32*3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectible size, e.g. 200x200x3, would lead to neurons that have 200*200*3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.*

A great theoretical explanation of Convolutional Neural Networks is available [here](http://cs231n.github.io/convolutional-networks).

<div id='achitectures'/>
#### Convolutional Neural Network Architectures

The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

where the `*` indicates repetition, and the `POOL?` indicates an optional pooling layer. Moreover, `N >= 0` (and usually `N <= 3`), `M >= 0`, `K >= 0` (and usually `K < 3`). For example, here are some common ConvNet architectures you may see that follow this pattern:

- `INPUT -> FC`, implements a linear classifier. Here `N = M = K = 0`.
- `INPUT -> CONV -> RELU -> FC`
- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`. Here we see that there is a single CONV layer between every POOL layer.
- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

*Prefer a stack of small filter CONV to one large receptive field CONV layer*. Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have \\(C\\) channels, then it can be seen that the single 7x7 CONV layer would contain \\(C \times (7 \times 7 \times C) = 49 C^2\\) parameters, while the three 3x3 CONV layers would only contain \\(3 \times (C \times (3 \times 3 \times C)) = 27 C^2\\) parameters. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.

**Recent departures.** It should be noted that the conventional paradigm of a linear list of layers has recently been challanged, in Google's Inception architectures and also in current (state of the art) Residual Networks from Microsoft Research Asia. Both of these (see details below in case studies section) feature more intricate and different connectivity structures.

<div id='example'/>
### Deep Convolutional Reinforcement Learning, an example
<div id='code'/>
#### Code explained
Let's analyze a 2D fruit fetch example based on [@bitwise-ben](https://github.com/bitwise-ben/Fruit)'s work. Code is available [here](examples/Fruit/qlearning.py):

Dependencies used by the Deep Q-learning implementation:

```python
import os
from random import sample as rsample
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt
```

The `GRID_SIZE` determines how big the environment will be (the bigger the environment, the tougther is to train it)
```python
GRID_SIZE = 15
```
The following function defines a Python coroutine that controls the generic Fruit game dynamics
(read about Python coroutines [here](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)). The coroutine is basically instantiated into a variable that receives `.next()` and `.send()` calls. The first one gets the function code to execute until the point where there's a call to `yield`. The `.send()` call includes an action as parameters which allows the function to finish its execution (it actually never finishes since the code is wrapped in an infinite loop, luckily we control its execution through the primitives just described).

```python
def episode():
    """ 
    Coroutine function for an episode.     
        Action has to be explicitly sent (via "send") to this co-routine.
    """
    x, y, x_basket = (
        np.random.randint(0, GRID_SIZE),        # X of fruit
        0,                                      # Y of dot
        np.random.randint(1, GRID_SIZE - 1))    # X of basket
        
    while True:
        # Reset grid
        X = np.zeros((GRID_SIZE, GRID_SIZE))  
        # Draw the fruit in the screen
        X[y, x] = 1.
        # Draw the basket
        bar = range(x_basket - 1, x_basket + 2)
        X[-1, bar] = 1.
        
        # End of game is known when fruit is at penultimate line of grid.
        # End represents either the reward (a win or a loss)
        end = int(y >= GRID_SIZE - 2)
        if end and x not in bar:
            end *= -1
            
        action = yield X[np.newaxis], end    
        if end:
            break

        x_basket = min(max(x_basket + action, 1), GRID_SIZE - 2)
        y += 1
```

Experience replay gets implemented in the coroutine below. Within this code, one should notice that the code blocks at `yield` expecting a `.send()` call that includes a `experience=(S, action, reward, S_prime)` tuple where:

- `S`: current state
- `action`: action to take
- `reward`: reward obtained after taking `action`
- `S_prime`: next state after taking `action`

```python
def experience_replay(batch_size):
    """
    Coroutine function for implementing experience replay.    
        Provides a new experience by calling "send", which in turn yields 
        a random batch of previous replay experiences.
    """
    memory = []
    while True:
        # experience is a tuple containing (S, action, reward, S_prime)
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)
```

Similar to what was described above, the images are saved using another coroutine:

```python
def save_img():
    """
    Coroutine to store images in the "images" directory
    """
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame = 0
    while True:
        screen = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.savefig('images/%03i.png' % frame)
        frame += 1
```

The model and hyperparameters are defined as follows:

```python
nb_epochs = 500
batch_size = 128
epsilon = .8
gamma = .8

# Recipe of deep reinforcement learning model
model = Sequential()
model.add(Convolution2D(16, nb_row=3, nb_col=3, input_shape=(1, GRID_SIZE, GRID_SIZE), activation='relu'))
model.add(Convolution2D(16, nb_row=3, nb_col=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3))
model.compile(RMSprop(), 'MSE')
```

The main loop of the code implementing Deep Q-learning:

```python
exp_replay = experience_replay(batch_size)
exp_replay.next()  # Start experience-replay coroutine

for i in xrange(nb_epochs):
    ep = episode()
    S, reward = ep.next()  # Start coroutine of single entire episode
    loss = 0.
    try:
        while True:
            action = np.random.randint(-1, 2) 
            if np.random.random() > epsilon:
                # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1

            S_prime, reward = ep.send(action)
            experience = (S, action, reward, S_prime)
            S = S_prime
            
            batch = exp_replay.send(experience)
            if batch:
                inputs = []
                targets = []
                for s, a, r, s_prime in batch:
                    # The targets of unchosen actions are the q-values of the model,
                    # so that the corresponding errors are 0. The targets of chosen actions
                    # are either the rewards, in case a terminal state has been reached, 
                    # or future discounted q-values, in case episodes are still running.
                    t = model.predict(s[np.newaxis]).flatten()
                    t[a + 1] = r
                    if not r:
                        t[a + 1] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                    targets.append(t)
                    inputs.append(s)
                
                loss += model.train_on_batch(np.array(inputs), np.array(targets))

    except StopIteration:
        pass
    
    if (i + 1) % 100 == 0:
        print 'Epoch %i, loss: %.6f' % (i + 1, loss)
```

To test the model obtained:

```python
img_saver = save_img()
img_saver.next()

for _ in xrange(10):
    g = episode()
    S, _ = g.next()
    img_saver.send(S)
    try:
        while True:
            act = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
            S, _ = g.send(act)
            img_saver.send(S)

    except StopIteration:
        pass

img_saver.close()

```

<div id='resources'/>
### Resources:
- Toy example of deep reinforcement model playing the game of snake, https://github.com/bitwise-ben/Snake
- Toy example of a deep reinforcement learning model playing a game of catching fruit, https://github.com/bitwise-ben/Fruit
- Keras plays catch, a single file Reinforcement Learning example, Eder Santana, http://edersantana.github.io/articles/keras_rl/
- Keras plays catch - a single file Reinforcement Learning example
Raw, https://gist.github.com/EderSantana/c7222daa328f0e885093
- [Create a GIF from static images](http://askubuntu.com/questions/648244/how-to-create-a-gif-from-the-command-line)
- [Improve Your Python: 'yield' and Generators Explained](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/)
- CS231n Convolutional Neural Networks for Visual Recognition, http://cs231n.github.io/convolutional-networks/

