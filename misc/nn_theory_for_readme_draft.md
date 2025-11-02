## Acknowledgements

Once again falling into formalities, though this time more pleasant ones, I would like to share a few acknowledgments that have accompanied me during these weeks of work on the project.

First, even if it is not directly related, I want to mention the book *In-Memory Data Analytics with Apache Arrow*. This project has not used Arrow to the level presented in that book, but I truly enjoyed it, and it inspired me in some way while working on this repository.

I also want to recognize the people who have documented NumPy and PyArrow. They have done an excellent job that makes it easy to resolve any question one might have. Huge thanks!

Finally, I would like to mention another book that I really appreciate for its clarity and breadth: *Neural Networks and Deep Learning* by Aggarwal. A wonderful book, truly. And I guess it's my go-to reference for all this theoretical side of DL, so it had to appear here.

## The Network

At the moment of writing this README, the project has only one neural network model of the Feedforward MultiLayer Perceptron type (or FMLP, as it is designated in the project), with ReLU and softmax activation, although it is possible that we will implement some additional model in the future, and almost surely tweak some other functions. We will theoretically introduce this structure below.

### What Really Is a Feedforward Multi-Layer Perceptron (FMLP)?

A Feedforward Multi-Layer Perceptron is a neural architecture composed of a series of layers of neurons, where data flows in only one direction: from the **input layer**, through one or more **hidden layers**, to the **output layer**. Each neuron in a given layer is connected to every neuron in the next one, forming a dense network of weighted connections. 

Before delving deeper into the structure of the model, let’s first understand the nature of the data it ingests. Before building and, of course, training a neural model, we need two structures: a **feature matrix**, usually denoted as $X$, and a **label vector**, symbolically $y$.

<p align = "center">
	<img src = "misc/python_stack.png" alt = "Python complete stack" width = "60%">
</p>
	
<p align = "center"><em> This really illustrative representation of the data used in a neural model project comes from from the book *Learning Apache Spark with Python*, by Wenqian Feng. Super cool reading! </em></p>

An hour in an introductory linear algebra course is enough to suggest why we use matrices and vectors. On one hand, the **rows of $X$** are defined by the features we wish to use to describe the observations (that is, each position along the rows represents a feature), while the **columns** effectively encode the observations themselves.

Thus, for the element $a_{nm}$ of the matrix $X$, what we have is the value $m$ of observation $n$. If this well-known encoding capability of matrices already made you realize why we need a matrix, you may also understand why we use a **vector** for the labels: each observation, at least in principle, has a single label: it’s a single kind of thing! Therefore, we’re only interested in encoding which observation our classification belongs to, and not several properties of the observation.

Alright, now that we understand what we generally work with when we take part in a neural model project, let’s move on to understanding the components of a neural network like the one used in this project. As we mentioned earlier, every neural model is composed of neurons or nodes, inputs (denoted as $x$), weights (denoted as $W$), biases (denoted as $b$), activation functions, loss functions, and layers (denoted as $L$).

Neurons can be thought of as a kind of **processing unit**. Around them revolves essentially everything that concerns a neural model, and the rest of the components we mentioned earlier can be understood starting from the neuron itself. Let’s take a look.

- **Input ($x$):** the input in a neural model is our connection to the matrix $X$ we saw above.

- **Weights ($W$):** the weights are parameters that define the influence each input has on the neuron. All the connections between neurons from one layer to another are controlled by weights, which can amplify or smooth out the statistical effects of the data associated with each connection. The weights take values between 0 and 1.
