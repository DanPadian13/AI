# Supervised Learning in Deep Neural Networks

**Lecturer:** Seán Froudist-Walsh
**Title:** Lecturer in Computational Neuroscience
**Institution:** University of Bristol
**Date:** 12/22/23

---

## How Can We Understand a Neural System?

### The Receptive Field Concept

- A neuron in visual cortex responds to inputs in a particular part of space (its **receptive field**), because its connections can be traced back to precisely that position on the retina of the eye
- It is therefore the **patterns of connectivity** that determine what each brain cell represents

### Key Question

**To understand a neural system we must ask: what is the principle by which the connections are learned?**

**Visual System Hierarchy:**
- V1 → edges and lines
- V2 → shapes
- V4 → objects
- IT → faces

The visual system processes information hierarchically from simple features (edges) to complex representations (faces and objects).

**References:** Summerfield, 2018; Manassi et al., *J. Vision*, 2013

---

## Intended Learning Outcomes

By the end of this video you will be able to:
- Describe **historical and modern approaches to supervised learning**
- Update weights in a neural network using the **backpropagation of errors algorithm**
- Critically assess the **success and failures of deep neural networks** trained with supervised learning as models of human vision
- Describe **two biologically-inspired variants** on the backpropagation algorithm

---

## Learning in a Simple Neural Network: Single-Layer Perceptron

### Historical Foundation

**McCulloch & Pitts (1943)** - *Bulletin of Mathematical Biophysics*
- Proposed the first computational model of a neuron

**Rosenblatt (1958)** - *Proceedings of the IRE*
- Developed the Perceptron algorithm

### Perceptron Architecture

```
Input Layer (I):    I0, I1, I2
                     ↓   ↓   ↓
                   Weights (W)
                     ↓   ↓   ↓
Perceptron Layer:   P1, P2
                     ↓   ↓
Output:            yP1, yP2
```

### Mathematical Formulation

#### Original Formulation
```
y = {1 if Σᵢ Wᵢ→ⱼ xᵢ > -θ
     0 otherwise
```

#### With Bias Term
```
y = {1 if Σᵢ₌₀ⁿ Wᵢ₀→Pⱼ xᵢ + b > 0
     0 otherwise
```

Where:
- `x0 = 1` (bias input)
- `WI0→P1 = bP1` (bias weight for P1)
- `WI0→P2 = bP2` (bias weight for P2)

### Perceptron Learning Rule

**Error Calculation:**
```
δⱼ = tⱼ - yⱼ
```
The error for each unit equals **the target minus the output activity**

**Weight Update Rule:**
```
ΔWᵢ→ⱼ = α δⱼ xᵢ
```

The change in the weights from input unit i to output unit j equals:
- **α** = the learning rate
- **δⱼ** = the error
- **xᵢ** = the activity of the input unit

---

## Supervised Learning

### Core Concept

**Supervised learning** involves an all-knowing teacher that informs the network of the correct response.

**Error Formula:**
```
δⱼ = tⱼ - yⱼ
```

The error for each unit equals the **target** minus the **output activity**.

---

## The Loss Function and Cost Function

### Loss Function
**Definition:** How wrong am I for this training example?

```
l = ½ Σⱼ₌₁ᵐ (tⱼ - yⱼ)²
```

The loss is half the sum of squared errors for all output units.

### Cost Function
**Definition:** How wrong am I on average over all training examples?

```
J = average loss over all training examples
```

**Goal:** We usually want to minimize the average loss over all training examples (the cost function J).

---

## Gradient Descent – Changing Weights to Become Less Wrong

### The Gradient

**Definition:** The gradient tells you how the cost changes with little changes to all of the weights and biases.

```
∇J = [∂J/∂w⁽¹⁾, ∂J/∂b⁽¹⁾, ∂J/∂w⁽²⁾, ∂J/∂b⁽²⁾, ...]ᵀ
```

The magnitude of each element in the gradient `∂J/∂w_old⁽ᴸ⁾` tells you **how sensitive the cost function is to change in each weight and bias**.

### The Gradient Descent Update Rule

```
w_new⁽ᴸ⁾ = w_old⁽ᴸ⁾ - α ∂J/∂w_old⁽ᴸ⁾
```

**With a little change in the weight, how much does the cost go up or down?**

We change the weight in proportion to the learning rate **α** so that we move down (towards lower cost).

This is done for each weight and bias.

### Visualization

The cost function J(w,b) can be visualized as a surface in weight space. Gradient descent finds the path down this surface to the minimum.

**Attribution:** Andrew Ng

### Requirements for Gradient Descent

**Learning by gradient descent is only possible when a small change in the weights leads to a small change in the output value.**

#### Activation Functions

**Not possible:**
- Step function (original perceptron)
- Output changes abruptly between 1 and 0

**Possible:**
- **Sigmoid function:** `y = 1/(1+e⁻ˣ)`
- **ReLU (Rectified Linear Unit)** and similar smooth activation functions

---

## Quiz Question 1 (Blackboard)

**Gradient descent is not possible for the original perceptron model because the __________ is not __________**

**Answer:** activation function / differentiable (or: output / continuous)

---

## Backpropagation – How to Figure Out the Gradient

### Goal

Our goal is to find **how sensitive the loss function is to changes in each of the weights and biases (parameters)**.

**Key Question:** Which changes to these parameters will cause the biggest decrease in the loss?

### Network Structure

```
... → a⁽ᴸ⁻¹⁾ → [w⁽ᴸ⁾, b⁽ᴸ⁾] → z⁽ᴸ⁾ → a⁽ᴸ⁾ → y → J
```

Where:
- **a⁽ᴸ⁾** = activation at layer L
- **w⁽ᴸ⁾** = weights at layer L
- **b⁽ᴸ⁾** = biases at layer L
- **z⁽ᴸ⁾** = weighted sum (pre-activation)
- **f** = activation function (like ReLU or sigmoid)

### Mathematical Relationships

```
a⁽ᴸ⁾ = f(w⁽ᴸ⁾ a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾)
a⁽ᴸ⁾ = f(z⁽ᴸ⁾)
z⁽ᴸ⁾ = w⁽ᴸ⁾ a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
```

### Loss Function

```
Jₙ = (aᴸ - y)²
```

### Goal 1: Change to Loss with Respect to Weight

Using the **Chain Rule**:

```
∂J/∂w⁽ᴸ⁾ = ∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂J/∂a⁽ᴸ⁾
```

---

## Backpropagation – Detailed Example

### Step-by-Step Calculation

#### 1. Loss Function Derivative
```
Jₙ = (aᴸ - y)²

∂Jₙ/∂a⁽ᴸ⁾ = 2(aᴸ - y)
```

#### 2. Activation Function Derivative
```
a⁽ᴸ⁾ = f(z⁽ᴸ⁾)

∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = f'(z⁽ᴸ⁾)
```

#### 3. Weighted Sum Derivative
```
z⁽ᴸ⁾ = w⁽ᴸ⁾ a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾

∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾
```

### Complete Gradient Calculation

```
∂Jₙ/∂w⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾ · f'(z⁽ᴸ⁾) · 2(aᴸ - y)
```

**Key Insight:** How much a change to the weight influences the loss depends on **the activity of the previous layer**.

### Gradient Vector

```
∇J = [∂J/∂w⁽¹⁾, ∂J/∂b⁽¹⁾, ..., ∂J/∂w⁽ᴸ⁾, ∂J/∂b⁽ᴸ⁾]ᵀ
```

Computed using:
```
∂J/∂w⁽ᴸ⁾ = 1/n Σₖ₌₀ⁿ⁻¹ ∂Jₖ/∂w⁽ᴸ⁾
```

(Average over all training examples)

### Backpropagation to Earlier Layers

```
∂Jₙ/∂a⁽ᴸ⁻¹⁾ = ∂Jₙ/∂a⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂z⁽ᴸ⁾/∂a⁽ᴸ⁻¹⁾

∂Jₙ/∂w⁽ᴸ⁻¹⁾ = ∂Jₙ/∂a⁽ᴸ⁻¹⁾ · ∂a⁽ᴸ⁻¹⁾/∂z⁽ᴸ⁻¹⁾ · ∂z⁽ᴸ⁻¹⁾/∂w⁽ᴸ⁻¹⁾
```

This process continues backwards through all layers.

---

## Quiz Question 2 (Blackboard)

**The objective of backpropagation is to find the ________ that minimize the error between the predicted and target outputs. This is done using the ____ rule.**

**Answer:** weights (and biases) / chain

---

## Deep Neural Networks and Brain Data

### Key Finding

**Deep neural networks trained with backpropagation that perform better on object recognition tasks also better predict cortical spiking data.**

### Evidence

The graph shows a strong positive correlation between:
- **X-axis:** Categorization performance (balanced accuracy)
- **Y-axis:** Fit to cortical neural activity (NSE)

**Models shown:**
- V1-like
- V2-like
- Gabor
- Pixels
- HMAX
- PLOS09
- Category-level

**Best performers:**
- Deep neural networks (top hidden layer)
- Models with more layers and better object recognition

**Poor performers:**
- Simple features (pixels, Gabor filters)
- Shallow models

**Reference:** Yamins & DiCarlo, *Nature Neuroscience*, 2016

---

## Criticism: Backpropagation is Not Biologically Realistic

### Three Main Problems

#### 1. Weight Symmetry Problem
**The weights going forwards equal the weights going back**

In standard backprop:
```
∂Jₙ/∂a⁽ᴸ⁻¹⁾ = w⁽ᴸ⁾ f'(z⁽ᴸ⁾) 2(aᴸ - y)
```

The same weights used for forward propagation are used (transposed) for backward error propagation. This is biologically implausible - there's no mechanism for the brain to ensure forward and backward connections have the same weights.

#### 2. Non-local Learning Problem
**The weight update depends on information from distant neurons**

```
∂Jₙ/∂w⁽ᴸ⁻¹⁾ = ∂z⁽ᴸ⁻¹⁾/∂w⁽ᴸ⁻¹⁾ · ∂a⁽ᴸ⁻¹⁾/∂z⁽ᴸ⁻¹⁾ · ∂Jₙ/∂a⁽ᴸ⁻¹⁾
```

Weight updates require information from later layers, which may be anatomically distant.

#### 3. Separate Phases Problem
**The network acts (forward-propagates activity) and learns (back-propagates errors) in two separate phases**

**Acting Phase:**
- Forward propagation of activity through the network
- Generate predictions/outputs

**Learning Phase:**
- Backward propagation of error signals
- Weight updates

This temporal separation is not clearly observed in biological neural networks, which appear to process and learn continuously.

---

## Biologically-Inspired Variants of Backpropagation

### Solution 1: Feedback Alignment

**Key Idea:** Random, fixed feedback weights can work almost as well as symmetric weights.

#### Standard Backpropagation
```
Layer i → Layer j → Layer k
    W           Wᵀ (transpose)
  (forward)   (backward)
```

Requires `W = Wᵀ` (weight symmetry)

#### Feedback Alignment
```
Layer i → Layer j → Layer k
    W           B (random, fixed)
  (forward)   (backward)
```

Uses random feedback matrix `B` instead of `Wᵀ`

#### Performance Comparison

Graph shows Error (NSE) vs Number of examples:
- **Shallow:** ~10⁻⁵
- **Reinforcement:** ~10⁻⁵
- **Backprop:** ~10⁻¹⁰ (best)
- **Feedback alignment:** ~10⁻¹⁰ (nearly as good as backprop!)

**Key Result:** Feedback alignment achieves nearly the same performance as standard backpropagation, solving the weight symmetry problem.

**Reference:** Lillicrap et al., *Nature Communications*, 2016

---

### Solution 2: Dendritic Error Model

**Key Contributors:** Rui Ponte Costa, Maija Filipovica, Ellen Boven, Joe Pemberton, Dabal Pedamonti, Will Greedy, Kevin Nejad & others

#### Neuron Structure Used

**Pyramidal cell anatomy:**
- **Apical dendrites:** Receive feedback/top-down input
- **Soma:** Cell body for activity propagation
- **Basal dendrites:** Receive feedforward/bottom-up input

#### The Model

```
x₁ → W₁ → x₂ → δ₂ → W₂ → x₃
              ↑
              W₂ (feedback)
              ↑
              t₂ (error signal from dendrites)
```

#### Weight Update Rule

```
dW₁/dt = α(g(x₂) - g(δ₂))x₁
```

Where:
- **α** = learning rate
- **g(x₂)** = firing rate at cell body (soma)
- **g(δ₂)** = voltage at dendrite
- **x₁** = firing rate input from the lower area

**Key Innovation:** The weights are updated according to:
1. Learning rate
2. Firing rate input from the lower area
3. **The difference in voltage between the dendrite and cell body**

#### How It Works

- When this network converges to equilibrium, the neurons encode their corresponding **error terms in their dendrites**
- A **single neuron** is used simultaneously for:
  - **Activity propagation** (at the cell body/soma)
  - **Error encoding** (at dendrites)
  - **Error propagation** to the cell body
- **No need for separate phases** - acting and learning happen simultaneously

**References:**
- Sacramento et al., *NeurIPS*, 2018
- Whittington & Bogacz, *TiCS*, 2019

---

## Quiz Question 3 (Blackboard)

**The dendritic error model provides a solution to the problem of non-local learning in backpropagation because the plasticity rule depends on three types of activity in the same _____.**

**Answer:** neuron (or: cell)

---

## Deep Neural Networks Do Not Solve Image Recognition Tasks the Way Humans Do

### Mixed Evidence

#### Supporting Evidence

- **DNNs do the best job in predicting brain signals** in response to images taken from various brain datasets
- Strong correlation between DNN performance and neural predictivity

#### Contradictory Evidence

- However, these behavioral and brain datasets **do not test hypotheses regarding what features are contributing to good predictions**
- DNNs make very different mistakes than humans

### Example: Texture vs Shape Bias

**Geirhos et al., ICLR, 2019**

Three test images:
1. **(a) Texture image:** Cat texture on elephant shape
   - Humans: 16.36% "Cat", 13.04% "Elephant"
   - Model: ~0% "Cat", ~100% "Elephant"

2. **(b) Cue-conflict image:** Elephant texture on cat shape
   - Humans: 17.3% "Cat", 53.7% "Elephant"
   - Model: Different pattern than humans

3. **(c) Texture-Shape cue conflict:** Elephant texture pattern
   - Humans: 30.4% "Cat", 33.53% "Elephant"
   - Model shows different bias

**Finding:** DNNs rely more heavily on **texture** while humans rely more on **shape**.

### Baker et al., 2018

Famous "camel" silhouette illusion: Humans can easily recognize camels from silhouettes, but DNNs struggle with such shape-based recognition.

### Critical Assessment

> **"Deep Neural Networks account for almost no results from psychological research."**
>
> — Bowers et al., *Behavioral and Brain Sciences*, 2022

---

## Recap

### Key Concepts

1. **Supervised learning** – learning from feedback about what exactly the response should have been, from a "teacher"

2. **Single layer perceptron networks** were the first learning neural networks

3. **The gradient** is how sensitive the cost is to changes to individual weights and biases (the direction and rate of fastest increase of the cost function)

4. **Gradient descent** – a method to find the local minimum of the cost function
   - Only works if the activation function can be differentiated
   - Doesn't work for the step function in the first perceptron model

5. **Backpropagation of error algorithm (backprop)** – use the Chain Rule of calculus to calculate the gradient with respect to all the weights and biases in the network, and use this to update the weights

6. **Deep neural networks trained with backprop** that perform better on object recognition tasks also better predict cortical spiking data

7. **Backprop is usually considered not biologically realistic** for several reasons:
   - Weight symmetry problem
   - Non-local learning problem
   - Separate phases problem

8. **Biologically-inspired variants of backprop** have been proposed and are quite successful:
   - Feedback alignment
   - Dendritic error models

9. **Deep Neural Networks do not solve image recognition tasks the way humans do**
   - Different reliance on texture vs shape
   - Different error patterns
   - Limited correspondence with psychological findings

---

## Further Reading (Optional)

### Early Criticism of Backpropagation
- **Crick, Francis.** "The recent excitement about neural networks." *Nature* 337, no. 6203 (1989): 129-132.
  - Nobel prizewinner Francis Crick's critique of the biological realism of backpropagation

### Comparing Deep ConvNets to Brains
- **Yamins, Daniel LK, and James J. DiCarlo.** "Using goal-driven deep learning models to understand sensory cortex." *Nature neuroscience* 19, no. 3 (2016): 356-365.

### Feedback Alignment
- **Lillicrap, Timothy P., Daniel Cownden, Douglas B. Tweed, and Colin J. Akerman.** "Random synaptic feedback weights support error backpropagation for deep learning." *Nature communications* 7, no. 1 (2016): 13276.

### Dendritic Error Model
- **Sacramento, João, Rui Ponte Costa, Yoshua Bengio, and Walter Senn.** "Dendritic cortical microcircuits approximate the backpropagation algorithm." *Advances in neural information processing systems* 31 (2018).

### Problems with Neural Network Models of Human Vision
- **Bowers, Jeffrey S., Gaurav Malhotra, Marin Dujmović, Milton Llera Montero, Christian Tsvetkov, Valerio Biscione, Guillermo Puebla et al.** "Deep problems with neural network models of human vision." *Behavioral and Brain Sciences* (2022): 1-74.
