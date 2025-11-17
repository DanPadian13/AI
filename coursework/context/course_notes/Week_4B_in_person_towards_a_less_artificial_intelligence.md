# Towards a less 'artificial' intelligence

**Lecturer:** Dabal Pedamonti
**Teaching Associate in Engineering Mathematics and Technology**

---

## Intended Learning Outcomes

By the end of this lecture you will be able to:
- Explain the motivation for building brain-realism into AI models
- Identify which aspects of a model are under our control when training a network to perform a task
- Implement features of brain connectivity in a trained recurrent neural network
- Penalise networks for non-brain-like solutions to performing tasks
- Identify unrealistic features of learning in neural networks and some proposed alternatives

---

## AI Models vs Biologically-Realistic Neural Networks

### AI Models – Designed for Performance

**Characteristics:**
- Often inspired by the brain
- Many biological details are removed or simplified
- Excel at multiple real-world applications
- Often do not make clear testable predictions for how the brain works

### Biologically-"Realistic" Neural Network Models

**Characteristics:**
- Parameters are set based on real neural data
- Vary in level of detail, but more biological details are kept vs AI models
- Designed to make directly testable predictions for experiments
- Incapable of complex functions & real-world applications

**References:** Amit & Brunel, Cerebral Cortex, 1997; Wang, Neuron, 2002; Wong & Wang, J. Neurosci., 2006

---

## Can We Bridge the Gap?

**Goal:** Models capable of:
- Complex perceptual, cognitive and motor functions
- Making directly testable predictions for experiments

---

## What is Under Our Control When Designing ANNs?

### NOT Under Our Control:
- The details of the task
- The exact weights

### Under Our Control:
1. **The architecture**
2. **The cost function**
3. **The learning rule**

---

## 1. The Architecture

### AI Approach:
- Choose an architecture that can best solve the problem

### Neuro-AI Approach:
- Choose an architecture that can best solve the problem with pieces connected in a **brain-like manner**

---

## The Architecture of a Cortical Area

**Key observations from Jiang et al., Science, 2015:**

### Connectivity Features:
1. **Sparsity** - not all connections exist
2. **Cell-types** - distinct excitatory and inhibitory neurons
3. **Structured connectivity** - specific patterns between cell types

---

## Implementing Brain-Inspired Architecture: Sparsity

**Method:** Use a mask with element-wise (Hadamard) product

```
W^rec_sparse = W^rec_dense ⊙ Mask
```

Where:
- W^rec_dense: trainable dense weight matrix
- Mask: binary matrix (0s and 1s) defining allowed connections
- ⊙: element-wise multiplication

**Reference:** Song et al., eLife, 2016

---

## Implementing Brain-Inspired Architecture: Dale's Law

### Recap: What is Dale's Law?

**Dale's law:** A neuron releases the same neurotransmitter(s) at all its outgoing synapses.

This means:
- **Excitatory neurons** (glutamate) → make post-synaptic neurons MORE likely to spike
- **Inhibitory neurons** (GABA) → make post-synaptic neurons LESS likely to spike

### Implementation in RNNs:

**Step 1:** Rectify the weights
- After each learning step, set any weights below 0 to zero
- Acts like a ReLU for the weights

**Step 2:** Matrix multiply by a diagonal matrix D
- D has 1 in columns of all Excitatory units
- D has -1 in columns of all Inhibitory units

```
W^rec = W^rec,+ × D
```

Where W^rec,+ contains only non-negative values after rectification.

---

## Brain-Like Solutions from Structured Connectivity

**Imposing biologically-motivated restrictions on RNN connectivity can lead to brain-like solutions**

Examples:
- Networks trained on decision-making tasks develop similar connectivity patterns to those found in prefrontal cortex
- Excitatory-excitatory recurrent connections within selective populations
- Inhibitory neurons provide broad suppression

**References:**
- Wang, Neuron, 2002
- Yang & Wang, Neuron, 2020
- Kuan et al., Nature, 2024

---

## Case Study: Wisconsin Card Sorting Test

### Task Description:
- Reference card shown, then test cards
- Subject must infer hidden rule (color, shape, etc.)
- Feedback provided after each choice
- Rule switches periodically without warning

### Neurobiological Relevance:
- Correct response depends on the task rule, which must be inferred from feedback
- Performance depends on the prefrontal cortex

**References:**
- Milner, Arch. Neurol. (1963)
- Mansouri et al., J. Neurosci. (2006)
- Liu et al., Neuron (2021)

---

## Implementing Multiple Cell-Types for Higher Cognition

**Liu & Wang, Nature Communications, 2024**

### Architecture Components:

```
W^rec,+ = M^rec × W^rec,plastic,+ + W^rec,fixed,+
```

Where:
- **M^rec**: Mask - specific patterns of possible connections for each cell-type
- **W^rec,plastic,+**: Trainable weights
- **W^rec,fixed,+**: Fixed weights (dendrites to cell-body)

### Key Finding:
**Inhibition to dendrites enables separation of activity for different rules**

Successfully performs complex cognitive task (Wisconsin Card Sorting Task)

**Reference:** Song et al., eLife, 2016

---

## 2. The Cost Function

### AI Approach:
Choose a cost function to minimise the error in the task across the training set

### Neuro-AI Approach:
Choose a cost function to minimise the error in the task across the training set, AND penalise the network for solutions that are not "brain-like"

---

## Cost Function: Sparsity

**L1 Regularisation of Weights**

```
J = J_task + J_L1,weight = J_task + β_L1,weight Σ(j=1 to N) Σ(i=1 to N) |w_i→j|
```

Where:
- **J_task**: Regular task cost (penalise poor performance)
- **β_L1,weight**: Hyperparameter balancing task performance vs sparsity
- **Σ|w_i→j|**: Sum of absolute values of weights

**Effect:** Using the absolute value encourages weights to be sent to zero

**Reference:** Yang et al., Nature Neuroscience, 2019

---

## Cost Function: Low Firing Rates

### Biological Motivation:
Neurons in the brain have relatively low firing rates during working memory tasks (typically 0-50 Hz, with most < 20 Hz)

**L2 Regularisation of Firing Rates**

```
J = J_task + J_L2,rates = J_task + β_L2,rates Σ(t=1 to T) Σ(i=1 to N) a²_i,t
```

Where:
- **a_i,t**: Activity of neuron i at time t
- **β_L2,rates**: Hyperparameter balancing task performance vs low activity
- **Squaring**: Discourages large activities

**Reference:** Goudar et al., Nature Neuroscience, 2023

---

## Reminder: Where are Recurrent Connections in the Brain?

**Key facts:**
- There are many long-distance recurrent loops across brain areas
- In primates, approximately 2/3 of all possible connections between brain areas exist (~97% in mice)
- However, most connections (~80%) are from within the same brain area (in the cortex)
- **The number of connections between two brain areas decreases (exponentially) with distance**

**Distance relationship:**
```
Number of neurons ∝ k × exp[-λd]
where λ⁻¹ = 0.23 mm (length constant)
```

**References:**
- Markov et al., Cerebral Cortex, 2011
- Markov et al., J. Comp. Neurol., 2014
- Stefanics et al., Front. Hum. Neurosci, 2014

---

## Cost Function: Distance

### How to discourage long-distance connections:

**Steps:**
1. Give each unit a location in space
2. Calculate the distance between each pair of units
3. Penalise strong weights between distant units

**Weighted Distance Cost:**

```
J = J_task + J_WD = J_task + β_WD Σ(j=1 to N) Σ(i=1 to N) |w_i→j| × |d_i→j|
```

Where:
- **d_i→j**: Distance between units i and j
- **β_WD**: Hyperparameter balancing task performance vs weighted distance
- Each connection incurs a penalty according to the **product of weight and distance**

**Embedding:** Units can be embedded in Euclidean (D) space

**Reference:** Achterberg et al., Nature Machine Intelligence, 2023

---

## 3. The Learning Rule

### AI Approach:
- Use backpropagation or backpropagation through time (BPTT)
- Highly effective but biologically implausible

### Neuro-AI Approach:
- Develop biologically plausible learning rules that still enable effective learning

---

## Recap: Backpropagation is Not Biologically Realistic

**Three main problems:**

### 1. Weight Transport Problem
The weights going forwards equal the weights going back

```
δJ₀/δa^(L-1) = W^(L) f'(z^(L)) 2(a^(L) - y)
```

The same weight matrix W is used in both forward (acting) and backward (learning) passes.

### 2. Credit Assignment Problem
The weight update depends on information from distant neurons

```
δJ₀/δw^(L-1) = (δJ₀/δa^(L-1)) × (δa^(L-1)/δz^(L-1)) × (δz^(L-1)/δw^(L-1))
```

### 3. Separate Phases
The network acts (forward-propagates activity) and learns (back-propagates errors) in two separate phases

---

## Biologically Plausible Learning: Feedback Alignment

**Lillicrap et al., Nat. Comms., 2016**

### Key Idea:
Replace the transpose of the forward weights with **random feedback weights B**

**Standard Backprop:**
```
W₀ → W → W^T (backward)
```

**Feedback Alignment:**
```
W₀ → W (forward)
     ↑
     B (random, backward)
```

### Surprising Result:
- The forward weights W learn to **align** with the random feedback weights B
- Performance approaches standard backpropagation
- Solves the weight transport problem

### Learning Dynamics:
1. Initially, feedback is random and misaligned
2. Over training, forward weights adjust
3. Eventually, W^T and B become aligned enough for effective learning

---

## Biologically Plausible Learning: Dendritic Error Model

**Sacramento et al., NeurIPS, 2018**
**Whittington & Bogacz, TiCS, 2019**

### Key Biological Feature:
**Pyramidal cells** (most common neurons in brain) have:
- **Distal dendrites**: receive feedback/top-down input
- **Proximal dendrites**: receive feedforward/bottom-up input
- **Soma**: integrates input from all dendrites

### How It Works:

1. Neurons encode error terms in their **dendrites** (δ₂)
2. Cell body activity represents the regular activation (x₂)
3. Weight updates based on **difference between dendrite and soma**:

```
dW₁/dt = α(g(x₂) - g(δ₂))r_x₁
```

Where:
- α: learning rate
- r_x₁: firing rate input from lower area
- g(x₂) - g(δ₂): voltage difference between cell body and dendrite

### Advantages:
- Single neuron used simultaneously for:
  - Activity propagation (at cell body)
  - Error encoding (at dendrites)
  - Error propagation to cell body
- **No need for separate phases** (acting vs learning)

---

## Biological (Im)plausibility of BPTT

**Two main problems for recurrent networks:**

### 1. Weight Transport Problem (same as feedforward)
The weights going forwards equal the weights going back

### 2. Perfect Memory Problem
The same neuron must store and retrieve, with **perfect accuracy**, the values of its activities from all points in the past

This is biologically implausible because:
- Neural activity is noisy
- Synaptic transmission is unreliable
- Neurons don't have unlimited memory capacity

---

## Biologically Plausible Learning for RNNs: RFLO

**Random Feedback Local Online (RFLO)**
**Murray, eLife, 2019**

### Learning Rule:

```
∂W_ab(t) = α [Bδ(t)]_a × p_ab(t)
```

Where:
- **α**: learning rate
- **[Bδ(t)]_a**: error distributed in proportion to random weights
- **p_ab(t)**: moving average of recent co-activations between neurons a and b

### Co-activation Trace:

```
p_ab(t) = (1/τ)φ'(u_a(t))h_b(t-1) + (1 - 1/τ)p_ab(t-1)
```

Where τ controls the timescale of the moving average.

### Strengths:
- **Local**: only depends on information each neuron should have
- **Online**: no need to run activations backwards at end of trial
- **No weight sharing**: forward and backward weights are independent

### Limitation:
- Co-activations are forgotten at a rate determined by the neuronal timescale
- Performance degrades for very long temporal dependencies

### Performance:
- Approaches BPTT performance for moderate timescales (T = 20τ to 160τ)
- More biologically plausible than BPTT

---

## Summary: Three Components Under Our Control

When designing an artificial neural network to perform a task:

### 1. Architecture
**Neuro-AI implementations:**
- **Sparsity**: Use masks or L1 regularisation
- **Dale's law**: Rectify weights + diagonal sign matrix
- **Cell types**: Different connection patterns for E/I neurons

### 2. Cost Function
**Neuro-AI additions to task loss:**
- **Sparsity**: L1 regularisation of weights
- **Low firing rates**: L2 regularisation of activities
- **Distance**: Penalise |weight| × |distance|

### 3. Learning Rule
**Biologically plausible alternatives:**
- **Feedback alignment**: Random backward weights
- **Dendritic error model**: Errors encoded in dendrites
- **RFLO**: Local, online learning with co-activation traces

---

## Recap

### Main Concepts:

1. **In neuro-AI**, we aim to build models capable of performing complex tasks that can also make testable predictions for how the brain may solve the task

2. **When designing ANNs**, three things are under our control:
   - The architecture
   - The cost function
   - The learning rule

   (The exact weights and task details are NOT under our control)

3. **Sparse networks** can be learned by:
   - Using a mask
   - L1 regularisation of weights or activity

4. **Dale's law** (excitatory/inhibitory) can be implemented by:
   - Rectifying weights (set negative to zero)
   - Multiplying by diagonal matrix of 1s and -1s

5. **High firing rates** can be penalised by:
   - L2 regularisation on the activity

6. **Strong long-distance connections** can be discouraged by:
   - Embedding units in spatial positions
   - Calculating distances between pairs
   - Penalising product of weight × distance

7. **Biologically-realistic learning** is ongoing research:
   - Feedback alignment
   - Dendritic error accumulation
   - Random feedback local online (RFLO) learning

---

## Further Reading

### Cell-type connectivity:
- Jiang et al. "Principles of connectivity among morphologically defined cell types in adult neocortex." Science 350, no. 6264 (2015): aac9462.

### Training E-I RNNs:
- Song, Yang, and Wang. "Training excitatory-inhibitory recurrent neural networks for cognitive tasks: a simple and flexible framework." PLoS computational biology 12, no. 2 (2016): e1004792.

### Wisconsin Card Sorting Task:
- Liu and Wang. "Flexible gating between subspaces by a disinhibitory motif: a neural network model of internally guided task switching." bioRxiv (2023): 2023-08

### Penalising long connections:
- Achterberg et al. "Spatially-embedded recurrent neural networks reveal widespread links between structural and functional neuroscience findings." bioRxiv (2022): 2022-11.

### Feedback alignment:
- Lillicrap et al. "Random synaptic feedback weights support error backpropagation for deep learning." Nature communications 7, no. 1 (2016): 13276.

### Dendritic error model:
- Sacramento et al. "Dendritic cortical microcircuits approximate the backpropagation algorithm." Advances in neural information processing systems 31 (2018)

### Biologically plausible BPTT alternative:
- Murray. "Local online learning in recurrent networks with random feedback." Elife 8 (2019): e43299.

### Training NNs for neuroscience:
- Yang and Wang. "Artificial neural networks for neuroscientists: a primer." Neuron 107, no. 6 (2020): 1048-1070.

---

## Quiz Questions

### Question 1:
**What is Dale's law? How can Dale's law be enforced on a recurrent neural network in which each unit uses the ReLU activation function?**

**Answer:**
Dale's law states that a neuron releases the same neurotransmitter(s) at all its outgoing synapses. To enforce this in an RNN:
1. Rectify weights after each learning step (set negative values to zero)
2. Multiply by a diagonal matrix D with +1 for excitatory units and -1 for inhibitory units

### Question 2:
**Which three components are specified by the designer in artificial neural networks? What is not specified by the designer?**

**Answer:**
- **Specified**: Architecture, cost function, learning rule
- **NOT specified**: Exact weights, task details

---

## Key Takeaways

1. **Brain-inspired AI** bridges performance and biological plausibility
2. **Architecture constraints** (sparsity, Dale's law, distance) can lead to brain-like solutions
3. **Cost function modifications** guide networks toward biologically realistic solutions
4. **Biologically plausible learning** remains an active area of research
5. These approaches enable models that both perform well AND make testable neuroscience predictions
