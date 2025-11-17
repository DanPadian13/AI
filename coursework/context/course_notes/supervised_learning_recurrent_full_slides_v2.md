# Supervised Learning in Recurrent Neural Networks

**Lecturer:** Seán Froudist-Walsh
**Position:** Lecturer in Computational Neuroscience
**Date:** 1/13/24

---

## Intended Learning Outcomes

By the end of this video you will be able to:

- Understand backpropagation through time, and its strengths and limitations in terms of performance and biological realism
- Build and write equations for distinct RNN architectures and detail how they deal with common issues to training RNNs over long timescales
- Describe the main ideas behind two biologically-inspired learning rules for RNNs
- Describe the potential benefits of, and best practices for, applying RNNs to understanding the neural mechanisms of cognition

---

## Recurrent Neural Networks (RNNs) in AI

### The Basic ("Vanilla") RNN

**Architecture:**
- Input layer (x)
- Hidden layer (a)
- Output layer (o)

**Equations:**

The hidden layer activity:
```
a(t) = f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁)
```

- Is a nonlinear function (e.g. ReLU, sigmoid)
- Of a weighted sum of the inputs
- Plus a bias
- And a weighted sum of the hidden layer activity from one timestep ago

The output layer activity:
```
o(t) = g(W_{a→o} a(t) + b₂)
```

- Is a nonlinear function
- Is a weighted sum of the hidden layer activity
- Plus a bias

**Reference:** Elman, *Cog. Sci.*, 1990

---

## Unrolled RNN

When unrolled across time, an RNN shows:

- Time 1: x⁽¹⁾ → a⁽¹⁾ → o⁽¹⁾
- Time 2: x⁽²⁾ → a⁽²⁾ → o⁽²⁾
- ...
- Time T: x⁽ᵀ⁾ → a⁽ᵀ⁾ → o⁽ᵀ⁾

With initial condition a⁽⁰⁾ at the start.

---

## Forward Propagation in the RNN

The same weight matrices are shared across all timesteps:
- **W_{x→a}**: Input to hidden weights
- **W_{a→a}**: Recurrent weights
- **b₁**: Hidden layer bias
- **W_{a→o}**: Hidden to output weights
- **b₂**: Output layer bias

---

## Cross Entropy Loss

**Loss function (Cross entropy):**
```
J⁽ᵗ⁾ = -y⁽ᵗ⁾ log(o⁽ᵗ⁾)

J = Σ(t=1 to T) J⁽ᵗ⁾
```

Used for classification tasks (not specific to RNNs)

**Properties:**
- Cross entropy J⁽ᵗ⁾ = -log(o⁽ᵗ⁾)
- Squared error = (1 - o⁽ᵗ⁾)²
- At terrible prediction (0): Loss is high
- At great prediction (1): Loss is low

---

## Loss in the RNN

The total loss is calculated across all timesteps:
```
J = Σ(t=1 to T) J⁽ᵗ⁾
```

Each timestep contributes: J⁽¹⁾, J⁽²⁾, ..., J⁽ᵀ⁾

---

## Backpropagation in the (Vanilla) RNN

Gradients flow backward through:
1. Output to hidden connections
2. Hidden to hidden (recurrent) connections across time
3. Input to hidden connections

The loss at each timestep affects all previous timesteps through backpropagation through time.

---

## Backpropagation Through Time (BPTT)

### Exploding Gradient Problem

For 100 timesteps, if w = 2:
```
2¹⁰⁰ = 1,267,650,600,228,229,401,496,703,205,376
```

### Vanishing Gradient Problem

For 100 timesteps, if w = 0.5:
```
0.5¹⁰⁰ = 0.0000000000000000000000000000008
```

**For a weight matrix W_{a→a}:**
- If the largest eigenvalue > 1: **Exploding gradient problem**
- If the largest eigenvalue < 1: **Vanishing gradient problem**

---

## Fixing Exploding Gradients - Gradient Clipping

**Quote:** "When the traditional gradient descent algorithm proposes to make a very large step, the gradient clipping heuristic intervenes to reduce the step size to be small enough that it is less likely to go outside the region where the gradient indicates the direction of approximately steepest descent"

**References:**
- Pascanu et al., *ICML*, 2013
- Goodfellow et al., *Deep Learning*, 2016

---

## Understanding the Vanishing Gradient Problem

**Key insight:**

The vanishing gradient problem for vanilla RNNs means that the algorithm believes that activity at early timepoints will not affect the loss.

Therefore it will not make big adjustments to the weights based on activity at early timepoints.

**Important:** This isn't because early time steps don't influence the loss, but because the backpropagation-through-time process struggles to recognise and act upon their influence.

---

## Leaky RNNs for Neuro-AI

**Discrete time version (Euler method - written in code):**

```
a(t) = a(t-1) + (Δt/τ)(-a(t-1) + f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁))
```

Let `Δt/τ = β`:

```
a(t) = a(t-1) + β(-a(t-1) + f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁))

a(t) = (1-β)a(t-1) + β(f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁))

new a = (1-β) old a + β(new inputs)
```

**If β = 1, we get back the vanilla RNN**

Can add noise: + noise

---

## Slowly Vanishing Gradients in Leaky RNNs

**Equation:**
```
a(t) = (1-β)a(t-1) + β(f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁))
```

**Gradient calculations:**
```
∂a(t)/∂a(t-1) = (1-β) + β f'(...) W_{a→a}

∂a(t-1)/∂a(t-2) = (1-β) + β f'(...) W_{a→a}
```

So the terms (1-β) and the recurrent weights β f'(...) W_{a→a} are multiplied over and over again as we backpropagate through time.

```
∂J/∂a(t-k) = ∂J/∂a(t) · ∂a(t)/∂a(t-1) · ∂a(t-1)/∂a(t-2) ... ∂a(t-k+1)/∂a(t-k)
```

**If β << 1**, then the magnitude of the gradient is dominated by (1-β).

On each backpropagation step, we multiply by a number close to 1, and the gradient vanishes slowly.

This will allow us to use activity further in the past to influence training.

---

## Sigmoid Activation Function

```
sigmoid f(x) = σ(x) = e^x / (e^x + 1)
```

Range: [0, 1]

---

## Light Gated Recurrent Units (Light GRUs)

**Leaky RNN** - scalar β (just one number controls the timescale of remembering/updating for all units):
```
a(t) = (1-β)a(t-1) + β(f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁))
```

**Light GRU** - vector β (a different number controls the timescale for each unit, and this changes in time):
```
a(t) = (1-β(t)) ∘ a(t-1) + β(t) ∘ f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁)
```

The gate β is learned, and depends on the inputs and recurrent connections:
```
β = σ(W_{x→β} x(t) + W_{a→β} a(t-1) + b_β)
```

**Key features:**
- The gate controls the fraction (sigmoid) of newly calculated activity that will contribute to the next memory state a(t) for each unit
- f = ReLU
- Normalise activations across units in a layer (subtract mean & divide by standard deviation plus a small number)
- Something (superficially) similar happens in the brain

**References:**
- Ravanelli et al., *IEEE*, 2018
- Carandini & Heeger, *Nat. Rev. Neurosci.*, 2012

---

## Sigmoid & Tanh Activation Functions

**Sigmoid:**
```
f(x) = σ(x) = e^x / (e^x + 1)
```
Range: [0, 1]

**Tanh:**
```
f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```
Range: [-1, 1]

---

## Gated Recurrent Units (GRUs)

**Light GRU:**
```
a(t) = (1-β(t)) ∘ a(t-1) + β(t) ∘ f(W_{x→a} x(t) + W_{a→a} a(t-1) + b₁)
```

**GRU** (a separate gate on previous timestep activations affects the update):
```
a(t) = (1-β(t)) ∘ a(t-1) + β(t) ∘ f(W_{x→a} x(t) + W_{a→a}(a(t-1) ∘ γ) + b₁)
```

The gates β and γ are learned, and depend on the inputs and recurrent activity/weights:
```
β = σ(W_{x→β} x(t) + W_{a→β} a(t-1) + b_β)
γ = σ(W_{x→γ} x(t) + W_{a→γ} a(t-1) + b_γ)
```

f = tanh

**Reference:** Cho et al., *arXiv*, 2014

---

## LSTM (Long Short-Term Memory)

LSTMs maintain and update separate **long-term memory** and **short-term memory** variables.

They control how long and short-term memories are updated by use of 3 gates.

**Each gate** controls the fraction (sigmoid) of 'something' contributing to a new memory state:

**Forget gate:**
```
f(t) = σ(W_{h→f} h(t-1) + W_{x→f} x(t) + b_f)
```
Depends on the short-term memory, inputs and a bias.

**Input gate:**
```
i(t) = σ(W_{h→i} h(t-1) + W_{x→i} x(t) + b_i)
```
The fraction you allow in of the candidate new long-term memory.

**Output gate:**
```
o(t) = σ(W_{h→o} h(t-1) + W_{x→o} x(t) + b_o)
```

**Candidate new long-term memory:**
```
ĉ(t) = tanh(z_c(t))
z_c = W_{h→c} h(t-1) + W_{x→c} x(t) + b_c
```
Each candidate new memory uses a tanh activation function.

**New long-term memory:**
```
c(t) = f(t) ∘ c(t-1) + i(t) ∘ ĉ(t)
```
The new long-term memory is the fraction you don't forget of the old long-term memory plus the fraction you allow in of the candidate new long-term memory.

**Candidate new short-term memory:**
```
ĥ(t) = tanh(c(t))
```

**New short-term memory (output):**
```
h(t) = o(t) ∘ ĥ(t)
```
The new short-term memory (output) is the fraction you output from the candidate new short-term memory.

**Note:** 'Long-term memory' here is not equivalent to long-term memory in neuroscience/psychology, which can die off in activity and be recalled years later.

**Reference:** Hochreiter & Schmidhuber, *Neural Computation*, 1997

---

## Solving Vanishing Gradients with LSTMs

**Key equation:**
```
c(t) = f(t) ∘ c(t-1) + i(t) ∘ ĉ(t)
```

**Gradient flow:**
```
∂c(t)/∂c(t-1) = f(t) + stuff
```

To avoid vanishing gradients, we just need at least one of the paths back from the loss to not vanish.

This will ensure a steady flow of the gradient back through time.

```
∂J(t)/∂c(t-k) = ∂J(t)/∂c(t) · ∂c(t)/∂c(t-1) · ∂c(t-1)/∂c(t-2) ... ∂c(t-k+1)/∂c(t-k)
```

The forget gates thus regulate the gradient flow, according to how much a long-term memory state c(t-1) contributed to the next state c(t).

**Credit:** Mitesh Khapra

---

## Realism in RNN Architectures

**Key observations:**

- Leaky RNNs are close to traditional (untrained) recurrent neural network models in computational neuroscience
- Leaky RNNs are often observed to have brain-like dynamics
- This is not always true for other RNN architectures (vanilla RNN, light GRU, GRU, LSTM)
- However other models (light GRU, GRU, LSTM) are more effective at learning long-term dependencies
- Efforts have been made to develop more biologically-plausible versions of these more powerful architectures

**Biologically-inspired variants:**
- **Biologically-inspired variant of LSTM** (Costa et al., *NeurIPS*, 2017)
- **Biologically-inspired variant of light GRU** (Berg et al., *bioRxiv*, 2023)

---

## Quiz - Blackboard

**Question:** Both GRUs and LSTMs incorporate _________ mechanisms that allow them to selectively remember or forget information, which is especially crucial for sequences where important information might be spaced apart. In contrast, leaky RNNs introduce a _____ in the recurrent loop to allow the older states to decay over time.

**Answer:** gating; leak

---

## Biological (Im)plausibility of BPTT

Two main issues:

1. The weights going forwards equal the weights going back (like in backpropagation for feedforward networks)

2. The same neuron must store and retrieve, with perfect accuracy, the values of its activities from all points in past

---

## Biologically Plausible Learning in RNNs?

**Murray's approach (eLife, 2019):**

```
∂W_ab(t) = α [δB(t)]_a p_ab(t)
```

The change in weight between neurons a and b at time t depends on:
- The learning rate
- The error distributed in proportion to random weights
- And a moving average of recent co-activations between neuron a and b

**Moving average equation:**
```
p_ab(t) = (1/τ)φ'(u_a(t))h_b(t-1) + (1 - 1/τ)p_ab(t-1)
```

**Strengths:**
- Only depends on information that each neuron should have (i.e. local)
- No need to run the activations backwards at the end of the trial to learn
- No need re-use weights in the forward and backward directions

**Limitation:**
- Co-activations are forgotten at a rate determined by the neuronal timescale
- But backwards replay does happen in the brain! (e.g., Ólafsdóttir et al., *Current Biology*, 2018)

**Alternative approach (Cheng and Brown, bioRxiv, 2023):**
```
ΔW ∝ r · Σ(t=1 to T) p^T · (l^T + l^(t+1)T)
```

Hook up an RNN to a variant of a Hopfield network, where memories are linked to their neighbours in time.

---

## Perhaps the Learning Rule Isn't Plausible, But the Learned Neural Network Is?

This leads us to using RNNs as models of cognition...

---

## The Brain Uses Context to Guide Behaviour

**Study:** Mante*, Sussillo* et al., *Nature*, 2013

**Key finding:** Identical sensory stimuli can lead to very different behavioral responses depending on context.

**Experimental design:**
- Monkeys performed a task requiring attention to either motion or color
- The context (which feature to attend to) changed the behavioral response
- Even with identical sensory inputs

---

## How the Brain Uses Context to Guide Behaviour

**Key finding:** Color information reaches prefrontal cortex but does not affect choice.

**Neurons show:**
- Neuron 1: Responds to motion
- Neuron 2: Responds to color
- Neuron 3: Shows choice-related activity

**Rules out dominant "early selection" hypothesis:** Sensory information is NOT 'gated out' before reaching prefrontal cortex.

**Reference:** Mante*, Sussillo* et al., *Nature*, 2013

---

## How a Leaky RNN Uses Context to Guide Behaviour

**Method for using RNNs to understand neural mechanisms of cognition:**

1. Train RNN on same task as individual (e.g. human, monkey, mouse) performing task
2. Analyse RNN using the same methods as used to analyse real neural activity
3. If the RNN is a good match to the neural data, then perform additional 'experiments' and analyses on the RNN to enable deeper insights into neural mechanisms

**This paper (Mante*, Sussillo* et al., Nature, 2013):**

1. Leaky RNN trained with BPTT
2. The RNN activity resembled real neural population activity seen in prefrontal cortex
3. Dynamical systems analysis showed that the context modifies the 'dynamical landscape', affecting how the neural responses to motion and color guide activity towards making a choice

**Conclusion:** RNNs, when applied and analysed carefully, can generate new hypotheses about how the brain performs its functions.

---

## Moving Beyond Traditional Computational Neuroscience Models

**Study:** Yang et al., *Nature Neuroscience*, 2019

**Achievement:** A single network performing 20 cognitive tasks

**Key insight:** Training neural networks can teach us how to build computational models for aspects of cognition that were previously overly complicated for 'hand-designed' networks.

**Tasks include:**
- Go / RT Go / Dly Go
- Anti / RT Anti / Dly Anti
- DM 1 / DM 2
- Ctx DM 1 / Ctx DM 2
- MultiSen DM
- MultiSen Dly DM
- DMS / DNMS
- DMC / DNMC

The network learns potential mechanisms for multiple tasks, showing clustering and compositionality in high-dimensional state space.

---

## Quiz - Blackboard

**Question:** Even if backpropagation through time is not considered realistic, the b_______, d______, and r_______ that emerge in trained RNNs can be, and can offer valuable insights for understanding brain function.

**Answer:** behaviors, dynamics, and representations

---

## Recap

- Backpropagation through time is effective at assigning credit/blame to synapses, but is not biologically realistic

- Vanishing and exploding gradient problems stop RNNs learning from timepoints in the distant past

- Different architectures (light GRU, GRU, LSTM) tackle the vanishing gradient problem, but their biological relevance is still debated

- Biologically-realistic learning rules avoid storing all previous activity states, or retrieve them using hippocampal-replay-like mechanisms

- Even if the learning rule is not realistic, the trained network could give insights into how the brain performs cognitive tasks

---

## Still Curious? You Can Dive In Deeper to Any of Today's Topics:

### Light GRU
- Ravanelli, Mirco, Philemon Brakel, Maurizio Omologo, and Yoshua Bengio. "Light gated recurrent units for speech recognition." *IEEE Transactions on Emerging Topics in Computational Intelligence* 2, no. 2 (2018): 92-102.

### Normalisation in the Brain
- Carandini, Matteo, and David J. Heeger. "Normalization as a canonical neural computation." *Nature Reviews Neuroscience* 13, no. 1 (2012): 51-62.

### GRU
- Cho, Kyunghyun, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. "Learning phrase representations using RNN encoder-decoder for statistical machine translation." *arXiv preprint arXiv:1406.1078* (2014).

### LSTM
- Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." *Neural computation* 9, no. 8 (1997): 1735-1780.
- Proof of LSTM solving the vanishing gradient problem by Mitesh Khapra: https://www.youtube.com/watch?v=dKKPtG-qeaI&ab_channel=NPTEL-NOCIITM

### Backpropagation Through Time and the Brain
- Lillicrap, Timothy P., and Adam Santoro. "Backpropagation through time and the brain." *Current opinion in neurobiology* 55 (2019): 82-89.

### Biologically Reasonable Alternative to Backpropagation Through Time
- Murray, James M. "Local online learning in recurrent networks with random feedback." *Elife* 8 (2019): e43299.

### Context-Dependent Decision-Making in the Brain and RNNs
- Mante, Valerio, David Sussillo, Krishna V. Shenoy, and William T. Newsome. "Context-dependent computation by recurrent dynamics in prefrontal cortex." *Nature* 503, no. 7474 (2013): 78-84.

### Training an RNN to Perform 20 Cognitive Tasks
- Yang, Guangyu Robert, Madhura R. Joglekar, H. Francis Song, William T. Newsome, and Xiao-Jing Wang. "Task representations in neural networks trained to perform many cognitive tasks." *Nature neuroscience* 22, no. 2 (2019): 297-306.
