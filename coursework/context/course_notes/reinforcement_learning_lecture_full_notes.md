# Reinforcement Learning

**Lecturer:** Seán Froudist-Walsh
**Title:** Lecturer in Computational Neuroscience
**Institution:** University of Bristol
**Date:** 11/27/23

---

## Opening Quote

> "The greatest teacher, failure is"
>
> — Yoda

---

## Intended Learning Outcomes

By the end of this lecture, you will understand:

- The **goal of reinforcement learning (RL)**
- The **fundamental elements** of reinforcement learning
- **Algorithms for learning the values** of states and actions
- The **basics of reinforcement learning in the brain**
- **Brain-inspired advances** in deep reinforcement learning
- How the brain builds **cognitive maps** & its connection to **model-based RL**
- How reinforcement learning is beginning to tackle **meta-cognition** (learning-to-learn) and **curiosity**

---

## The Goal of Reinforcement Learning

**The goal of reinforcement learning is to maximise the total amount of (discounted) reward in the future**

### Return (Gt)

```
Gt = Rt+1 + γ Rt+2 + γ² Rt+3...

   = Σ(k=0 to T) γᵏ Rt+k+1
```

Where:
- **γ** (gamma) = discount factor
- The value of future rewards is **discounted** (often γ = 0.9) to reflect a preference for more immediate rewards
- **T** = time horizon

### Supervised Learning vs Reinforcement Learning

#### Supervised Learning
- The correct output is given by a mysterious **teacher**
- Error signal: difference between output and target
- Network learns from explicit correct answers

#### Reinforcement Learning
- **Rewards** (including negative rewards = punishments), and information about the current **state** are given by the environment
- Agent takes **actions** based on states
- No explicit "correct" answer, only rewards

**Images adapted from Rui Ponte Costa**

---

## The Elements of a Reinforcement Learning System

### Part 1: An Agent
**The thing that performs the task**, e.g., a robot, a neural network, etc.

### Part 2: The Environment
**The external system or setting** with which the learning agent interacts. It encompasses everything that the agent observes and responds to, but which is **outside the agent's control**.

### Part 3: A Reward Signal
**A single number** (which can be zero, positive or negative) sent from the environment to the agent on every timestep. The agent's **only goal is to maximize the total reward** it receives over the long run.

---

## Pavlovian (Classical) Conditioning
### Learning to Predict Stimuli & Rewards

**Pavlov, 1897**

**conditioning = learning**

### Four Stages of Classical Conditioning

#### 1. Before Conditioning
- **Unconditional stimulus** (Food) → **Unconditional response** (Salivation)
- **Neutral stimulus** (Bell ringing) → **No conditional response**

#### 2. Before Conditioning (repeated)
- Bell ringing → **Neutral stimulus**
- Food → **No salivation**

#### 3. During Conditioning
- **Bell ringing + Food** → **Salivation** (Unconditional response)

#### 4. After Conditioning
- **Bell ringing** → **Salivation** (now a **Conditional response**)
- The bell (now a **Conditional stimulus**) alone triggers salivation (**Conditional response**)

---

## The Elements of a Reinforcement Learning System (Part 4/5)

### Part 4: A Value Function

**A value function specifies what is good in the long run.**

#### State-Value Function

```
v(s) = 𝔼[Rt+1 + γ Rt+2 + γ² Rt+3... | St = s]
   = 𝔼[Σ(k=0 to T) γᵏ Rt+k+1 | St = s]
```

**The value of a state** is the total amount of (discounted) reward an agent can **expect** to accumulate over the future, **starting from that state**.

#### Bellman Equation

```
V(St) = 𝔼[Rt+1] + γ 𝔼[V(St+1)]
```

The **estimated value** of the **current state** is just the **discounted** estimated value of the **next state** plus the **expected reward** received during the transition to the next state.

**Reference:** Sutton & Barto, 2018

---

## The Temporal Difference (TD) Learning Model of Classical Conditioning

### TD Update Rule

```
V(St) ← V(St) + α [Rt+1 + γV(St+1) − V(St)]
```

Where:
- **V(St)** = estimated value of state at time t
- **α** = learning rate
- **Rt+1** = reward received
- **γ** = discount factor
- **V(St+1)** = estimated value of next state

### Key Components

**The TD model updates the estimated value of the previous state based on:**
1. **Surprising rewards**
2. **The value of the following state**

**The temporal difference error** (in neuroscience called the **reward prediction error**) captures:
- How much **higher/lower** a reward is than expected and/or
- If the following state's value is **higher/lower** than expected

**The estimated value of the previous state** is updated using the **reward prediction error (RPE**, aka **TD error**)

**The size of the update** is determined by the **learning rate** α.

**Reference:** Sutton & Barto, 2018

---

## The Reward Prediction Error Shifts Backwards in Time

### Trial 1 (Early Learning)
- **Conditional stimulus** (Bell ringing) appears at time **t**
- **Reward** (Food) appears at time **T**
- **Reward prediction error** shows a large spike at reward delivery

### Trial 200 (After Learning)
- **Conditional stimulus** appears at time **t**
- **Reward prediction error** now appears at the **conditional stimulus**, not the reward
- The RPE has **shifted backwards** from the reward to the **stimulus that predicts it**

This demonstrates how the value function learns to predict future rewards earlier and earlier in the sequence.

---

## Dopamine Strongly Resembles the Reward Prediction Error

### Dopamine Neuron Firing

**Schultz et al., Science, 1997**

#### No Prediction - Reward Occurs
- **(No CS)** → **R**
- Large dopamine response at reward delivery

#### Reward Predicted - Reward Occurs
- **CS** → **R**
- Dopamine response at CS, no response at reward
- The reward is now expected

### Dopamine Release

**Amo et al., Nature Neuroscience, 2022**

Odor-evoked dopamine release shown across days:
- **Day 1:** Response primarily at reward
- **Day 2-6:** Response shifts to odor (CS)

Time course measured in seconds (0-4s)

**Key Finding:** Dopamine encodes the **reward prediction error**, not the reward itself.

---

## Where Does Dopamine Target in the Brain?

### Primary Dopamine Target Regions

#### Striatum
- Deep brain structure
- Receives dense dopaminergic innervation
- Critical for action selection and habit learning

#### Orbitofrontal Cortex
- At the **top of the cortical hierarchy**
- Integrates value information
- Important for decision-making

### D1 Receptor Density
- Heat map showing density: **50** (low) to **96** (high)
- Highest density in striatum
- Moderate to high density in frontal cortex

### Brain Activity Resembles Reward-Prediction Error

**O'Doherty et al., Neuron, 2003**

Brain activity in dopamine-responsive regions resembles the reward-prediction error signal.

**References:**
- Froudist-Walsh et al., *Neuron*, 2021
- Froudist-Walsh, PhD Thesis, 2015
- Artist: Sofía Minguillón Hernández

---

## Blackboard Quiz 1

**Using the TD-learning, given the following reward sequence R and state space sequence S compute the value function for V(St=1), V(St=2) and V(St=3), for the first, second and third update steps, corresponding to three trials of the same sequences of states and rewards.**

**Assume that:**
- The value function starts at **0** for all states
- Learning rate **α = 1**
- Future discounting **γ = 0.5**

**Given:**
- S = {St=1, St=2, St=3}
- R = {Rt=1 = 0, Rt=2 = 0, Rt=3 = 0, Rt=4 = 0.5}

---

## Instrumental Conditioning
### Learning Which Actions to Take

**Thorndike, 1898, 1911**

### Thorndike's Law of Effect

> "Of several responses made to the same situation, those which are accompanied or closely followed by **satisfaction** to the animal will, other things being equal, be **more firmly connected** with the situation, so that, when it recurs, they will be more likely to recur;
>
> those which are accompanied or closely followed by **discomfort** to the animal will, other things being equal, have their connections with that situation **weakened**, so that, when it recurs, they will be less likely to occur.
>
> **The greater the satisfaction or discomfort, the greater the strengthening or weakening of the bond.**"

**Core Principle:** Learning from **trial and error**

### Experimental Evidence

Graph showing **Time required to escape** (seconds) vs **Trials**:
- Early trials: 600 seconds
- Later trials: ~100 seconds
- Shows clear learning curve

[Image shows Thorndike's puzzle box experiment]

---

## The Elements of a Reinforcement Learning System (Part 5/5)

### Part 5: A Policy

**A policy is a mapping from states to probabilities of selecting each possible action**

```
π(a|s) = P(At = a|St = s)
```

The **policy** gives the **probability of taking any action in any state**.

### Value Functions with Policies

In any system in which the agent takes actions, the value functions are defined **with respect to the policy being followed**.

```
vπ(s) = 𝔼π[Σ(k=0 to T) γᵏ Rt+k+1 | St = s]
```

The value of state s under policy π.

**Reference:** Sutton & Barto, 2018

---

## Action-Value Function

### Definition

```
qπ(s, a) = 𝔼π[Σ(k=0 to T) γᵏ Rt+k+1 | St = s, At = a]
```

**The value of taking an action a in a state s** is the total amount of (discounted) reward an agent can **expect** to accumulate over the future, **starting from taking action a in state s** and **following the policy π afterwards**.

### State Transitions

```
St → [At] → St+1
   (agent takes action)
   (agent receives reward Rt+1)
```

**Reference:** Sutton & Barto, 2018

---

## ε-Greedy Action Selection

### Algorithm

**At each state:**

1. **With probability 1-ε:** Select the action a with the highest value Q(s,a) – **(exploit)**
2. **With probability ε:** Select randomly among all actions **(explore)**

### Policy Definition

**ε-greedy combined with:**
- An estimate **Q(s,a)** of the action-value function
- **Defines a policy**

This balances **exploration** (trying new actions) with **exploitation** (using known good actions).

---

## Q-Learning

**Watkins, 1989**

### Q-Learning Update Rule

```
Q(St, At) ← Q(St, At) + α [Rt+1 + γ max_a Q(St+1, a) − Q(St, At)]
```

### Components

- **Q(St, At)** = the estimated value of an action taken in the previous state
- **α** = learning rate (determines size of update)
- **Rt+1** = reward received
- **γ** = discount factor
- **max_a Q(St+1, a)** = value of the **best available action** in the next state

### Key Decision Point

**To calculate the RPE, we need:**
1. The **old estimate** of the value of the state-action pair (before seeing the reward)
2. Subtract this from the **reward received** plus
3. The value of the following state-action pair

**But should we use:**
- The value of the next action **chosen**? → **On-policy methods** (e.g., SARSA)
- The value of the **best available action**? → **Off-policy methods** (e.g., Q-learning)

**What if the next action is exploratory/random?**

### Q-Values Storage

Q-values (value of any action in any state) are stored in a **look-up table**

[Table showing states S1-S4 and actions A1-A4 with Q-values]

**References:** Watkins, 1989; Sutton & Barto, 2018

---

## Blackboard Quiz 2

**What makes Q-learning different from TD learning?**

**Answer:** Q-learning learns action-values Q(s,a) instead of state-values V(s), and uses the maximum Q-value of the next state (off-policy) rather than the value of the next state.

---

## Action Values in the Brain

**Shin et al., eLife, 2021**

### Experimental Finding

**Activity in a neuron in the striatum correlates with the estimated value of choosing left (QL)**

### Behavioral Data

Graph showing:
- **Rat's choice** (probability of choosing left averaged over 10 trials)
- **Q-learning prediction** (probability of choosing left)

Strong correspondence between neural activity and Q-learning model predictions.

### Neural Firing Pattern

Neuron firing rate shows different levels for different Q-values:
- α=0.25
- β=0.5
- γ=0.6-0.75

Time from delay onset (s): -1, 0, 1, 2, 3

### Brain Regions Encoding Action Values

**Action-value signals found in neurons in:**
- **Striatum**
- **Orbitofrontal cortex**
- **Hippocampus** (not shown)

---

## What If the State-Space Is So Large That You Can't Explore It All?
### Deep Q Learning

### The Problem

- In real life, the **sensory input at any moment is slightly different** from all other moments
- Is each moment a state?
- Having values for each state & action in a **look-up table is infeasible**

### The Solution

**Use deep network as a Q(S,A; θ) function approximator** when in the presence of **high-dimensional state space**

### Making RL More Like Supervised Learning

#### Input
- Each **state is a sequence of frames** from a video game

#### Network Architecture
- **Convolutional layers** process visual input
- **Fully connected layers** compute values

#### Output
- The **estimated value for each possible action** from that state

#### Loss Function

```
L(θ) = 𝔼(s,a,r,s')~U(D) [(r + γ max_a' Q(s',a';θ⁻) − Q(s,a;θ))²]
```

Where:
- **target** = r + γ max_a' Q(s',a')

**References:**
- Lex Fridman
- Volodymyr Mnih
- Serrano Academy
- Mnih et al., *Nature*, 2015

---

## Replay Memory – A Brain-Inspired Trick to Make Deep Q Learning Work

### The Problem

1. **Consecutive states are highly correlated**
   - Learning 'on-line' would therefore be **inefficient**
2. Can get **stuck in local minima**

### The Solution

1. **Replay past experiences**
   - Store experiences (actions, state transitions & rewards) in a **memory bank**
   - **Replay these experiences randomly**, and use this for training

This breaks the correlation between consecutive samples and improves learning stability.

---

## Blackboard Quiz 3

**Briefly explain what is the problem in standard reinforcement learning methods that deep reinforcement learning methods address. How does it address it?**

**Answer:**
- **Problem:** Standard RL methods cannot handle high-dimensional state spaces (too many states to store in a lookup table)
- **Solution:** Deep RL uses neural networks as function approximators to estimate Q-values for any state, even unseen ones. Additionally, replay memory is used to break correlations between consecutive experiences.

---

## Hippocampal Replay & Memory Consolidation

### Place Cells

**O'Keefe & Dostrovsky, Brain research, 1971**

**The hippocampus contains place cells** – cells that fire preferentially in particular locations.

[Maze diagram showing place cell firing location]

### Replay During Sleep

**Wilson & McNaughton, Science, 1994; Ólafsdóttir et al., Current Biology, 2018**

**During sleep, and rest,** the neurons that fired during the day's behavior **reactivated in a similar sequence** in the hippocampus (and to some degree the cortex)

#### Run Activity
- Place cells A, B, C, D, E fire in sequence
- Time scale: 0-5 seconds

#### 'Forward' Replay
- Same sequence compressed in time
- Place cells B, C, D, E fire
- Time scale: 0-0.3 seconds

### Function

**This may drive memory consolidation** – transferring information from hippocampus to cortex for long-term storage.

---

## Above Human-Performance on Video Games..... That Don't Involve Planning

**Mnih et al., Nature, 2015**

### Performance Across Games

Graph showing **% of human level** performance:
- Many games: **>100%** (superhuman)
- Games requiring **planning**: much lower performance
- Examples:
  - **Breakout**
  - **Pong**
  - **Space Invaders** (with note about planning requirement)
  - **Montezuma's Revenge** (very low performance)

### Space Invaders Example

**DQN controls the green laser cannon to clear columns of aliens while preventing them from reaching the bottom** by moving left/right and shooting at the top of the screen

**Key Limitation:** Games that require **planning** show poor performance.

**Google Deepmind**

---

## The Elements of a Reinforcement Learning System (Part 6/5)

### Part 6: A Model of the Environment (Optional, but Awesome)

**A model of the environment is anything the agent can use to predict how the environment will respond to its actions**

### Model-Based Systems Can:

- **Simulate** the next state, an entire trial, or all possible trials
- **Learn value functions** from simulated experience and
- **Use this to improve its policy** for really interacting with that environment
- This entire process is known as **planning**

### Model-Free vs Model-Based

- An RL system **with a model** = **model-based system**
- An RL system **without a model** = **model-free system**

[Diagram showing agent imagining/planning with cheese and maze]

**Reference:** Mattar & Daw, *Nature Neuroscience*, 2018

---

## Blackboard Quiz 4

**Is Q-learning model-free or model-based?**

**Answer:** **Model-free**. Q-learning learns action-values directly from experience without building an explicit model of state transitions or rewards.

---

## Model-Free vs Model-Based Reinforcement Learning

### Model-Free RL
- **Habits**
- **Trial & error**
- **Behaviourism**

### Model-Based RL
- **Goal-directed behavior**
- **Cognitive maps**
- **Cognitive Psychology**

---

## Learning Without Reinforcement

### Behaviourism

**Thorndike, 1898** - Trial & error learning

**Key Tenets:**
- All behaviour are learned reactions to stimuli
- Ignore unobservable mental processes (outside the realm of science)

### Latent Learning

**Blodgett, 1929**

**Key Findings:**
- **The environment can be learned without explicit rewards or penalties**
- This is **not possible with model-free reinforcement learning**
- Animals can **learn associations between stimuli (states)** by experiencing sequences of stimuli as they explore an environment
- This was a **forerunner to Cognitive Psychology**, which broke off from behaviourism

[Graph showing learning curves for different groups - maze navigation]

---

## Assessing If People Use Model-Based or Model-Free Reinforcement Learning

**Daw et al., Neuron, 2011; Decker et al., Psych. Science, 2016**

### The Two-Step Task

#### Do people use model-free or model-based RL?

This is commonly assessed with this **two-step task:**

1. **First-stage choice:** Participants choose between two spaceships
2. **Probabilistic transition:** Followed by a transition to either:
   - Red planet (common or rare transition)
   - Purple planet (common or rare transition)
3. **Second-stage choice:** Choose between two aliens
4. **Outcome:** Rewarded with space treasure or not
5. **Reward probabilities change over time** for each alien

### Key Analysis

**If you are rewarded following a rare transition** (e.g., blue spaceship to purple planet), **on the next trial should you:**

#### Model-Free RL (Trial & Error Learning)
- Be **more likely to choose the blue spaceship**
- Simply repeat what worked

#### Model-Based RL (Planning)
- Be **less likely to choose the blue spaceship**
- Choose the option more likely to get you back to the purple planet (since that's where the reward was)

[Diagram showing spacecraft choices and planetary transitions with outcome graphs]

---

## Model-Based Learning Across Brains

**Decker et al., Psych. Science, 2016; Vikhbladh et al., Neuron, 2019**

### Key Findings

- **Adults** use a **mixture of model-based and model-free RL**
- **Children** use more **model-free RL**
- **Patients with lesions to the hippocampus** use more of a **model-free RL strategy**

### Developmental Trajectory

**Model-based RL:**
- Develops over **childhood/adolescence**
- Probably depends (in part) on the **hippocampus**

### Evidence from Two-Step Task

Graphs showing:
- **Common Transition vs Rare Transition** on Previous Trial
- **Outcome of Previous Trial** (Reward vs No Reward)

#### Children vs Adolescents vs Adults
- Children show primarily model-free pattern
- Development of model-based increases with age

#### Hippocampal Patients
Graph showing **aggregate accuracy**:
- **Model-based** behavior reduced in patients
- **Model-free** behavior intact

---

## Simulating Model-Based Learning

**Sutton & Barto, 2018**

### Dyna-Q Algorithm

**In the Dyna-Q model-based RL algorithm, learning and planning are accomplished by exactly the same algorithm, operating on:**
- **Real experience** for learning
- **Simulated experience** for planning

### Architecture

```
value/policy
   ↑    ↓
planning  acting
   ↓    ↑   ↓
 model ← experience
   ↓
model learning
```

### Tabular Dyna-Q Pseudocode

```
Initialize Q(s,a) and Model(s,a) for all s ∈ S and a ∈ A(s)
Do forever:
  (a) S ← current (nonterminal) state
  (b) Execute action A, observe resultant reward, R, and state, S'
  (c) Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]
  (d) Model(S,A) ← R,S' (assuming deterministic environment)
  (e) Repeat n times:
      S ← random previously observed state
      A ← random action previously taken in S
      R,S' ← Model(S,A)
      Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]
```

**n = planning steps** (number of simulated experiences per real experience)

### Model-Free Component

The Q-learning update **(highlighted in box)** is the model-free component used for both real and simulated experience.

### Performance

Graphs showing:
- **Episodes** vs performance
- Different numbers of planning steps (0, 5, 50 planning steps)
- More planning steps → faster learning

---

## Cognitive Map

**Tolman, Psychological Review, 1948**

### Experiment Setup

[Diagram showing maze with blocked and unblocked paths]

- **Apparatus used in preliminary training**
- **Apparatus used in test trial**

### Key Findings

- When the **usual route is blocked, and new routes are opened**, rats **choose the correct direction**
- They can do this **despite never having experienced that path before** (no trial-and-error learning)
- They are likely learning a **'map' of the environment in their mind** – a **'cognitive map'**
- In other words, the rats must have some mental **'model' of the environment**

**Reference:** (From E. C. Tolman, B. F. Ritchie and D. Kalish. Studies in spatial learning. I. Orientation and short-cut. J. exp. Psychol., 1946, 36, p. 17.)

---

## The Hippocampus and Entorhinal Cortex Contain a Cognitive Map

### Place Cells

**O'Keefe & Dostrovsky, Brain research, 1971**

**The hippocampus contains place cells** – cells that fire preferentially in particular locations.

[Maze image with place cell location marked]

### Grid Cells

**Hafting et al., Nature, 2008**

**The entorhinal cortex** (main input to hippocampus) **contains grid cells** - cells that fire at multiple evenly-spaced locations on a **hexagonal (or triangular) grid**.

[Hexagonal grid pattern showing firing locations]

### Application to RL

**Each location could be a state in a reinforcement learning problem**

### Recognition

**Nobel prize in physiology or medicine, 2014**
for "discovering the brain's GPS system"

---

## What About Problems That Don't Involve Navigating Space?

**Constantinescu et al., Science, 2016**

### Abstract State Space Experiment

#### Task Design

Participants had to navigate in **'bird space'**:
- Each bird differed in **neck and leg length**
- Each bird was associated with a particular **Christmas decoration**

#### Task
- Given one Christmas decoration, they had to **navigate, by changing neck and leg lengths**, to the target Christmas decoration

### Key Findings

Despite not being related to physical space, the **prefrontal cortex (vmPFC/OFC)** and **entorhinal cortex (ERH)** encoded this abstract space in a **hexagonal grid** (see slide on 'grid cells')

### Brain Regions

- **PCC** (Posterior Cingulate Cortex)
- **vmPFC/OFC** (ventromedial Prefrontal Cortex / Orbitofrontal Cortex) - α=8
- **ERH** (Entorhinal Cortex) - γ=4, z=14

### Interpretation

This suggests that the **same grid systems used to navigate physical space may be used to plan and navigate abstract problems**.

[Hexagonal modulation effect size plot shown]

---

## Recap - Week 1: The Most Effective Strategy to Improve Learning?

### Learning-to-Learn (Meta-cognition)

- **Identifying specific strategies** for planning, monitoring, and evaluating your own learning
- **Develop a repertoire of strategies** to choose from and the skills to select the most suitable strategy for a given learning task

### Self-Regulated Learning Components

1. **Cognition** – the mental process involved in knowing, understanding, and learning
2. **Metacognition** – often defined as 'learning to learn'
3. **Motivation** – willingness to engage our metacognitive and cognitive skills

**Source:** https://educationendowmentfoundation.org.uk/education-evidence/teaching-learning-toolkit

[Teaching and Learning Toolkit screenshot showing various interventions and their effect sizes]

---

## Learning to Reinforcement Learn

**J.X. Wang et al., Nature Neuroscience, 2018**

### Architecture

```
Maintenance gate
      ↓
Input → LSTM units → Output
gate              gate
      At    Vt    Rt+1
```

### Training

**LSTM units trained with model-free RL**

### Key Innovation

- **Weights frozen during test time** – any 'learning' occurs due to **dynamics of activity**, not changes to weights
- **Deep & recurrent neural networks can approximate any function** – including model-based RL
- **Model-free RL can be used to learn model-based RL** (learning to reinforcement learn)

### JX Wang Model Performance

Comparison showing:
- **Common Transition on Previous Trial**
- **Rare Transition on Previous Trial**

#### Model-Free Learner
Lower probability of staying with rewarded choice

#### Model-Based Learner
- Common/Uncommon patterns emerge
- **Stay probability:** Shows model-based behavior signature

Graph showing:
- **Pre-reward** vs **Unrewarded** conditions
- Clear distinction between strategies

---

## Learning to Learn - Monkeys

**Harlow, Psychol. Rev., 1949**

### The Rule

- One of the two stimuli (picture cards) is **rewarded**
- The other is **not**

### Learning Process

1. The monkey first learns by **trial-and-error** which stimulus is rewarded
2. Then the **stimuli are changed**
3. The monkey eventually **learns the rule**, enabling it to **'learn' how to respond to new stimuli after just a single trial**

### Performance Graph

**Performance (%)** vs **Trial**

Shows multiple episode ranges:
- 257-312 (highest performance)
- 201-256
- 101-200
- 1-100 (lowest initial performance)
- 17-24
- 9-16
- 1-8

Clear improvement in **learning to learn** across episode ranges.

[Image shows monkey working at experimental apparatus]

---

## Learning to Learn with ConvNets, RNNs & Reinforcement Learning

**Harlow, Psychol. Rev., 1949; J.X. Wang et al., Nature Neuroscience, 2018**

### Architecture

- A **ConvNet** is used to present different stimuli
- The **RNN (with LSTM units)** is trained as before with model-free RL
- The network **'learns to learn' more efficiently** – seemingly learning the rule, which is not possible under model-free RL

### Task Screens

[Three game screens shown with different stimuli combinations]

### Performance

#### Episode Range
Shows similar pattern to monkey experiments:
- 257-312
- 201-256
- 101-200
- 1-100
- 17-24
- 9-16
- 1-8

#### Training Quartile
Graph showing **Performance (% correct)** across trials:
- Final quartile (best)
- 5th
- 4th
- 3rd
- 2nd
- 1st (worst)

Network shows improvement in **one-shot learning** ability over training.

---

## Recap - Week 1: Active Engagement – Curiosity

### Definition of Curiosity

**Curiosity occurs whenever we detect a gap between what we want to know and what we already know.** (Loewenstein, 1994)

### Properties of Curiosity

- Curiosity is driven by the **value of acquiring information**
- **Greater curiosity is associated with greater academic achievement** (Shah et al., 2018)

**Example:** Neil deGrasse Tyson (astrophysicist & science communicator)
"Kids are born scientists."

---

## Curiosity – An Intrinsic Reward Signal

**Kulkarni et al., NeurIPS, 2016; Gottlieb & Oudeyer, Nature Rev. Neurosci., 2018**

### The Problem

**How to deal with environments that give feedback rarely?**

### The Solution

**Curiosity – an intrinsic reward!**

### Hierarchical Architecture

```
action → External Environment → observations
            ↓
         extrinsic reward
            ↓
        Meta-Controller → goal
            ↓
Qt(st,gt;θt)
            ↓
        Controller → intrinsic reward ← Critic
            ↓
        st  st+1  st+N
            yt    yt+1
```

### Components

1. **Meta-controller** chooses a **goal** to maximise future extrinsic reward
2. **Controller** chooses among **actions** according to the current goal
3. **Critic** gives **'intrinsic reward'** to controller if the current goal is complete

### Performance on Montezuma's Revenge

Graph showing **Total extrinsic reward** vs **Steps (1e6)**:
- Steady increase to ~400 points
- Much better than Mnih et al. deep Q network (compare with DQN results)

**Key Result:** Curious agents learn better in environments in which rewards are uncommon

---

## Blackboard Quiz 5

**Provide a definition of curiosity from a reinforcement learning point of view.**

**Answer:** Curiosity in RL is an intrinsic reward signal that motivates agents to explore novel states or acquire new information, especially in environments where extrinsic rewards are sparse. It can be implemented as a bonus reward for visiting new states or reducing uncertainty about the environment.

---

## Recap

### Summary of Key Concepts

1. **The goal of reinforcement learning** is to maximise the total amount of (discounted) reward in the future

2. **The temporal difference (TD) learning model** successfully explains many aspects of classical conditioning (state-value prediction)

3. **The reward prediction error from TD-learning** strongly correlates with dopamine neuron activity in the brain

4. **Q-learning** is a successful off-policy algorithm for learning action-values

5. **Deep RL** approximates the value function – enabling RL in environments with many states

6. **Replay of past memories** (like in the hippocampus) helps deep RL to learn

7. **A model of the environment** is anything the agent can use to predict how the environment will react to its actions

8. **Models of the environment ('cognitive maps')** are built in the hippocampus & orbitofrontal cortex (and potentially other areas)

9. **An intrinsic reward signal** – like curiosity – can help agents learn in environments that rarely give out rewards

---

## Further Reading (Optional)

### The Foundational Book on Reinforcement Learning
- **Sutton, Richard S., and Andrew G. Barto.** *Reinforcement learning: An introduction.* MIT press, 2018 (2nd edition).

### The Main Paper Describing the Link Between TD RPE & Dopamine
- **Schultz, Wolfram, Peter Dayan, and P. Read Montague.** "A neural substrate of prediction and reward." *Science* 275, no. 5306 (1997): 1593-1599.

### Deep Reinforcement Learning
- **Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves et al.** "Human-level control through deep reinforcement learning." *Nature* 518, no. 7540 (2015): 529-533.

### Hippocampal Replay
- **Wilson, Matthew A., and Bruce L. McNaughton.** "Reactivation of hippocampal ensemble memories during sleep." *Science* 265, no. 5172 (1994): 676-679.

### Cognitive Maps
- **Tolman, Edward C.** "Cognitive maps in rats and men." *Psychological review* 55, no. 4 (1948): 189.

### Learning to Reinforcement Learn
- **Wang, Jane X., Zeb Kurth-Nelson, Dharshan Kumaran, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Demis Hassabis, and Matthew Botvinick.** "Prefrontal cortex as a meta-reinforcement learning system." *Nature neuroscience* 21, no. 6 (2018): 860-868.

### Curiosity & Intrinsic Rewards
- **Kulkarni, Tejas D., Karthik Narasimhan, Ardavan Saeedi, and Josh Tenenbaum.** "Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation." *Advances in neural information processing systems* 29 (2016)
