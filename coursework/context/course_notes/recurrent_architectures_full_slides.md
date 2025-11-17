# Recurrent Neural Networks in Brains and Machines

**Lecturer:** Seán Froudist-Walsh
**Title:** Lecturer in Computational Neuroscience
**Institution:** University of Bristol
**Date:** 12/22/23

---

## Opening Question

### Why have recurrent connections?

---

## A Popular Working Memory Task (n-back)

### Instructions

- We will present a series of letters on the screen, one at a time
- Whenever the letter on the screen is the **same as the previous letter**, say 'now'

### Example Sequence
O, G, B, F, X, F, C

---

## Feedforward Neural Networks Cannot Do This Easy Task

### Challenge

**But do you fancy a challenge?**

Rewind the video, and now respond any time the letter on the screen is the same as the one shown **three letters back** (e.g., A, B, C, A).

**Key Problem:** Feedforward networks have no mechanism to maintain information about previous inputs. They process each input independently without memory of past states.

---

## Intended Learning Outcomes

By the end of this video you will be able to:

- **Explain why recurrent connections can be useful**
- **Describe where most recurrent connections are in the brain**
- **Compare biologically-realistic and trained recurrent neural networks**
- **Design a dynamical model of the whole cortex step-by-step**
- **Write the equation for AI-style recurrent neural networks (RNNs) and leaky RNNs**
- **Formally compare feedforward and recurrent neural networks**

---

## What Are Recurrent Connections?

### Definition

**Recurrent connections are neural pathways that enable information to:**
- Travel in loops
- Be maintained
- Be integrated
- Re-enter the system

### Types of Recurrent Connections

#### Local vs Long-Range
- **Within a brain area (local)** - connections between neurons in the same region
- **Between brain areas (long-range)** - connections across different brain regions

#### Excitatory vs Inhibitory
- Can be either excitatory or inhibitory

### Visual Examples

**E1 and E2** (Excitatory populations):
- Show local recurrent connections within population
- Bidirectional connections between populations

**F1 and F2** (Different configuration):
- More complex recurrent connectivity patterns
- Multiple feedback loops

**Reference:** Byrne, 2023
https://nba.uth.tmc.edu/neuroscience/m/s1/introduction.html

---

## Where Are the Recurrent Connections in the Brain?

### Key Statistics

- **Many long-distance recurrent loops** across brain areas
- In **primates**: approximately **2/3 of all possible connections** between brain areas do exist
- In **mice**: ~**97%** of possible connections exist
- However, **most connections (~80%)** are from **within the same brain area** (in the cortex)

### Connectivity Patterns

**Connection Density by Distance:**
- Graph showing exponential decay
- r² = Cexp(−λd)
- r² = 0.92 (R=0.96)
- Most connections are local (<0.5 mm)
- Rapidly decreasing with distance

**Connectivity Matrix:**
Shows interconnections between brain areas with hierarchical organization

**Brain Network Diagram:**
- Sensory input pathways
- Feedback/recurrent connections
- Multiple processing hierarchies

**References:**
- Markov et al., *Cerebral Cortex*, 2011
- Markov et al., *J. Comp. Neurol.*, 2014
- Stefanics et al., *Front. Hum. Neurosci*, 2014

---

## A Parallel Tradition of Biologically-Realistic Models of Brain Functions

### Chapter Topics from Theoretical Neuroscience Books

**"Neuronal Dynamics"** sections include:
- Firing Rate Models
- Spiking Neuron Networks and Recurrent Networks
- Plasticity and Coding
- Neural Oscillations Transformation
- Recurrent Networks and Associative Memory
- Plasticity and Learning
- Selective Amplification
- Short-term Plasticity
- Reward-Based Learning
- Excitatory-Inhibitory Networks
- Management Feedback and Inhibitory Feedback
- Biophysical Models
- The Generalize HH-
- The Oscillatory Model of Simple Cells in Primary Visual Cortex
- A Recurrent Model of Simple Cells in Primary Visual Cortex
- Integrate-and-Fire Model
- Sustained Activity
- Threshold and Neural Network Encoding

**"IV Dynamics of Cognition"** sections:
- 16 Competing Populations and Decision Making
- 17 Memory and Attractor Dynamics
- 18 Cortical Field Models for Perception
- 19 Synaptic Plasticity and Learning
- 20 Outlook: Dynamics in Plastic Networks

### Applications to Brain Functions
- **V1** - vision
- **Hippocampus** - memory
- **Prefrontal cortex** - working memory
- **Prefrontal/parietal** - choice
- **Hippocampus** - memory

---

## Biologically-Realistic Models Link Physiology to Behaviour

### Model Types

#### Detailed Spiking Models
- Can be detailed models of many **spiking neurons**
- Or simplified (via **mean-field techniques**) to **firing rate models** of populations of cells

#### Spiking Network Model
**Diagram showing:**
- Network of interconnected spiking neurons
- Individual spike trains
- Population activity patterns
- Trial-by-trial variability

### Strengths and Limitations

#### Detailed Spiking Models
- Lead to **clear experimental predictions**
- Can simulate realistic neural dynamics

#### Simplified Firing Rate Models
- Allow **mathematical analysis**
- Provide **deeper intuitions**
- More computationally efficient

### Success Record

**Both approaches have been successful at:**
- Explaining neural activity
- Predicting behavior

### Comparison with AI-Style RNNs

#### Main Implementation Difference
Parameters are **not trained** but:
- Taken directly from neuroscience experiments, OR
- Chosen to represent a hypothesis about how the brain computes

#### Current Trade-offs
- **Biologically-realistic models:** More directly testable predictions for experiments
- **Trained RNNs:** More capable of high-level functions

### Key References
- Amit & Brunel, *Cerebral Cortex*, 1997
- Wang, *Neuron*, 2002
- Wong & Wang, *J. Neurosci.*, 2006

**Contributors:** Bobby Ines

---

## Blackboard Quiz 1

**Question:** Discuss the pros and cons of using biologically-realistic models versus trained RNNs to simulate a cognitive task.

---

## Most Brain Functions Emerge from Interactions of Many Areas
### A Challenge for Local RNNs

**Steinmetz et al., Nature, 2019**

### Revolution in Neural Recording Technology

**Neuropixels probe:**
Graph showing exponential growth in simultaneously recorded neurons:
- 1950s: ~10 neurons
- 1970s: ~100 neurons
- 1990s: ~100 neurons
- 2010s: ~1000 neurons (dramatic increase)

**Jun et al., Nature, 2017**
**Image credit:** Matteo Carandini

### Neural Encoding Across Brain Areas

**Three types of encoding measured:**

#### Visual Encoding
- Neurons (%) showing selectivity across different brain areas
- Distributed across V1, visual areas, and higher regions

#### Action Encoding
- Different pattern of distribution
- More prominent in motor and frontal areas

#### Choice Encoding
- Highest in frontal and parietal regions
- Sparse in early sensory areas

**Brain Areas Analyzed:**
- **V1** (primary visual cortex)
- **PFC** (prefrontal cortex)
- **M1** (primary motor cortex)
- **S1** (primary somatosensory cortex)
- **Hippocampus**
- **Subcortical** regions

**Key Insight:** Different cognitive functions recruit neurons distributed across many brain areas, not just localized regions.

---

## How Do We Build a Dynamical Model of the Whole Cortex?

### Step 1/6: Building Blocks – Local Circuit Models

---

## Canonical Local Cortical Circuit

### The Isocortex Hypothesis

> "Our view is that the rapid evolutionary expansion of neocortex has been made possible by building an '**isocortex**' — a structure that uses **repeats of the same basic local circuits throughout a single [cortical] sheet**."
>
> — RJ Douglas and KA Martin (2012)

**Key Concept:** The cortex is built from repeating units of the same basic circuit motif, allowing rapid evolutionary expansion through replication rather than invention of new circuit types.

---

## A Canonical Local Circuit Model

### Circuit Components

**Two Population Model:**
- **E** (Excitatory population)
- **I** (Inhibitory population)

### Mathematical Formulation

```
τ drx,E/dt = −rx,E + βE[wE,E rx,E + Σy∈Y Cy→x ry,E − wE,I rx,I]+
```

#### Interpretation

**At each step in time, the firing rate of the Excitatory neurons in area x would drop down towards zero if not for synaptic inputs.**

#### Activity is Pushed Up By:
- **Positive connections** from the Excitatory population to itself (`wE,E rx,E`)
- **Long-range inputs** from other brain areas y (`Σy∈Y Cy→x ry,E`)

#### Activity is Pushed Down By:
- **Negative inputs** from the Inhibitory population (`−wE,I rx,I`)

### Key Parameters

#### Input-Output Relationship
**Current → β (slope) → Firing rate**

- **The slope (β)** determines the size of the response to input
- **The time constant (τ)** determines how quickly the rate can rise and fall in response to input
- **Firing rates cannot be negative** (rectification)

### Parameter Sources

**w, β, G set to match:**
- Binzegger et al (2009) experimental data

**Reference:** Chaudhuri et al., *Neuron*, 2015

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

### Step 2/6: Connect the Blocks – Anatomical Connectivity Data

---

## Modeling Large-Scale Resting-State Networks

### Model Architecture

**Synaptic-level model parameters:**
- **w** = local recurrent coupling
- **G** = global coupling strength across nodes

**Local Circuit Diagram:**
```
    W   G
  ┌─────┐
E─┤  E  │──▶
  │  I  │
  └─────┘
```

### Data Integration

**Three Data Types:**

#### 1. Structural Connectivity
- **Constrain** model with anatomical connection matrix
- Shows which areas connect to which

#### 2. Functional Connectivity
- **Fitting** empirical fMRI correlation patterns
- Measures statistical dependencies

#### 3. Empirical Data
- **Gene expression**
- **Cellular data** (receptor densities, etc.)

### Equations

```
τ drx,E/dt = −rx,E + βE[wE,E rx,E + Σy Cy→x ry,E − wE,I rx,I]+
```

**Credit:** Gustavo Deco and others

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

**Step 2/6:** Connect the blocks – anatomical connectivity data ✓

### Step 3/6: (Large-Scale 2.0) Allow Local Variation of Circuit Properties, Based on Data

---

## Brain Areas Do Not Have an Identical Local Circuitry

### Dendritic Spine Variation

**Microscopy Image:** Shows pyramidal neurons with different dendritic complexity

**Spine = site of excitatory synapse**

### Hierarchical Relationship

**Graph:** Spine count vs Hierarchical position

- **X-axis:** Hierarchical position (0 to 1)
- **Y-axis:** Spine count
- **r² = 0.90** (very strong correlation)

**Brain Areas Labeled:**
- **V1, V2** (low hierarchy, ~0 position, low spine count)
- **V4** (intermediate)
- **TEO** (higher)
- **5**, **4**, **10**, **TE**, **12-7**, **24** (highest hierarchy, highest spine count)

**Key Finding:** The number of excitatory synapses per neuron systematically increases with hierarchical position in the cortex.

**Visual Input vs Somatosensory Input:**
- Visual pathway: V1 at bottom
- Somatosensory pathway: different hierarchy

**Reference:** Elston, 2007

---

## A Quantitative Change to Parameters Driven by Anatomy

### Modified Equation

```
τ drx,E/dt = −rx,E + βx,E[(1+Δx) (wE,E rx,E + G Σy∈Y Cy→x ry,E) + σηt] − wE,I rx,I]+
```

**Where Δx scales up the strength of excitatory inputs to an area based on the spine count (number of excitatory synapses per neuron)**

### Hierarchical Gradient

**Graph showing:**
- **r² = 0.90** correlation
- **7500** maximum spine count
- Brain areas: V1, V2, V4, TEO, 5, 4, 10, TE, 12-7, 24, 6

### Circuit Diagram

**E-I Network:**
```
    ┌───┐
    │ E │──┐
    └───┘  │
      ↓    │ (1+Δx) factor
    ┌───┐  │
    │ I │←─┘
    └───┘
```

### Key Innovation

**Only two free parameters** in the entire model!

The hierarchical variation is determined by data (spine counts), not free fitting.

**Reference:** Chaudhuri et al., *Neuron*, 2015

---

## The Emergence of a Hierarchy of Temporal Windows

### Time Constant Variation

**20 ms → 100 ms → 1 s → 10 s**

**τfitted** varies across brain areas

### Experimental Results

**Graph A: Fitted time constants**
Network diagram showing:
- V1, V2, V4 (fast, 20-100ms)
- 8l, TEO, TEpd (intermediate, 100ms-1s)
- 9/46d, 7A, 24c (slow, 1-10s)

**Graph B: Input Response**
- Shows firing rate (Hz) vs Time (s)
- Input pulse
- Different decay rates for different areas

**Graph C: Change in Firing Rate**
Response to perturbation showing different temporal dynamics

**Graph D: Autocorrelation**
- V1 (fastest decay)
- V2, V4 (fast)
- TEO, TEpd (medium)
- Multiple prefrontal areas (slowest decay)
- Shows intrinsic timescales increasing with hierarchy

### Brain Area Network

Complex connectivity diagram showing:
- 9/46v, 9/46d, 46d, 8B
- 8m, 8l, ProM, PBr
- F1, F2, F5, F7
- 2, 5, 7m, 7A, 7B
- DP, MT, V4, V2, V1
- STPc, STPi, STPr, TEpd, TEO

**Reference:** Chaudhuri et al., *Neuron*, 2015

---

## Rapid Responses in Early Visual Areas

### V1 Receptive Field Organization

**Brain Diagram:**
- **LGN** (lateral geniculate nucleus)
- **After calories** (label showing location)
- **Primary made of cells** (?)
- **Small V1**
- Visual pathway from retina

### Neural Response Patterns

**"On"-center ganglion cell** vs **"Off"-center ganglion cell**

#### (A) Light spot in center
- On-center: **Strong response** (rapid firing)
- Off-center: No response

#### (B) Dark spot in center
- On-center: No response
- Off-center: **Strong response**

#### (C) Light spot in surround
- On-center: Suppressed
- Off-center: Suppressed

#### (D) Diffuse light, covering both center and surround
- On-center: Weak response
- Off-center: Weak response

**Stimulus timing:** 0 - Stimulus on - Stimulus off

**Key Feature:** Very rapid, transient responses (tens to hundreds of milliseconds)

**Credit:** slide: XJ Wang

---

## Neural Integrators with Long Time Constants

### Persistent Firing

**Graph:** Firing rate (Hz) vs Time from stimulus onset (ms)

- **X-axis:** -500 to 2500 ms
- **Y-axis:** 0 to 30 Hz
- **Three example neurons** (red, black, green traces)

**Pattern:**
- Brief stimulus presentation
- Firing rate ramps up during stimulus
- **Persistent activity continues for >2 seconds** after stimulus offset
- Different neurons show different levels of persistent activity

**Key Feature:** Activity persists long after stimulus is gone, enabling working memory.

**Credit:** slide: XJ Wang

---

## Visual System: Hierarchy of Timescales Across Species

### Monkey

**Graph:** Integration timescale (log ms) vs Cortical area

**Brain areas** (low to high hierarchy):
- **MT** (motion processing)
- **LIP** (lateral intraparietal)
- **LPFC** (lateral prefrontal cortex)
- **OFC** (orbitofrontal cortex)
- **ACC** (anterior cingulate cortex)

**Anatomical Hierarchy:**
```
   ACC
    ↑
   OFC
    ↑
   LPFC
    ↑
    LIP
    ↑
    MT
```

**Data:** Paatelmann, Freeman, Vinck, Moskovitch, Wulff, Perez-Schoonbroodt

**r² correlation** shown with trend line

**Reference:** Murray et al., *Nature Neurosci.* 2014

### Mice (Neuropixels Survey)

**Graph:** Autocorrelation timescale (ms)

**Brain areas:** LGN, V1, LM, AL, RL, AM, PM, MMA

**Anatomical hierarchy score** (-0.5 to 0.5)
- Strong positive correlation
- Error bars showing variability

**Reference:** Siegle et al., *Nature* 2021

### Human

**Diagram:** Processing hierarchy

**Temporal receptive window (TRW):**

**Short (msec)** → **Medium (sec)** → **Long (min)**

**Brain representations:**
- **Small dots** (phonemes) - shortest timescale
- **Medium dots** (words) - medium timescale
- **Larger dots** (sentences, paragraphs, narrative) - longest timescale

**Key Concept:** Information processing integrates over increasingly long timescales as you move up the cortical hierarchy.

**Reference:** Hasson et al., *Trends in Cogn. Sci.* 2015

---

## How Do Qualitatively Different Functions Emerge Across the Cortex?

[Blank slide - likely introduces next concept]

---

## Vector Field Visualization

**Prof. Ghrist**

[Image shows a complex 2D vector field with circular/spiral patterns in blue, showing dynamic system flow patterns]

**Key Concept:** Dynamical systems can be visualized as vector fields showing how the system state evolves over time.

---

## Working Memory Emerges with Sufficiently Strong Recurrent Connections

### Bifurcation Diagram

**Graph:** Firing rate (Hz) vs Synaptic strength relative to a baseline

**X-axis:** 1.9 to 2.3
**Y-axis:** 0 to 50 Hz

**Two Regimes:**

#### 1. Spontaneous Activity (Below Threshold)
- **Firing rate:** ~10 Hz
- System returns to baseline after stimulus
- No persistent activity

#### 2. Persistent Activity (Above Threshold)
- **Threshold** at ~2.0 synaptic strength
- **Firing rate:** Can sustain 20-50 Hz
- Activity persists after stimulus removal
- Working memory enabled

**Key Concept:**
**Bifurcations in nonlinear dynamical systems:**
**Graded differences give rise to qualitatively novel behavior/functions**

**Credit:** slide: XJ Wang

---

## Experimental Evidence for Working Memory

### A. Recording Area

**Brain diagram:**
- **AS** (arcuate sulcus)
- **PS** (principal sulcus)
- Recording location in prefrontal cortex

### B. ODR Task (Oculomotor Delayed Response)

**Task Structure:**
```
Fixation → Cue → Delay → Response
```

**Visual display:**
- Central fixation point
- Peripheral cue locations (135°, 90°, 45°, 180°, 0°, 225°, 270°, 315°)
- Delay period (blank screen with fixation)
- Response (arrow indicating saccade)

### C. Delay Cell with Persistent Firing

**Multiple trial rasters and PSTHs:**

Shows neural responses across different cue locations:
- Trials aligned to different epochs
- Clear sustained firing during delay period
- Direction-selective persistent activity
- Different firing rates for different remembered locations

**Key Features:**
- Cue-specific responses
- Maintained activity during delay (working memory)
- Response selectivity preserved across delay

**References:**
- Funahashi et al., 1989
- Constantinidis et al., 2018

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

**Step 2/6:** Connect the blocks – anatomical connectivity data ✓

**Step 3/6:** (Large-scale 2.0) Allow local variation of circuit properties, based on data ✓

### Step 4/6: (Large-Scale 2.0) Simulate Task Stimuli as Input & Measure Activity

---

## Working Memory Activity Across Cortical Areas and Multiple Cell Types

### Task Structure

**Delayed Match-to-Sample Task:**
```
Target → Delay 1 → Distractor → Delay 2 → Probe
  ■              -           ■           -        ■  ■
```

### Brain Model

**3D cortical surface model** showing distributed activity patterns

### Local Circuit Model

**Detailed circuit diagram:**

```
    ┌──┐E1├──┐
    │CB1│    │
    ├──┤     │E2
    │CB2│    │
    ├──┤  ├──┤
    │PV │  │CB2│
    ├──┤  ├──┤
    │VIP│  │PV │
    │CR1│  ├──┤
    ├──┤  │VIP│
    │CR2│  │CR2│
    └──┘  └──┘
```

**Cell Types:**
- **E1, E2** - Excitatory populations
- **CB1, CB2** - Calbindin/SST inhibitory neurons (dendritic targeting)
- **PV** - Parvalbumin inhibitory neurons (somatic targeting)
- **VIP/CR1, CR2** - VIP/Calretinin inhibitory interneurons

**References:**
- Froudist-Walsh et al., *Neuron*, 2021
- See also: Mejías & Wang, *eLife*, 2022

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

**Step 2/6:** Connect the blocks – anatomical connectivity data ✓

**Step 3/6:** (Large-scale 2.0) Allow local variation of circuit properties, based on data ✓

**Step 4/6:** (Large-scale 2.0) Simulate task stimuli as input & measure activity ✓

### Step 5/6: (Large-Scale 2.0) Validate Model Against Real Neural Data

---

## The Model Captures the Persistent Activity Pattern of >90 Experimental Studies

### Mega-Analysis of Experimental Data

**Leavitt et al., TiCS, 2017**

**Two brain views showing:**
- **Persistent activity** (teal/blue regions)
- **No persistent activity** (tan/beige regions)

### Model Simulation

**Froudist-Walsh et al., Neuron, 2021**

**Two brain views showing:**
- Model predictions match experimental patterns
- **p = 0.0001** (highly significant correspondence)

**Key Result:** The model successfully predicts which brain areas show persistent working memory activity, validated against a meta-analysis of >90 experimental studies.

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

**Step 2/6:** Connect the blocks – anatomical connectivity data ✓

**Step 3/6:** (Large-scale 2.0) Allow local variation of circuit properties, based on data ✓

**Step 4/6:** (Large-scale 2.0) Simulate task stimuli as input & measure activity ✓

**Step 5/6:** (Large-scale 2.0) Validate model against real neural data ✓

### Step 6/6: (Large-Scale 2.0) Make Predictions for Future Experiments

---

## "Zoom In" to Find the Cell-Type Responsible for Distractor-Resistance

### Task Structure

```
Target → Delay 1 → Distractor → Delay 2
```

**Trial Time →**

### Circuit Manipulation

**Local circuit with cell types:**
```
E1 ─┬→ E2
    │
CB/SST1 → CB/SST2
    ↓         ↓
   PV    ←   PV
    ↓         ↓
CR/VIP1   CR/VIP2
```

### Key Experiment

**A) VIP - SST Population Balance**

**Graph:** VIP - SST population firing rate (Hz)
- **Higher VIP** (top)
- **Higher SST** (bottom)

### B) Distractor-Resistant Condition (DA = 1.5)

**Time series for all cell types:**
```
E1, E2, SST/CB1, SST/CB2, PV, VIP/CR1, VIP/CR2
```

Shows:
- **Target selective populations** maintain activity
- **Distractor selective populations** suppressed
- Activity persists through distractor presentation

### C) Transient Inhibition

**Detail of critical period during distractor**

### D) Distractible Condition

Different activity pattern - memory disrupted

### E-H) Dopamine Modulation

**Brain views showing:**
- **46d** region highlighted
- **DA = 1.5:** Distractor-resistant
- **DA = 2.0:** Distractible

**With increasing D1 stimulation:**
- Changes balance of excitation/inhibition
- Shifts from resistant to distractible

### Mechanism

**CB2 inhibition** (highlighted in circuit):
- SST/CB neurons provide dendritic inhibition
- Critical for filtering distractors
- VIP neurons modulate SST activity
- Balance determines distractor resistance

**Reference:** Froudist-Walsh et al., *Neuron*, 2021

---

## How Do We Build a Dynamical Model of the Whole Cortex?

**Step 1/6:** Building blocks – local circuit models ✓

**Step 2/6:** Connect the blocks – anatomical connectivity data ✓

**Step 3/6:** (Large-scale 2.0) Allow local variation of circuit properties, based on data ✓

**Step 4/6:** (Large-scale 2.0) Simulate task stimuli as input & measure activity ✓

**Step 5/6:** (Large-scale 2.0) Validate model against real neural data ✓

**Step 6/6:** **(Large-Scale 2.0) Make Predictions for Future Experiments** ✓

---

## Now You Have a Dynamical Model of the Whole Cortex and Predictions for New Experiments

[Completion slide]

---

## Blackboard Quiz 2

**What is the sequence of steps for building a whole-cortex model?**

**Answer:**
1. Building blocks – local circuit models
2. Connect the blocks – anatomical connectivity data
3. Allow local variation of circuit properties, based on data
4. Simulate task stimuli as input & measure activity
5. Validate model against real neural data
6. Make predictions for future experiments

---

## Recurrent Neural Networks (RNNs) in AI

### The Basic ("Vanilla") RNN

**Elman, Cog. Sci., 1990**

#### Network Architecture

```
x (inputs)
   ↓
a (hidden layer) ←┐
   ↓             │
o (outputs)      │
                 │
         a(t-1)──┘
```

#### Mathematical Formulation

**Hidden layer activity:**
```
a(t) = f(Wx→a x(t) + Wa→a a(t-1) + b1)
```

**The hidden layer activity is a nonlinear function (e.g. ReLU, sigmoid) of:**
- A **weighted sum of the inputs** (Wx→a x(t))
- Plus a **bias** (b1)
- And a **weighted sum of the hidden layer activity from one timestep ago** (Wa→a a(t-1))

**Output layer activity:**
```
o(t) = g(Wa→o a(t) + b2)
```

**The output layer activity:**
- Is a **weighted sum of the hidden layer activity** (Wa→o a(t))
- Plus a **bias** (b2)
- Passed through a **nonlinear function** g

**Key Feature:** The recurrent connection (a(t-1) → a(t)) allows the network to maintain information over time.

---

## The Mathematical Equivalence of (Some) Feedforward and Recurrent Neural Networks

**van Bergen & Kriegeskorte, Curr. Opin. Neurobiol, 2020**

### Four Equivalent Representations

#### 1. Feedforward
```
input → area 1 → area 2 → area 3 → output
```

Simple hierarchical processing

#### 2. Recurrent
```
input → area 1 ⇄ area 2 ⇄ area 3 → output
```

Bidirectional connections between areas

#### 3. Recurrent, Unrolled in Time
```
t = 1       t = 2       t = 3

input → area 1 → area 2 → area 3 → output
           ↓       ↓       ↓
        area 1 → area 2 → area 3
                   ↓       ↓
                area 1 → area 2 → area 3
```

Shows how recurrent network evolves over discrete time steps

#### 4. Recurrent, Unrolled in Space
```
area 3 (t=3)
    ↓
area 3 (t=2)
    ↓
area 3 (t=1)
    ↓
area 2 (t=2)
    ↓
area 2 (t=1)
    ↓
area 1 (t=1)
    ↓
input
```

**Question:** Unrealistic depth?

### Key Insight

**Some recurrent networks can be mathematically equivalent to very deep feedforward networks.** However, the biological plausibility differs:
- **Recurrent:** Biologically realistic (brain has recurrent connections)
- **Feedforward (deep):** May require unrealistic number of layers

---

## Leaky RNNs for Neuro-AI

### Architecture

```
x (inputs)
   ↓
a (hidden layer) ←┐
   ↓       leak   │
o (outputs)       │
                  │
          a(t-1)──┘
```

### Continuous Time Version (Written in Papers)

```
τ da/dt = -a(t-1) + f(Wx→a x(t) + Wa→a a(t-1) + b1)
```

### Discrete Time Version (Euler Method - Written in Code)

```
a(t) = a(t-1) + (Δt/τ)(-a(t-1) + f(Wx→a x(t) + Wa→a a(t-1) + b1))
```

### Parameters

#### Δt (Simulation Timestep)
- The length of the simulation timestep
- Logically, **shorter timesteps lead to smaller changes in activity between timesteps**

#### τ (Neuronal Time Constant)
- Dictates **how rapidly the activity changes in response to inputs/leak**
- Larger τ → slower dynamics
- Smaller τ → faster dynamics

#### Leak Term
- **-a(t-1)** causes activity to decay back to zero
- Without input, activity exponentially decays with time constant τ

### Output

```
o(t) = g(Wa→o a(t) + b2)
```

---

## Blackboard Quiz 3

**Comparing feedforward and recurrent neural networks.**

**Key differences:**
1. **Recurrent networks** have connections that feed back to earlier layers or the same layer
2. **Recurrent networks** can maintain information over time (memory)
3. **Recurrent networks** process sequential data naturally
4. **Feedforward networks** process each input independently
5. Some **recurrent networks can be unrolled in time** to create equivalent deep feedforward networks
6. **Recurrent networks** are more biologically realistic

---

## Recap

### Key Concepts

1. **Recurrent connections enable neural networks to use and compute on previous information**, which contributes to many cognitive functions, including memory and decision-making

2. **The whole cortex is recurrent**, but **most recurrent connections are from other neurons within the same brain area**

3. **Biologically-realistic models** make more directly testable predictions for experiments, but are **less capable of high-level functions** compared to trained RNNs

4. **Most brain functions rely on activity across many different brain areas**

5. **Whole cortex models are built up by:**
   - Connecting many local circuits
   - Adjusting parameters across areas according to anatomy

6. **There is an increase in the time constants of neurons along the cortical (visual and auditory) hierarchy**

7. **RNNs are like feedforward networks, but with inputs from the previous timestep**

8. **Some feedforward and recurrent neural networks are mathematically equivalent**

9. **In neuroscience, leaky RNNs are often used** to allow for more biologically-realistic neural dynamics

---

## Further Reading (Optional)

### Most Cortical Connections Are From Within the Same Brain Area
- **Markov, Nikola T., P. Misery, Arnaud Falchier, C. Lamy, J. Vezoli, R. Quilodran, M. A. Gariel et al.** "Weight consistency specifies regularities of macaque cortical networks." *Cerebral cortex* 21, no. 6 (2011): 1254-1272.

### An Early Cortex-Wide Dynamical Model
- **Chaudhuri, Rishidev, Kenneth Knoblauch, Marie-Alice Gariel, Henry Kennedy, and Xiao-Jing Wang.** "A large-scale circuit mechanism for hierarchical dynamical processing in the primate cortex." *Neuron* 88, no. 2 (2015): 419-431.

### A Cortex-Wide Dynamical Model of Working Memory
- **Froudist-Walsh, Sean, Daniel P. Bliss, Xingyu Ding, Lucija Rapan, Meiqi Niu, Kenneth Knoblauch, Karl Zilles, Henry Kennedy, Nicola Palomero-Gallagher, and Xiao-Jing Wang.** "A dopamine gradient controls access to distributed working memory in the large-scale monkey cortex." *Neuron* 109, no. 21 (2021): 3500-3520.

### Comparing Feedforward and Recurrent Neural Networks
- **van Bergen, Ruben S., and Nikolaus Kriegeskorte.** "Going in circles is the way forward: the role of recurrence in visual inference." *Current Opinion in Neurobiology* 65 (2020): 176-193.

### The "Vanilla" RNN Paper
- **Elman, Jeffrey L.** "Finding structure in time." *Cognitive science* 14, no. 2 (1990): 179-211.

---

## Next: Training Deep Neural Networks and Comparisons with Brains

**Video:** James DiCarlo (MIT)

**Reference:** Yamins et al., *PNAS*, 2014

[Image showing visual processing hierarchy from eye through RGC, LGN, V1, V2, V4, to IT, with neural representations and mental perceptual states including "giraffe", "kiwi", "building", "strawberry"]
