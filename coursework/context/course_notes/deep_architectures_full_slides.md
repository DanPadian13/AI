# Deep Neural Networks in Brains and Machines

**Lecturer:** Seán Froudist-Walsh
**Title:** Lecturer in Computational Neuroscience
**Institution:** University of Bristol
**Date:** 12/22/23

---

## Opening Question

**Just how similar are deep neural networks in brains and machines?**

### Comparison:
- **Monkey brain** (Markov et al., J. Comp. Neurol., 2014)
  - Complex hierarchical structure with 10 levels
  - Multiple brain areas (V1, V2, V4, TEO, TH/TF, etc.)
  - Intricate connectivity patterns between areas

- **Artificial neural network** (LeCun et al., Nature, 2015)
  - Input units
  - Hidden units H1
  - Hidden units H2
  - Output units
  - Feedforward connections between layers

---

## Intended Learning Outcomes

By the end of this video you will be able to:
- Describe the major types of neurons in the brain
- Describe how distinct features of the brain's anatomy and physiology change along a deep hierarchy of areas
- Illustrate the basic architecture of deep neural networks
- Critically compare the similarities and differences of deep and convolutional neural networks in brains and machines

---

## The Anatomy of a Single Biological Neuron

### Historical Context
- **Ramón y Cajal**: Pioneer in understanding neuronal structure
- Used staining techniques to visualize individual neurons

### Neuron Components
1. **Dendrites (inputs)**
   - Dendritic branches receive incoming signals
   - Complex branching patterns
   - Act as input receivers

2. **Cell body (soma)**
   - Contains nucleus
   - Integrates signals from dendrites

3. **Axon (output)**
   - Transmits signals to other neurons
   - Can extend long distances
   - Myelin-wrapped for faster transmission

---

## A Biological Neuron as a Two-Layer Artificial Neural Network?

**Research:** Poirazi et al., Neuron, 2003

### Key Findings:
- Individual biological neurons can perform complex computations
- Dendritic branches (355 μm and 212 μm shown) act as separate computational units
- Each dendrite can perform its own nonlinear transformations
- The soma integrates outputs from multiple dendritic units

### Mathematical Representation:
- Multiple dendritic inputs (n₁, n₂, n₃, ... n₃₇) converge
- Each dendrite applies weights (α₁, α₂, etc.)
- Nonlinear function g at soma
- Final output y

### Performance Results:
- Linear model: r² = 0.82
- Sigmoid model: r² = 0.94
- **Conclusion:** Biological neurons are more complex than simple artificial units

---

## Neurons Are Not All the Same

### Major Division: Excitatory vs. Inhibitory

**Dale's Law:** A neuron releases the same neurotransmitter(s) at all its outgoing synapses.*

*Note: Laws in biology never strictly work, there are always exceptions! However, Dale's law is still a good first approximation to reality.

### Excitatory Neurons
- **Neurotransmitter:** Glutamate
- **Effect:** Makes the postsynaptic neuron **more likely to spike**
- **Properties:**
  - Pyramidal cell morphology
  - Long-range projections
  - Complex dendritic trees

### Inhibitory Neurons
- **Neurotransmitter:** GABA (gamma-aminobutyric acid)
- **Effect:** Makes the postsynaptic neuron **less likely to spike**
- **Properties:**
  - Various morphologies (basket, chandelier, etc.)
  - Local connections
  - Critical for circuit regulation

### Morphological Diversity Along Cortical Hierarchy

**Research:** Elston, Ev. Nerv. Sys., 2007; Jiang et al., Science, 2015

**Observation:** Excitatory neurons change shape along the visual hierarchy:
- **V1** (primary visual cortex): Simpler dendritic trees, smaller
- **V2**: More complex
- **V4**: Even more elaborate dendritic arbors
- **TEO**: Most complex dendritic structures

**Implication:** Higher-level neurons have greater computational capacity

---

## Quiz Question 1

**Compare and contrast biological neurons and units (neurons) in artificial neural networks. What may be some functional implications of these differences?**

### Key Differences:
1. **Computational Complexity**
   - Biological: Each neuron is a complex computational unit (potentially equivalent to a 2-layer network)
   - Artificial: Simple weighted sum + nonlinearity

2. **Diversity**
   - Biological: Excitatory vs. inhibitory, multiple subtypes
   - Artificial: Typically homogeneous units

3. **Spatial Structure**
   - Biological: Dendritic trees with spatial processing
   - Artificial: Point-like units

4. **Temporal Dynamics**
   - Biological: Rich temporal dynamics, spiking behavior
   - Artificial: Typically static activations (in feedforward nets)

---

## Sensory Cortex Has a Deep Hierarchical Structure

### What Changes as You Step Up the Hierarchy?

**Hierarchical Organization:**
- Multiple levels (1-10+)
- Areas organized from sensory (low) to cognitive (high)
- Systematic changes in properties across levels

**Key Areas in Visual Hierarchy:**
- **Level 1-2:** V1, V2 (early visual processing)
- **Level 3-5:** V4, V3, BL, etc. (intermediate processing)
- **Level 6-8:** TEO, MT, DP, etc. (higher-level features)
- **Level 9-10:** TH/TF, MST, STPc, 7A, FST, V3A, LIP, etc. (complex object/motion processing)

---

## The Time for a Neuron to Respond to a Stimulus Increases

**Research:** Thorpe & Faber-Thorpe, Science, 2001

### Processing Timeline:
1. **Retina:** 20-40 ms
   - Simple visual forms: edges, corners

2. **LGN (Lateral Geniculate Nucleus):** 30-50 ms

3. **V1 (Primary Visual Cortex):** 40-60 ms
   - Simple visual forms

4. **V2:** 50-70 ms
   - Intermediate visual forms

5. **V4:** 60-80 ms

6. **AIT (Anterior Inferotemporal cortex):** 70-90 ms
   - Intermediate visual forms, feature groups, etc.

7. **PIT:** 80-100 ms
   - High level object descriptions, faces, objects

8. **PFC (Prefrontal Cortex):** 100-130 ms

9. **PMC (Premotor Cortex):** 120-160 ms
   - Categorical judgments, decision making

10. **MC (Motor Cortex):** 140-190 ms
    - Motor command

**Key Principle:** Response latency increases as you move up the hierarchy

---

## The Types of Stimuli That Activate a Neuron Change

**Research:** Manassi et al., J. Vision, 2013

### Changes Along the Visual Hierarchy:

#### Receptive Field Size:
- **V1:** Small receptive fields (edges and lines)
- **V2:** Slightly larger (shapes)
- **V4:** Larger still (objects)
- **IT (Inferotemporal):** Very large (faces)

#### Features Detected:
1. **V1:** Edges and lines (simple oriented features)
2. **V2:** Shapes (combinations of edges)
3. **V4:** Objects (complex shapes)
4. **IT:** Faces (highly complex, category-specific)

### Two Key Properties Increase Up the Hierarchy:

1. **Spatial Invariance**
   - Neurons respond to stimuli in more positions in visual space
   - Receptive fields become larger

2. **Feature Complexity**
   - From simple (lines) to complex (people, faces, objects)
   - Increasing abstraction of representation

**Diagram Shows:**
- Brain with LGN, V1, V2, V4, IT labeled
- Progression from "shapes" → "faces and objects"
- Visual field representations becoming larger and more complex

---

## The Number of Neurons Per Area Goes Down

**Research:**
- Froudist-Walsh et al., eLife, 2018
- Collins et al., PNAS, 2010

### Neuronal Density Map:
- Color-coded brain showing neuronal density
- Scale: 4.5 to 10.0 × 10⁷ neurons

### Specific Examples:
- **~300 million neurons** in area V1 (primary visual cortex)
- **~3 million neurons** in area TH (high-level association cortex)

### Interpretation:
**"This suggests that the sensory information is compressed higher up the cortical hierarchy"**

**Implications:**
1. Dimensional reduction occurs along the hierarchy
2. Information is compressed into more abstract representations
3. Fewer neurons encode more complex, invariant features
4. Efficient coding principle: complex information with fewer units

---

## Increased Number of Inputs

**Research:** Elston, Evolution of Nervous Systems, 2007

### Dendritic Complexity Along the Hierarchy:

**Visual Examples:** Pyramidal neurons from different cortical areas showing increasing dendritic complexity

**Quantitative Data:**
- Graph showing "No. of spines" (synaptic inputs) across cortical areas
- Macaque monkey data
- **V1:** ~1,000 spines
- **V2:** ~2,000 spines
- **V4:** ~3,000 spines
- **TEO:** ~6,000 spines (highest level shown)

### Key Finding:
**As you ascend the cortical hierarchy:**
- Number of dendritic spines increases dramatically
- More synaptic inputs per neuron
- Greater integration capacity
- Higher computational complexity

**Relationship:**
- Higher areas have both MORE inputs per neuron AND larger receptive fields
- This allows integration of information from broader spatial regions and multiple features

---

## Increased Capacity for Modulation

**Research:** Froudist-Walsh et al., Nature Neuroscience, 2023

### Neurotransmitter Receptor Distribution:

#### Correlation Analysis:
- **X-axis:** Receptors per neuron (normalised)
- **Y-axis:** Cortical hierarchy
- **Correlation:** r = 0.81 (very strong positive correlation)

#### Brain Maps Showing Receptor Density:
- Left hemisphere and right hemisphere views
- Color scale: -7.5 (low, blue) to 2.5 (high, red)
- **Pattern:** Higher cortical areas (red) have more receptors per neuron
- Lower cortical areas (blue) have fewer receptors per neuron

### Types of Receptors:
This includes various neuromodulatory receptors:
- Dopamine receptors
- Serotonin receptors
- Acetylcholine receptors
- GABA receptors
- Glutamate receptors

### Functional Implications:
1. **Greater modulatory capacity** in higher cortical areas
2. More flexible information processing
3. Enhanced ability to adapt to context
4. Increased capacity for attention, arousal, and cognitive control
5. Higher areas are more "tunable" by brain state

**Conclusion:** The brain's neuromodulatory systems can exert greater influence on higher cognitive areas compared to early sensory areas.

---

## Different Sensory Modalities, Different Hierarchies

**Research:** Felleman & Van Essen, Cerebral Cortex, 1991

### Visual Hierarchy:
- Extremely complex network
- Many interconnected areas
- Multiple parallel processing streams
- Hierarchical organization with many intermediate levels

### Touch Hierarchy (Somatosensory):
- Simpler organization
- Fewer areas
- More direct pathway from periphery to higher levels
- Still hierarchical but less complex than vision

### Key Insight:
- **Different sensory systems have different architectural organizations**
- Visual system: Most complex (reflects importance of vision in primates)
- Somatosensory: More streamlined
- Auditory (not shown): Intermediate complexity

**Why the difference?**
- Different computational demands
- Different evolutionary pressures
- Different types of information to process

---

## Feedback Connections

### Importance of Feedback:
**Not all connections go "forward" up the hierarchy**

**Research Areas:**

#### 1. Selective Attention (Tirin Moore)
- **Bottom-up vs. Top-down processing**
- Brain areas: V1/V2, FEF (Frontal Eye Field), LIP, SC, LGN
- Feedback connections from FEF to V1/V2 modulate sensory processing
- Attention enhances processing of attended locations

#### 2. Predictions (Rao & Ballard, Nature Neuroscience, 1999)
**Research:** Stefanics et al., Front. Hum. Neurosci, 2014

**Predictive Coding Framework:**
- Higher areas send **predictions** down to lower areas (feedback)
- Lower areas send **prediction errors** (mismatch response) up to higher areas (feedforward)
- Brain constantly tries to minimize prediction errors

**Diagram shows:**
- Sensory input enters at bottom
- Multiple levels of processing (B → E → B → E)
- **Blue arrows (predictions):** Flow from higher to lower areas
- **Red arrows (prediction errors/mismatch response):** Flow from lower to higher areas
- Iterative refinement of predictions

#### 3. Context
- Feedback provides contextual information
- Helps interpret ambiguous stimuli
- Influences lower-level processing based on higher-level knowledge

### Key Principle:
**The brain is not just a feedforward system!**
- Massive feedback connections (~40% of all cortical connections)
- Reciprocal connectivity between areas
- Dynamic interplay between bottom-up and top-down processing

---

## If We Are Interested in Modeling Cognition, Should We Build Models with a Different Architecture?

### Graph Analysis:

**X-axis:** Hierarchy rank (brain areas from sensory to cognitive)
- Early sensory areas: V1, V2, V3, V4, etc.
- Intermediate: MT, TEO, etc.
- High-level cognitive: Multiple frontal and parietal areas

**Y-axis:** Hierarchy value (0 to 1)

### Two Regions:

#### 1. **Sensory Region** (steep rise)
- Sharp increase in hierarchy value
- Clear hierarchical organization
- Well-defined levels
- Strong anatomical hierarchy

#### 2. **Cognitive Region** (plateau)
- Flatter curve
- Less clear hierarchical organization
- Multiple areas at similar levels
- More parallel, distributed processing?

### Debate Question:
**Is there a strict hierarchy in cognitive cortex?**

- **For hierarchical view:** Some gradation still exists
- **Against hierarchical view:** Much flatter than sensory systems
- **Alternative:** Network/hub architecture rather than strict hierarchy

**Implications for AI:**
- Should cognitive models use different architectures than sensory models?
- Perhaps more recurrent, less strictly hierarchical?
- More lateral connections between "cognitive" areas?

---

## Quiz Question 2

**How do neurons at the bottom and top of the hierarchy differ in the brain and in machines? What (if any) could be the benefits of incorporating more of the missing biological features into artificial neural networks?**

### Differences in the Brain:

| Feature | Bottom of Hierarchy (V1) | Top of Hierarchy (IT/PFC) |
|---------|-------------------------|---------------------------|
| Response latency | Short (40-60 ms) | Long (100-190 ms) |
| Receptive field size | Small | Large |
| Feature complexity | Simple (edges, lines) | Complex (faces, objects) |
| Number of neurons | Many (~300M in V1) | Fewer (~3M in TH) |
| Inputs per neuron | Fewer (~1,000 spines) | Many (~6,000 spines) |
| Receptor density | Lower | Higher |
| Modulation capacity | Lower | Higher |

### Differences in Machines:

**Standard Deep Neural Networks:**
- Units are typically **homogeneous** across layers
- Same computational rules at each layer
- No biological-style diversity
- No excitatory/inhibitory distinction

### Potential Benefits of Adding Biological Features:

1. **Diverse neuron types (excitatory/inhibitory)**
   - Better balance between activation and suppression
   - More stable learning dynamics
   - Better pattern separation

2. **Increased complexity at higher layers**
   - More inputs per unit in higher layers
   - Greater integration capacity
   - Better abstraction

3. **Feedback connections**
   - Predictive coding
   - Context-dependent processing
   - Better handling of ambiguity

4. **Neuromodulation**
   - Adaptive learning rates
   - Context-dependent processing
   - Attention mechanisms

5. **Dendritic computation**
   - Each unit as a 2-layer network
   - Greater computational capacity per unit
   - More efficient networks (fewer units needed)

---

## A Neuron (Unit) in an Artificial Neural Network

**Historical Background:** McCulloch & Pitts, Bull. Math. Biophys., 1943

### Mathematical Formulation:

#### Basic Operation:
```
g(x₁, x₂, x₃, ... xₙ) = g(x) = Σⁿᵢ₌₁ xᵢ
```

#### With Weights:
Multiple inputs (x₁, x₂, x₃, x₄, xₙ) each with weights (w₁, w₂, w₃, w₄)
- First layer (g): weighted sum of inputs
- Second layer (f): applies activation function

#### Activation Function:
```
y = f(g(x)) = {1 if g(x) ≥ θ
              {0 if otherwise
```

Where θ is the **thresholding parameter**

### Modern Activation Function:

**Sigmoid Function:**
```
y = 1 / (1 + e^(-(w^T x+b)))
```

**Properties:**
- Smooth, differentiable
- Outputs between 0 and 1
- S-shaped curve
- Allows gradient-based learning

**Visualization (Lederer, arXiv, 2021):**
- Shows various activation functions
- Sigmoid highlighted
- Comparison with other nonlinearities

**Credit:** Niranjan Kumar for visualizations

### Key Insight:
Artificial neurons are **much simpler** than biological neurons:
- Single nonlinearity
- No dendritic computation
- No distinction between excitation/inhibition (except sign of weights)
- Point process (no spatial structure)

---

## XOR – Linear Decoding & AI Winter

**Research:** Summerfield, 2018

### The XOR Problem:

**XOR (Exclusive OR) Truth Table:**
```
x₁ = -1, x₂ = +1  →  Output: A (class 1)
x₁ = +1, x₂ = +1  →  Output: B (class 2)
x₁ = -1, x₂ = -1  →  Output: A (class 1)
x₁ = +1, x₂ = -1  →  Output: B (class 2)
```

### Visualization:
- 2D space with x₁ and x₂ axes
- Class A (green) and Class B (orange) points
- **Problem:** Cannot draw a single straight line to separate the classes
- This is a **linearly inseparable** problem

### Historical Significance:

**1969:** Minsky & Papert showed that single-layer perceptrons cannot solve XOR

**Impact:**
- Led to "AI Winter" (period of reduced funding and interest in AI)
- Seemed to show fundamental limitations of neural networks
- Damaging to the field for nearly two decades

**Resolution:**
- Multi-layer networks CAN solve XOR
- But required new training methods (backpropagation)
- Didn't become practical until 1980s

---

## Introducing Depth (Hidden Layers)

**Research:** Summerfield, How to build a brain from scratch, 2018

### Network Architecture:

```
X → W₁ → H = h(X · W₁) → W₂ → Y = h(H · W₂) → p(cat) = Φ[Y]
   input   hidden            output        decision
```

### Key Components:

1. **Input (X):** Original data
2. **First weights (W₁):** Transform input to hidden layer
3. **Hidden layer (H):** Intermediate representation
   - H = h(X · W₁)
   - Where h is a nonlinear activation function
4. **Second weights (W₂):** Transform hidden to output
5. **Output (Y):** Final representation
   - Y = h(H · W₂)
6. **Decision:** Classification based on Y

### Solution to XOR:

**Visualization:**
- Same XOR problem (green A and orange B points)
- With hidden layer, network can create **curved decision boundaries**
- Now the classes ARE separable!

### Key Principles:

**"Depth & non-linearities helps the network solve complex classification problems."**

**"The output Y is now computed not directly from X, but by passing X through another set of weights."**

### Why It Works:
1. **Hidden layer creates new feature space**
   - Non-linear transformation of input
   - More powerful representation

2. **Multiple layers allow arbitrary decision boundaries**
   - Can approximate any function (universal approximation theorem)
   - With enough hidden units

3. **Hierarchical feature learning**
   - First layer: simple features
   - Deeper layers: more complex features

---

## A Dog is a Dog, Wherever It Is

### Translation Invariance Problem:

**Grid of dog images** showing dogs in different:
- Positions
- Sizes
- Poses
- Backgrounds

### Biological Solution:

**Key Observation:**
- Neurons in the dog-sensing part of the brain respond to dogs whether they are close, far, or at any position in space
- This is called **translation invariance**
- Response won't vary much (invariance) if you move (translate) the dog

### Problem with Fully-Connected Networks:

**Limitation:**
- Neurons in feedforward neural networks that are fully connected (such as those studied so far) do NOT have translation invariance emerge naturally
- Each pixel position is treated independently
- Network must learn the same feature at every possible location
- Hugely inefficient and prone to overfitting

### Needed Solution:
- Architecture that builds in translation invariance
- Shared features across spatial locations
- This leads to → **Convolutional Neural Networks**

---

## How the Brain Builds Translation Invariance

**Research:**
- Manassi et al., J. Vision, 2013
- Carandini, J. Physiol., 2006
- Movshon et al., J. Physiol., 1978a,b

### Brain Hierarchy and Receptive Fields:

#### Visual Hierarchy Diagram:
- **V1:** Small receptive fields (edges and lines)
- **V2:** Slightly larger (shapes)
- **V4:** Even larger (objects)
- **IT:** Very large (faces)

**Features column shows complexity increasing:**
- Edges and lines → shapes → objects → faces

### Two Types of Cells:

#### A. Simple Cell:
- **Receptive field:** Specific location and orientation
- Image → Receptive field → Threshold → Response
- **Characteristic:** Position-dependent

#### B. Complex Cell:
- **Multiple simple cell inputs** with different positions
- Same orientation preference but different locations
- Image feeds into multiple simple cells
- Outputs combined (summed) → Response
- **Characteristic:** Position-invariant (within limits)

### Hierarchical Process:

1. **Simple cells in V1:** Detect features at specific locations
2. **Complex cells pool over simple cells:** Gain some translation invariance
3. **Higher areas pool over complex cells:** Even greater invariance
4. **IT neurons:** Respond to objects across much of visual field

**Key Principle:**
**Translation invariance is built hierarchically through pooling operations**

---

## Convolutions – Receptive Fields and Features

### Detailed Explanation:

**"The CNN builds in an algorithmic feature that ensures translation invariance, and it does so by copying a salient feature of biological visual systems"**

#### Key Biological Feature:
- Rather than each neuron receiving data from every single input location (e.g., image pixel)
- Network units have **spatially selective receptive fields**
- They learn a (local) filter specific to a location in space

#### Example:
- 100 × 100 pixel image
- A unit in the first hidden layer might receive inputs only from the first **5 × 5 square of pixels** in the top left corner

#### Critical Innovation: **Weight Sharing**
- Each filter is "shared" across multiple regions of space
- Resulting activations stacked along a separate dimension
- As if each unit were not a single neuron, but a **bank of neurons** with:
  - **Distinct receptive field locations**
  - **Same tuning properties**

### Mathematical Example:

#### Receptive Field (Feature/Filter):
```
1   0  -1
1   0  -1
1   0  -1
```
(Detects vertical edges)

#### Input Image:
```
3  0  1  2  7  4
1  5  8  9  3  1
2  7  2  5  1  3
0  1  3  1  7  8
4  2  1  6  2  8
2  4  5  2  3  9
```

#### Convolution Operation (*):
Slide the filter across the image and compute element-wise multiplication and sum

#### Output (Feature Map):
```
-5  -4   0   8
-10 -2   2   3
0   -2  -4  -7
-3  -2  -3  -16
```

### Visual Hierarchy Connection:
- Brain diagram showing V1, V2, V4, IT
- **Receptive fields size** column
- **Features** column showing progression

**Visualization:** Filter detecting vertical edges (shown as dark/light bands)

---

## Additional CNN Features (Continued)

### 1. Pooling (Downsampling):

**Purpose:** Reduce dimensionality at each layer

**Example - Max Pooling:**
```
Input:
3  0  1  2
1  5  8  9
0  1  3  1
2  4  5  2

Output:
5  9
4  5
```

**Operation:** Take maximum value in each 2×2 region

**Biological Parallel:**
- Dimensionality reduction in the primate visual system
- Fewer neurons at higher levels (~300M in V1 → ~3M in TH)
- Compression increases efficiency of image representation

### 2. Normalization (Gain Control):

**Formula:**
```
bᵢ = aᵢ / (k + α · Σ aⱼ²)^β
```

**Process:**
- Input → (÷) → Response
- Other inputs feed into divisive normalization
- Divide by average of nearby neurons

**Purpose:** Accentuate differences

**Biological Mechanism:**
- Canonical feature of computation in neural circuits
- Lateral inhibition among simple cells in V1
- Local average activation used for normalization

**Research:** Carandini & Heeger, Nature Reviews Neuroscience, 2012

**Brain Image:** Showing neuronal density (related to compression)

### 3. Fully-Connected Layers:

**Architecture:**
- Convolutional layers (early)
- Pooling layers (interspersed)
- Fully-connected layers (late)
- Similar to multi-layer perceptron (MLP) at the end

### Canonical Computation:

**Key Insight:**
- Each layer of the CNN is a **repeating motif** with similar form
- Just like in neocortex, a simple algorithm (implemented in the canonical microcircuit) is repeated at each processing stage
- **Complex behavior emerges from a succession of simple operations**

---

## A Full Convolutional Neural Network

### Complete Architecture:

**Pipeline visualization:**

1. **Input Image** (e.g., dog photo)
   ↓
2. **Convolution** (feature extraction)
   - Multiple feature maps
   - Learn edge detectors, texture detectors, etc.
   ↓
3. **Pooling + Normalisation** (dimensionality reduction)
   - Smaller spatial dimensions
   - Enhanced features
   ↓
4. **Convolution** (higher-level features)
   - More abstract feature maps
   - Combinations of lower-level features
   ↓
5. **Pooling + Normalisation** (further compression)
   - Even smaller spatial dimensions
   ↓
6. **Fully-connected** (classification)
   - Final layers
   - Output probabilities for classes:
     □ dog
     □ cat
     □ lion
     □ bird

### Key Features:

1. **Progressive abstraction:** Simple features → Complex features
2. **Progressive compression:** Large images → Small feature vectors
3. **Translation invariance:** Through convolution and pooling
4. **Hierarchical processing:** Like biological vision
5. **End-to-end learning:** All weights learned jointly

---

## Quiz Question 3

**Practically, convolutional neural networks seem to be less likely to overfit the training data, and generalize better to new data, compared to fully-connected deep neural networks. Why do you think this is?**

### Answer:

#### 1. **Parameter Sharing (Weight Sharing)**
- **Fully-connected:** Every neuron has its own weights for every input
  - For 100×100 image → 10,000 inputs per neuron
  - Many neurons → millions of parameters
- **Convolutional:** Same filter used across all spatial locations
  - 5×5 filter → only 25 parameters (plus bias)
  - Same 25 parameters used everywhere
- **Result:** Dramatically fewer parameters to learn

#### 2. **Built-in Translation Invariance**
- CNNs don't need to learn the same feature at every location separately
- Automatically generalize across positions
- Matches structure of visual problems (objects can appear anywhere)

#### 3. **Local Connectivity**
- Each unit only connects to small local region
- Matches local structure of images (nearby pixels are related)
- Prevents learning spurious long-range correlations

#### 4. **Hierarchical Feature Learning**
- Forces network to build complex features from simpler ones
- Provides good inductive bias for visual problems
- More structured learning process

#### 5. **Dimensionality Reduction Through Pooling**
- Progressively reduces spatial dimensions
- Creates compact representations
- Less opportunity to memorize training data

#### 6. **Regularization Through Architecture**
- The convolutional structure itself acts as regularization
- Constrains the hypothesis space
- Only considers functions with appropriate symmetries

**Summary:** CNNs have **fewer parameters** and **better inductive biases** for visual problems, both of which reduce overfitting.

---

## Coming Soon: Training Deep Neural Networks and Comparisons with Brains

### Preview Image (Yamins et al., PNAS, 2014):

**Diagram showing:**
- Patterns of photon energy applied (input images, e.g., Easter Island statues)
- Eye
- RGC (Retinal Ganglion Cells)
- LGN (Lateral Geniculate Nucleus)
- V1, V2, V4, IT (visual hierarchy)
- Neural states (actual brain recordings)
- Mental perceptual states:
  - "giraffe"
  - "kiwi"
  - "building"
  - "strawberry"

**Key Question:** How well do artificial neural networks predict actual brain responses?

**Video:** James DiCarlo (MIT) - pioneering work comparing CNN representations to primate visual cortex

---

## Recap

### Biological Neurons:
- **More complex than artificial units:** Capable of more computation
- **Diverse types:** Many different excitatory and inhibitory types with different connectivity patterns
- **Multiple sensory hierarchies:** Different for vision, touch, audition, etc.

### Changes Along Cortical Hierarchies:

**As you move up from early sensory cortex to higher areas:**
1. **Response time increases** (40 ms → 190 ms)
2. **Receptive field gets bigger** (small local → large, even whole visual field)
3. **Preferred stimulus gets more complex** (edges → objects and faces)
4. **Number of neurons gets smaller** (dimensionality goes down)
   - V1: ~300M neurons
   - High-level areas: ~3M neurons
5. **Number of inputs per neuron goes up** (1,000 → 6,000 spines)
6. **Capacity for modulation goes up** (more receptors per neuron)

### Cognitive Cortex:
- **Debated hierarchy:** Less clear hierarchical organization than sensory systems
- May require different modeling approaches

### Artificial Neural Networks:

#### Deep Neural Networks:
- **Depth & non-linearities** allow solving complex classification problems
- Can learn hierarchical feature representations

#### Convolutional Neural Networks:
**Directly inspired by biological vision:**
1. **Hierarchical structure** (like cortical hierarchy)
2. **Receptive fields** (local connectivity)
3. **Feature detectors** (learned filters)
4. **Pooling** (dimensionality reduction)
5. **Normalization** (gain control)

**Key Innovation:** Weight sharing creates translation invariance

---

## Still Curious? You Can Dive in Deeper to Any of Today's Topics:

### Hierarchies in the Brain's Anatomical Connections:
- **Felleman, Daniel J., and David C. Van Essen.** "Distributed hierarchical processing in the primate cerebral cortex." *Cerebral cortex (New York, NY: 1991)* 1, no. 1 (1991): 1-47.
- **Markov, Nikola T., Julien Vezoli, Pascal Chameau, Arnaud Falchier, René Quilodran, Cyril Huissoud, Camille Lamy et al.** "Anatomy of hierarchy: feedforward and feedback pathways in macaque visual cortex." *Journal of Comparative Neurology* 522, no. 1 (2014): 225-259.

### Biological Neurons as 2-Layer Neural Networks:
- **Poirazi, Panayiota, Terrence Brannon, and Bartlett W. Mel.** "Pyramidal neuron as two-layer neural network." *Neuron* 37, no. 6 (2003): 989-999.

### Predictive Coding:
- **Rao, Rajesh PN, and Dana H. Ballard.** "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature neuroscience* 2, no. 1 (1999): 79-87.

### Selective Attention:
- **Noudoost, Behrad, Mindy H. Chang, Nicholas A. Steinmetz, and Tirin Moore.** "Top-down control of visual attention." *Current opinion in neurobiology* 20, no. 2 (2010): 183-190.

### Increased Number of Inputs Per Neuron Along the Cortical Hierarchy:
- **Elston, Guy N.** "Specialization of the neocortical pyramidal cell during primate evolution." *Evolution of nervous systems* (2007): 191-242.

### Convolutions in the Brain:
- **Carandini, Matteo.** "What simple and complex cells compute." *The Journal of physiology* 577, no. Pt 2 (2006): 463.

### Normalisation:
- **Carandini, Matteo, and David J. Heeger.** "Normalization as a canonical neural computation." *Nature Reviews Neuroscience* 13, no. 1 (2012): 51-62.

### McCulloch-Pitts Model:
- **McCulloch, Warren S., and Walter Pitts.** "A logical calculus of the ideas immanent in nervous activity." *The bulletin of mathematical biophysics* 5 (1943): 115-133.

### The First Paper on Convolutional Neural Networks Trained with Backpropagation:
- **LeCun, Yann, Bernhard Boser, John Denker, Donnie Henderson, Richard Howard, Wayne Hubbard, and Lawrence Jackel.** "Handwritten digit recognition with a back-propagation network." *Advances in neural information processing systems* 2 (1989).
