# How We Learn: Changing Our Brain

**Lecturer:** Seán Froudist-Walsh
**Title:** Lecturer in Computational Neuroscience
**Institution:** University of Bristol
**Date:** 12/22/23

---

## Opening Statement

**When you overcome a challenge through learning, you have done more than gain knowledge.**

**Your brain at the beginning is different from your brain at the end.**

**You have physically reshaped it into a brain that is better able to handle that challenge in the future.**

---

## Intended Learning Outcomes

By the end of this video you will be able to:
- Describe how the brain encodes, stores and retrieves information at different timescales
- Understand the brain processes that facilitate new learning
- Describe mathematically a classic computational model of memory storage and retrieval
- Explain the purpose and basic neuroanatomy of memory consolidation

---

## What is Learning? (Neuroscience)

### Definition:

**Learning is the change to neural circuits in response to experiences, which may enable a change in future behaviour.**

### Key Mechanism:

**This often involves changes to the strength of synapses (the connections between neurons).**

### Core Concepts:
1. **Learning = Physical change in the brain**
   - Not just abstract knowledge
   - Actual modification of neural connections

2. **Experience-dependent plasticity**
   - Brain changes in response to what we experience
   - Enables behavioral change

3. **Synaptic modification**
   - Primary mechanism is changing connection strength
   - Synapses can be strengthened or weakened

---

## What is an Experience, to the Brain?

### Neural Representation of Experience:

**Experiences are coded as patterns of neural activity**

**The pattern of neural activity is just the set of neurons that are active at (nearly) the same time**

### Visual Example (Yamins et al., PNAS, 2014):

**Video:** James DiCarlo (MIT)

**Diagram showing:**
- Input: Patterns of photon energy applied here (e.g., Easter Island statues image)
- **Eye** → **RGC** (Retinal Ganglion Cells) → **LGN** → **V1** → **V2** → **V4** → **IT**
- Output: Neural states and Mental perceptual states:
  - "giraffe"
  - "kiwi"
  - "building"
  - "strawberry"

### Key Principle:
- Each experience creates a unique pattern of neural firing
- This pattern distributed across multiple brain areas
- The combination of active neurons = the brain's representation of that experience

---

## Sensory (Iconic) Memory – Shallow in the Brain

**Research:**
- Teeuwen et al., Current Biology, 2021
- Klatzmann*, Froudist-Walsh* et al., bioRxiv, 2022

### Characteristics of Sensory Memory Neural Activity:

1. **Brief**
   - Lasts only 100s of milliseconds
   - Rapidly decays

2. **Restricted to sensory parts of the cortex**
   - Does not spread to higher areas
   - Stays in early visual cortex (V1)

3. **Does not propagate deeply into the brain**
   - Shallow processing
   - Limited hierarchical spread

### Visual Example:

**Three images showing:**
1. **Picture** (zebra - clear, vivid)
2. **Decaying iconic memory** (zebra - fading, ghostly)
3. **Brain activation map** (pre-stimulus, time = 0.09 s)
   - V1 area highlighted
   - Color scale: 0 to 60 firing rate (Hz)
   - Shows activity restricted to early visual cortex

**Conclusion:** Sensory memory is a fleeting, shallow representation that doesn't engage deeper brain circuits.

---

## Working Memory Activity – Propagates Deep into the Brain's Hierarchy

**Research:**
- Leavitt et al., TiCS, 2017
- Froudist-Walsh et al., Neuron, 2021

### Characteristics of Working Memory Neural Activity:

**Neural activity corresponding to working memory:**

1. **Propagates deeply into the brain**
   - Spreads through cortical hierarchy
   - Engages high-level areas

2. **The pattern of neural activity largely persists after the sensory memory has died**
   - Sustained activity over seconds
   - Maintains information after stimulus offset

3. **Activity in early sensory areas usually dies off**
   - Corresponding to a loss of fine sensory detail
   - High-level abstract representation remains

### Experimental Evidence:

**Two brain visualizations showing:**
- **Experimental data** (two 3D brain renderings)
- **Model simulation** (p = 0.0001)

**Brain map showing cortical hierarchy:**
- **Persistent activity** (cyan/turquoise areas)
  - Higher in cortical hierarchy
  - Dorsolateral prefrontal cortex
  - Posterior parietal cortex

- **No persistent activity** (orange/brown areas)
  - Lower in hierarchy
  - Early sensory areas

### Related Topics:

**"See video on deep architectures"**
- Why does information travel up the hierarchy?

**"See video on recurrent architectures"**
- Why can some areas maintain their patterns of neural activity better than others?

---

## What Are the Consequences of Deeper Processing for Memory?

### Two Key Consequences:

1. **Greater depth of processing = information travels further up the brain's hierarchy**
   - From sensory cortex to association cortex
   - More abstract representation
   - More areas involved

2. **Information is sustained for longer, neurons remain active at the same time for longer**
   - Longer co-activation
   - More opportunity for learning (Hebbian plasticity)
   - Stronger memory encoding

### Implication:
**The deeper the processing, the better the encoding into long-term memory**

---

## Hebbian Plasticity: The Foundation of Learning in the Brain

### The Fundamental Principle:

**"Neurons that fire together, wire together."**

**Named after Donald Hebb, who proposed the idea in 1949.**

### Mechanism:

1. **Strengthening:**
   - When one neuron frequently activates another, their synaptic connection strengthens
   - Repeated co-activation → stronger synapse

2. **Weakening:**
   - Conversely, if two neurons rarely activate together, their connection weakens
   - Lack of correlation → weaker synapse

3. **Associative learning:**
   - Forms the basis for associating things in a single memory
   - Links together elements of an experience

### Visual Representation:

**Image: The Reward Foundation**

**Two panels:**
1. **"Nerve Cells that Fire Together..."**
   - Pre-synaptic neuron (purple dendrites)
   - Post-synaptic neuron (purple dendrites)
   - Weak connection (thin yellow synapse)

2. **"...Wire Together"**
   - Same neurons
   - Strengthened connection (thicker, more robust yellow synapse)
   - Enhanced communication

### Mathematical Formulation:

**Basic Hebbian Rule:**

```
ẇ = H(pre, post)
```

**Interpretation:**
- ẇ = rate of change in synaptic weight
- From the presynaptic neuron to the postsynaptic neuron
- H = a function of the presynaptic firing and the postsynaptic activity

**The rate of change in synaptic weight from the presynaptic neuron to the postsynaptic neuron is a function of the presynaptic firing and the postsynaptic activity**

---

## Selective Encoding: From Working to Long-Term Memory

### Key Question:

**Not all working memories get transferred to long-term storage**

### Interactive Question:

"Do you still remember the one-time code from the video: Human memory: cognitive science foundations?"

(Most people won't - illustrating selective encoding)

### Central Question:

**What enables some working memories to be marked for encoding into long-term memory?**

### Answer Preview:
The next section on neuromodulators provides the answer...

---

## The Third Player in Plasticity: Neuromodulators

### Definition:

**A neuromodulator is a chemical messenger released in the brain that changes (modulates) how two other neurons communicate at the synapse.**

### Key Neuromodulators:

**Neuromodulators like dopamine and acetylcholine signal:**
- **Reward**
- **Importance**
- **Salience**
- **Attention**

### 3-Factor Learning Rule:

**"Adding dopamine or serotonin to Hebbian plasticity is like putting your name on the list of the long-term memory club"**

### How Neuromodulators Work:

**Mechanism:**
- Like all brain chemicals, neuromodulators act by binding to receptors (like a key opening a lock)
- Different receptor types for different neuromodulators
- Receptors trigger intracellular signaling cascades

### Distribution in the Brain:

**Research:** Froudist-Walsh et al., Nature Neuroscience, 2023

**Key Finding:**
**"The receptors for neuromodulators are much more prevalent deep in the cortical hierarchy than in early sensory areas."**

**Brain visualization showing:**
- **Less modulation** (blue, -7.5): Early sensory areas
- **More modulation** (red, 2.5): Higher cognitive areas

**Implication:** Higher cognitive areas are more "tunable" and plastic

### Mathematical Formulation: 3-Factor Rule

**Enhanced Learning Rule:**

```
ẇ = y·H(pre, post)
```

**Where:**
- ẇ = rate of change in synaptic weight
- y = neuromodulator concentration
- H = function of presynaptic and postsynaptic activity
- pre = presynaptic firing
- post = postsynaptic activity

**Interpretation:**
**"The rate of change in synaptic weight from the presynaptic neuron to the postsynaptic neuron depends on a neuromodulator and a function of the presynaptic firing and the postsynaptic activity"**

### Visual Diagram:

**Showing:**
- **pre** (green circle) → **w** (synapse) → **post** (green circle)
- **y** (yellow star/neuromodulator) modulating the synapse

**Key Insight:**
- Without neuromodulator (y = 0): No learning, even if pre and post are correlated
- With neuromodulator (y > 0): Learning occurs based on correlation
- This gates which experiences get encoded into long-term memory

---

## Quiz – Blackboard

**Question:**

**"Why is certain information encoded and other information forgotten? Write down initial thoughts integrating a neuroscience and a cognitive science point of view."**

### Answer Framework:

#### Neuroscience Perspective:
1. **Depth of processing**
   - Shallow processing (sensory memory) doesn't engage learning mechanisms
   - Deep processing (working memory) allows for synaptic changes

2. **Neuromodulator release**
   - Important/rewarding information triggers dopamine release
   - Neuromodulators gate Hebbian plasticity
   - No neuromodulator = no long-term encoding

3. **Duration of neural activity**
   - Longer co-activation = stronger synaptic changes
   - Working memory maintenance enables encoding

#### Cognitive Science Perspective:
1. **Attention and relevance**
   - We pay attention to important information
   - Attention correlates with neuromodulator release

2. **Emotional significance**
   - Emotional events are better remembered
   - Emotion triggers neuromodulator systems

3. **Repetition and rehearsal**
   - Repeated activation strengthens connections
   - Maintenance rehearsal in working memory

4. **Meaning and elaboration**
   - Meaningful information processed more deeply
   - Elaborative encoding creates richer representations

---

## Long-Term Memory, from Retrieval to Encoding

(Section title - introduces the next major topic)

---

## Your Experience

### Interactive Exercise:

**Instructions:**
- "Take one minute to recall, in as much detail as you can, a recent experience."
- "Write it down (privately)"

### Purpose:
This exercise demonstrates:
1. Memory retrieval in action
2. Pattern completion (you won't remember everything perfectly)
3. The subjective experience of reconstruction
4. Gap-filling and inference during recall

---

## (Blank Slide for Reflection)

(Slide 15 appears to be blank - likely for student reflection/writing time)

---

## The Nature of Memory Retrieval

### Core Principle:

**"Retrieval is the brain's way of recreating the neural activity pattern from the original event."**

**"But it can't do this perfectly"**

### Visual Representation:

**Diagram (Yamins et al., PNAS, 2014; video: James DiCarlo, MIT):**

**Showing:**
- Original image (Easter Island statues) at top
- **Red circular arrow** indicating retrieval/reconstruction
- Visual processing hierarchy: eye → RGC → LGN → V1 → V2 → V4 → IT
- **Hippocampus / prefrontal cortex** (highlighted in red) initiating retrieval
- Neural states leading to mental perceptual states
- Patterns of photon energy applied here (at bottom)
- **Another red arrow** showing the loop from memory back to perception

### Key Insights:

1. **Reconstruction, not playback**
   - Memory is not like playing a video
   - It's rebuilding the pattern from fragments

2. **Imperfect process**
   - Some details lost
   - Some details inferred/filled in
   - Potential for distortion

3. **Top-down process**
   - Initiated by hippocampus/prefrontal cortex
   - Travels back down the hierarchy
   - Attempts to reactivate the original pattern

4. **Pattern completion required**
   - From partial cues, reconstruct the whole
   - Next section explains this mechanism

---

## Pattern Completion in Memory Retrieval

### Definition:

**Pattern completion is the brain's ability to reconstruct a pattern of neural activity corresponding to a memory from a partially overlapping pattern of neural activity.**

### Key Brain Structure: Hippocampus

**"The hippocampus (in particular a part called CA3) is the area most famously involved in this function"**

#### CA3 Properties:
- Dense recurrent connections
- Auto-associative network
- Specialized for pattern completion

### Beyond the Hippocampus:

**"However, some form of pattern completion must also take place across the cortex in order to recreate the original pattern of activity during retrieval"**

### Visual Representation:

**Brain diagram showing:**
- **Cortex** (labeled at top, textured surface)
- **Hippocampus** (labeled, shown in cyan/blue, deep in temporal lobe)
- Sound waves icon (representing partial cue)
- Arrows suggesting interaction between hippocampus and cortex

### Process:

1. **Partial cue** (e.g., a smell, a sound, a thought)
2. **Hippocampal CA3** recognizes pattern from fragment
3. **Reconstruction** of full pattern
4. **Cortical reinstatement** of original activity pattern
5. **Conscious experience** of the memory

### Example:
- Hear a few notes of a song → remember the whole song
- See part of a face → recognize the person
- Small reminder → recall entire event

---

## Quiz – Blackboard

**Question:**

**"Explain the difference between sensory memory, working memory, and long-term memory in terms of neural activity patterns."**

### Answer:

#### 1. Sensory Memory:
**Neural Activity Pattern:**
- **Location:** Restricted to early sensory cortex (e.g., V1)
- **Duration:** Very brief (100s of milliseconds)
- **Depth:** Shallow - doesn't propagate up hierarchy
- **Quality:** High fidelity, detailed
- **Mechanism:** Lingering sensory activation
- **Example:** Brief afterimage of a flash

#### 2. Working Memory:
**Neural Activity Pattern:**
- **Location:** Distributed across cortical hierarchy, especially higher areas
- **Duration:** Sustained for seconds (up to ~30 seconds)
- **Depth:** Deep - engages prefrontal and parietal cortex
- **Quality:** Abstract, less detailed than sensory
- **Mechanism:** Persistent neural firing, recurrent activity
- **Example:** Holding a phone number in mind

#### 3. Long-Term Memory:
**Neural Activity Pattern:**
- **Encoding:** Changes in synaptic strengths (not ongoing activity)
- **Storage:** Distributed synaptic weights across cortex and hippocampus
- **Duration:** Potentially permanent (structural changes)
- **Retrieval:** Reconstruction of activity pattern via pattern completion
- **Quality:** Variable - gist often preserved, details may be lost
- **Mechanism:** Hebbian plasticity, neuromodulator-gated, consolidated during sleep
- **Example:** Remembering your childhood home

---

## A Classic Model of Pattern Completion for Memory Retrieval (Hopfield)

### The Hopfield Network

**Key Features:**

1. **Fully connected**
   - Every neuron connects to every other neuron
   - Rich interconnectivity

2. **Symmetric weights (wᵢⱼ = wⱼᵢ)**
   - Connection from i to j equals connection from j to i
   - Ensures energy function exists

3. **Binary outputs: (+1) or (-1)**
   - Neuron is either active (+1) or inactive (-1)
   - Simplified from biological reality

4. **Energy-based system**
   - The network seeks a state of minimum energy (attractor state)
   - Stored patterns = energy minima

5. **Can store multiple patterns with Hebb's law**
   - Multiple memories in same network
   - Retrieved based on partial cue

### Network Diagram:

**Visualization:**
- 5 neurons (circles) arranged in pentagon
- Fully connected (every neuron connected to all others)
- Bidirectional arrows showing symmetric connections

### Learning Rule: Hebbian Storage

**Formula:**

```
wᵢⱼ = (1/P) Σₐ₌₁ᴾ xᵢᵃ xⱼᵃ
```

**Interpretation:**

"If for a pattern a, both neurons i and j are active (or both inactive), then the synaptic weight should be increased, and decrease otherwise. If the network has seen the entire collection of patterns P, the synaptic weight between i and j should reflect the fraction of patterns for which the two neurons are co-active."

**Where:**
- wᵢⱼ = synaptic weight from neuron j to neuron i
- P = total number of patterns stored
- xᵢᵃ = activity of neuron i in pattern a (+1 or -1)
- xⱼᵃ = activity of neuron j in pattern a (+1 or -1)

### Dynamics: Pattern Completion/Retrieval

**Update Rule (at each timestep):**

```
xᵢ(t + 1) = {
  +1  if  θ < Σⱼ wᵢⱼxⱼ
  -1  otherwise
}
```

**Interpretation:**

"At the next timepoint neuron i is active (+1) if the weighted sum of all the other neurons' activities is greater than a threshold. Otherwise it is inactive (-1)."

**Where:**
- xᵢ(t+1) = activity of neuron i at next timestep
- θ = threshold parameter
- wᵢⱼ = synaptic weight from j to i
- xⱼ = current activity of neuron j

### Energy Function

**Formula:**

```
E = -(1/2) Σᵢⱼ wᵢⱼxᵢxⱼ + Σᵢ θᵢxᵢ
```

**Key Principle:**

"Activity will continue to evolve until it reaches a local minimum of this energy function E, which is an attractor state."

**Diagram showing:**
- **Energy landscape** (curved surface)
- **Basin of attraction** (valley)
- **Attractor state** (energy minimum)
- Starting state rolls down to attractor

### Storage Capacity:

**Rule of thumb:** Can store approximately **0.15N patterns** reliably
- Where N = number of neurons
- Beyond this, spurious attractors and interference occur

### Biological Relevance:

**Similarities to CA3:**
- Dense recurrent connections (like Hopfield network)
- Auto-associative properties
- Pattern completion function

**Differences:**
- Biological neurons not binary
- Weights not perfectly symmetric
- Inhibitory neurons (not in basic Hopfield)
- Synaptic plasticity is dynamic, not static

---

## Quiz – Blackboard

**Question:**

**"Briefly explain the main principle behind the Hopfield model of memory storage and retrieval."**

### Answer:

#### Main Principle:

**Energy Minimization and Attractor Dynamics**

#### Storage:
- **Memories stored as patterns of activity** (attractors)
- **Hebbian learning** sets synaptic weights
- Weights encode correlations between neurons across stored patterns
- Each pattern becomes a stable attractor state (energy minimum)

#### Retrieval:
- **Present partial/noisy cue** (partial pattern)
- **Network dynamics** evolve according to update rule
- System **descends energy landscape**
- **Settles into nearest attractor** (stored pattern)
- **Pattern completion** achieved

#### Key Insight:
**The same Hebbian mechanism that stores memories (setting weights) also enables retrieval (dynamics settle to stored patterns)**

**Content-addressable memory:** Any part of the pattern can retrieve the whole

---

## Consolidating Episodic Memories During Sleep: Strengthening and Generalising

**Research:**
- Wilson & McNaughton, Science, 1994
- Winocur and Moscovitch, JINS, 2011
- Klinzing et al., Nature Neuroscience, 2019

### The Process:

1. **Hippocampal Replay**
   - "During sleep, neurons in the hippocampus replay the activity that played out during the day, at ~10x speed"
   - Compressed reactivation of daily experiences
   - Particularly during slow-wave sleep

2. **Hippocampal-Cortical Dialogue**
   - "Hippocampal activity spreads to the cortex"
   - Coordinated reactivation
   - Strengthens cortical representations

3. **Transfer of Representation**
   - "The representation of the memory may gradually move from the hippocampus to the cortex"
   - Systems consolidation
   - Hippocampus → Cortex transfer

4. **Transformation**
   - "This could coincide with older memories seeming less 'episodic' and more 'semantic' (the gist)"
   - Loss of contextual details
   - Retention of general meaning/facts

### Visual Representation:

**Two brain diagrams showing:**

**Left brain:**
- Hippocampus (highlighted in blue)
- Multiple cortical areas
- Red arrows showing:
  - **Recently encoded neuronal part of a representation** (from hippocampus to cortex)
  - **Recently encoded neuronal part of a representation** (different pattern)

**Right brain:**
- Similar structure
- Blue hippocampus
- Red arrows showing:
  - **Associated pre-existing representation**
  - **Unrelated pre-existing representation**

**Legend:**
- **Cell assemblies contributing to:**
  - Recently encoded neuronal part of a representation (red circles/connections)
  - Recently encoded neuronal part of a representation (different color)

### Functions of Sleep Consolidation:

1. **Strengthening**
   - Replay strengthens synaptic connections
   - More robust memory traces

2. **Generalizing**
   - Extraction of common features
   - Schema formation
   - Integration with existing knowledge

3. **Interference reduction**
   - Separation of similar memories
   - Reduced competition

---

## Consolidating Skills During Sleep: Automatising and Freeing Up Cortex

**Research:** Dehaene-Lambertz et al., PLoS biology, 2018

### The Process:

1. **Reduced Cortical Activation**
   - "With expertise, skills (a.k.a. procedural memories such as reading) activate less of the cortex"
   - More efficient processing
   - Neural efficiency

2. **Automatisation**
   - "Can be performed automatically"
   - Less conscious attention required
   - Fluent execution

3. **Transfer to Subcortical Structures**
   - "Automatic expertise relies more on areas deep in the brain that are involved in habits (basal ganglia)"
   - Cortex → Basal ganglia transfer
   - Habitual control

4. **Freeing Resources**
   - "This frees up major cortical networks to consciously tackle challenging problems"
   - Cognitive resources freed
   - Capacity for new learning

### Visual Evidence:

**Series of brain scans showing reading development:**

**Rows showing progression:**
- At birth (day 1)
- 1 month old
- 3 months old
- 6 months old
- 12 months old
- 24 months old
- 37 months old
- One year later

**Columns:**
- Left hemisphere view (network visualization)
- Right hemisphere view (network visualization)
- Brain scan highlighting **Words > others** activation

**Observation:**
- Complex, widespread activation early in learning
- More focused, efficient activation with expertise
- Progressive refinement of neural circuits

### Types of Consolidation:

| Memory Type | Consolidation Process | Outcome |
|-------------|----------------------|---------|
| **Episodic** | Hippocampus → Cortex | Strengthening, generalizing (semantic) |
| **Procedural/Skills** | Cortex → Basal Ganglia | Automatisation, efficiency |

---

## Quiz – Blackboard

**Question:**

**"Compare and contrast episodic, semantic and procedural memories in terms of their consolidation during sleep."**

### Answer:

#### Episodic Memory Consolidation:

**Process:**
- Hippocampal replay of specific events
- Transfer from hippocampus to cortex
- Coordinated reactivation during sleep

**Outcome:**
- Strengthened memory traces
- Potential transformation to semantic memory (lose details, keep gist)
- Integration with existing knowledge

**Brain regions:**
- Hippocampus → Neocortex

**Sleep stage:**
- Primarily slow-wave sleep (SWS)

#### Semantic Memory Consolidation:

**Process:**
- May result from episodic consolidation
- Extraction of common features across episodes
- Schema formation

**Outcome:**
- General knowledge, facts
- Context-free information
- Organized conceptual networks

**Brain regions:**
- Distributed across neocortex
- Less hippocampal dependence

#### Procedural Memory Consolidation:

**Process:**
- Practice-dependent plasticity
- Consolidation during sleep
- Transfer to subcortical structures

**Outcome:**
- Automatisation of skills
- More efficient cortical processing (less activation)
- Subcortical control (basal ganglia, cerebellum)
- Frees up cortical resources

**Brain regions:**
- Cortex → Basal ganglia, Cerebellum

**Sleep stage:**
- REM sleep and Stage 2 NREM important for procedural consolidation

#### Key Contrasts:

| Feature | Episodic | Semantic | Procedural |
|---------|----------|----------|------------|
| **What** | Specific events | General facts | Skills/habits |
| **Brain shift** | Hippocampus → Cortex | Across cortex | Cortex → Basal ganglia |
| **Transformation** | Event → Gist | Episodes → Concepts | Controlled → Automatic |
| **Consciousness** | Consciously recalled | Consciously recalled | Performed without awareness |

---

## Recap

### Key Points:

1. **Experiences are represented as patterns of neural activity**
   - Distributed across brain areas
   - Each experience = unique pattern

2. **Sensory memory neural activity patterns are restricted to the outer sensory areas of the cortex and fade quickly (100s of ms)**
   - Shallow processing
   - Brief duration
   - High fidelity

3. **Working memory activity patterns spread deep into the cortical hierarchy and are maintained for 10s of seconds**
   - Deep processing
   - Persistent activity
   - Enables encoding

4. **Working memories may fade or be encoded into long-term memory through Hebbian plasticity**
   - "Neurons that fire together, wire together"
   - Synaptic strengthening

5. **Encoding into long-term memory is more likely if a neuromodulator is present**
   - 3-factor learning rule
   - Dopamine/acetylcholine gate plasticity
   - Marks important experiences

6. **Memory retrieval is the brain's attempt to reconstruct a pattern of activity that arose during the original event (Hopfield model)**
   - Pattern completion
   - Energy minimization
   - Attractor dynamics

7. **Consolidation of memory during sleep can strengthen, generalise and/or automatise a memory**
   - Replay during sleep
   - System-level reorganization
   - Transformation of representations

8. **Consolidation involves a shift in the anatomical storage sites of memory**
   - Episodic: Hippocampus → Cortex
   - Procedural: Cortex → Basal ganglia
   - More distributed, more stable

---

## Still Curious? You Can Dive in Deeper to Any of Today's Topics:

### Sensory/Iconic Memory:
- **Teeuwen, Rob RM, Catherine Wacongne, Ulf H. Schnabel, Matthew W. Self, and Pieter R. Roelfsema.** "A neuronal basis of iconic memory in macaque primary visual cortex." *Current Biology* 31, no. 24 (2021): 5401-5414.

### Working Memory:
- **Froudist-Walsh, Sean, Daniel P. Bliss, Xingyu Ding, Lucija Rapan, Meiqi Niu, Kenneth Knoblauch, Karl Zilles, Henry Kennedy, Nicola Palomero-Gallagher, and Xiao-Jing Wang.** "A dopamine gradient controls access to distributed working memory in the large-scale monkey cortex." *Neuron* 109, no. 21 (2021): 3500-3520.

### Hebbian Plasticity:
- **Hebb, Donald Olding.** *The organization of behavior: A neuropsychological theory.* Psychology press, 2005.

### Neuromodulators and Plasticity:
- **Frémaux, Nicolas, and Wulfram Gerstner.** "Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules." *Frontiers in neural circuits* 9 (2016): 85.

- **Kuśmierz, Łukasz, Takuya Isomura, and Taro Toyoizumi.** "Learning with three factors: modulating Hebbian plasticity with errors." *Current opinion in neurobiology* 46 (2017): 170-177.

- **Froudist-Walsh, Sean, Ting Xu, Meiqi Niu, Lucija Rapan, Ling Zhao, Daniel S. Margulies, Karl Zilles, Xiao-Jing Wang, and Nicola Palomero-Gallagher.** "Gradients of neurotransmitter receptor expression in the macaque cortex." *Nature Neuroscience* (2023): 1-14.

### Hopfield Network:
- **Hopfield, John J.** "Neural networks and physical systems with emergent collective computational abilities." *Proceedings of the national academy of sciences* 79, no. 8 (1982): 2554-2558.

### Replay of Neural Activity During Sleep:
- **Wilson, Matthew A., and Bruce L. McNaughton.** "Reactivation of hippocampal ensemble memories during sleep." *Science* 265, no. 5172 (1994): 676-679.

### Consolidation of Memories:

**Episodic-to-Semantic Transfer:**
- **Winocur, Gordon, and Morris Moscovitch.** "Memory transformation and systems consolidation." *Journal of the International Neuropsychological Society* 17, no. 5 (2011): 766-780.
