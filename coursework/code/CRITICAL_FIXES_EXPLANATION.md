# Critical Fixes Applied to ReadySetGo DQN Implementation

## Summary

The original implementation had **4 critical issues** that fundamentally broke the RNN-based DQN approach. All have been fixed in `question_2d_readysetgo_dqn_fixed.py`.

---

## CRITICAL FIX 1: Use Observation Sequences (Not Single Timesteps)

### Problem
**Original code (lines 157-158):**
```python
state = torch.from_numpy(obs).float().unsqueeze(0)
state = state.unsqueeze(1)  # [1,1,obs] - SINGLE TIMESTEP!
```

**Why this was broken:**
- RNNs need temporal sequences to work properly
- Feeding a single timestep is like using an RNN as a feedforward network
- ReadySetGo requires measuring time intervals - impossible with single observations
- The temporal integration capability of Leaky RNNs was completely wasted

### Fix
**New code:**
```python
# Maintain observation history with sliding window
obs_history = deque(maxlen=max_seq_len)  # max_seq_len = 50
obs_history.append(obs)

# Create sequence tensor [seq_len, 1, obs_dim]
obs_sequence = torch.from_numpy(np.array(list(obs_history))).float().unsqueeze(1)
```

**Impact:**
- RNNs now see the temporal context needed for timing tasks
- Networks can integrate information over multiple timesteps
- Enables actual temporal processing, not just reactive responses

---

## CRITICAL FIX 2: Persist Hidden States Across Episode

### Problem
**Original code (line 58):**
```python
q_values, _ = policy_net(state_batch)  # Hidden state discarded with underscore!
```

**Why this was broken:**
- RNN hidden states encode temporal information
- Discarding them resets the network's "memory" every step
- Leaky RNNs build up timing representations in their hidden state
- Without persistence, the network can't accumulate temporal evidence

### Fix
**New code:**
```python
# Initialize hidden state at episode start
hidden = None

# Persist across timesteps
for step in episode:
    action, hidden = select_action(policy_net, obs_sequence, hidden, ...)
    # hidden state carries forward to next step
```

**Impact:**
- Networks maintain temporal context throughout episodes
- Leaky integration actually works now
- Bio-realistic models can use their temporal dynamics properly

---

## CRITICAL FIX 3: Fix Reward Shaping

### Problem
**Original code (lines 162-183):**
```python
# Problem 1: Fixation penalty every timestep
shaped_reward = -0.005  # Accumulates to -1.0 over 200 steps!

# Problem 2: Massive timing bonus
timing_bonus = max(0.0, 10.0 - 0.2 * timing_error)  # Up to +10.0

# Problem 3: Ignoring environment's carefully designed rewards
shaped_reward = timing_bonus  # Environment reward completely replaced
```

**Why this was broken:**
- Penalizing fixation punishes the CORRECT behaviour (waiting)
- Reward scale (10.0) is 10-50x typical RL rewards → training instability
- NeuroGym already provides proportional timing rewards via `prod_margin`
- Your custom shaping fights against the environment's design

### Fix
**New code:**
```python
if action == go_idx:
    if has_pressed_go:
        shaped_reward = -0.5  # Prevent spam
    else:
        has_pressed_go = True
        # TRUST the environment's reward
        shaped_reward = reward
        # Small curriculum bonus ONLY for early training
        if ep < 500:
            timing_error = abs(steps - target_go)
            if timing_error <= 20:
                shaped_reward += 0.5 * (1.0 - timing_error / 20.0)
else:
    # NO penalty for fixating - it's the correct action!
    shaped_reward = reward
```

**Impact:**
- Agent learns to wait patiently (correct strategy)
- Reward scale is stable and reasonable
- Curriculum bonus helps exploration early, then fades
- Works WITH the environment, not against it

---

## CRITICAL FIX 4: Update Target Network by Steps (Not Episodes)

### Problem
**Original code (lines 106, 201):**
```python
target_update = 100  # Every 100 EPISODES

if ep % target_update == 0:
    update_target_network()
```

**Why this was broken:**
- With 200 steps/episode, this is ~20,000 optimization steps between updates
- Standard DQN updates every 1,000-10,000 **steps**, not episodes
- Target network becomes extremely stale
- Q-values diverge, training becomes unstable

### Fix
**New code:**
```python
target_update_steps = 1000  # Every 1000 STEPS

global_steps = 0
for each optimization step:
    global_steps += 1
    if global_steps % target_update_steps == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

**Impact:**
- Target network stays reasonably synchronized
- Training stability dramatically improved
- Follows standard DQN practice

---

## Additional Improvements in Fixed Version

### 5. Reduced Episode Length
```python
max_steps_per_episode = 150  # Down from 200
```
- Max interval: 2000ms × gain 1.5 = 3000ms = 150 steps @ 20ms
- Reduces wasted computation

### 6. Faster Epsilon Decay
```python
epsilon_decay = 1000  # Down from 1500
```
- Reaches epsilon_end by episode 1000 (33% through training)
- More exploitation once basic exploration is done

### 7. Smaller Batch Size
```python
batch_size = 32  # Down from 64
```
- More frequent updates with sequence-based training
- Better for small replay buffers

### 8. Added L1/L2 Regularization for Bio-Realistic Model
```python
if model_type == 'bio_realistic':
    l1_loss = beta_L1 * sum(abs(p).sum() for p in policy_net.parameters())
    l2_loss = beta_L2 * (rnn_activity ** 2).mean()
    loss = loss + l1_loss + l2_loss
```
- Maintains sparsity and low firing rates during RL training
- Preserves bio-realistic constraints from supervised training

---

## How to Use

### Run the fixed version:
```bash
python code/question_2d_readysetgo_dqn_fixed.py
```

### Compare outputs:
- **Old:** `images/question_2d_readysetgo_dqn_rewards.png`
- **New:** `images/question_2d_readysetgo_dqn_fixed_rewards.png`

### Expected improvements:
1. **Better learning curves** - smoother, more stable
2. **Higher final rewards** - agents actually learn timing
3. **Bio-realistic model works** - previously broken, now functional
4. **Leaky models outperform vanilla** - temporal dynamics actually help

---

## Key Takeaway

The original implementation had the **right architecture** (RNNs) but the **wrong interface** (single timesteps, no hidden state persistence). It's like buying a sports car but only driving in first gear - the capability was there, but it wasn't being used properly.

The fixed version lets the RNNs do what they're designed for: **process temporal sequences with persistent state**.
