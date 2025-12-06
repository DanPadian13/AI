# CRITICAL DQN FIXES - Degenerate Policy Problem

## Problem Identified
Your DQN models had a **confusion matrix of ((200,0),(200,0))** - they were always choosing the same action (either always Left or always Right), completely ignoring the task structure.

## Root Causes

### 1. **NO TEMPORAL CONTEXT** ⚠️ CRITICAL
**Problem**: The code was only passing a single timestep to the RNN:
```python
state = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)  # [1, 1, obs]
```

This is just ONE observation! The RNN never got to see the temporal sequence needed for evidence integration.

**Why this breaks the task**:
- MultiSensoryIntegration requires accumulating evidence over ~750ms stimulus period
- Without history, the model can't integrate information over time
- It's like asking someone to make a decision based on 1ms of data instead of the full stimulus

### 2. **WRONG Q-VALUE INDEXING**
**Problem**: Using `q_vals[0, 0]` which is the FIRST timestep
**Should be**: `q_vals[-1, 0]` which is the LAST (current) timestep

### 3. **BATCHING ISSUES**
**Problem**: Can't batch variable-length sequences directly
**Solution**: Pad sequences to same length before batching

### 4. **INSUFFICIENT EXPLORATION**
**Problem**: 
- Epsilon decays too fast (reaches 0.1 at 70% of training)
- Not enough exploration to discover non-trivial strategies

## Fixes Implemented

### Fix 1: Track Full Observation History ✅
```python
# NEW: Track full history
obs_history = []

while not done:
    obs_history.append(obs)
    state = torch.from_numpy(np.array(obs_history)).float().unsqueeze(1)  # [T, 1, obs]
    
    # Now the RNN sees the FULL temporal context!
```

**Impact**: RNN can now use its temporal dynamics to integrate evidence over time

### Fix 2: Use Final Timestep for Decisions ✅
```python
# OLD (WRONG):
action = int(q_vals[0, 0].argmax().item())  # First timestep - meaningless!

# NEW (CORRECT):
action = int(q_vals[-1, 0].argmax().item())  # Last timestep - current Q-values!
```

**Impact**: Model makes decisions based on accumulated evidence, not initial state

### Fix 3: Proper Sequence Batching ✅
```python
# Find max length in batch
max_len = max(s.shape[0] for s in batch.state)

# Pad all sequences to same length
for s, ns in zip(batch.state, batch.next_state):
    if s.shape[0] < max_len:
        pad_len = max_len - s.shape[0]
        s_padded = torch.cat([torch.zeros(pad_len, 1, s.shape[2]), s], dim=0)
    # ...

state_batch = torch.stack(state_batch, dim=1).to(device)  # [T, batch, obs]
```

**Impact**: Can properly batch variable-length sequences for efficient training

### Fix 4: Increased Exploration ✅
```python
# OLD:
epsilon_min = 0.1
epsilon_decay_end = int(num_episodes * 0.7)  # 70% of training

# NEW:
epsilon_min = 0.15  # Higher minimum exploration
epsilon_decay_end = int(num_episodes * 0.8)  # Slower decay (80% of training)
```

**Impact**: Model explores longer, preventing premature convergence to degenerate policies

### Fix 5: Action Distribution Monitoring ✅
```python
action_counts = [0, 0, 0]  # Track fixate, left, right

# In training loop:
action_counts[action] += 1

# Print action distribution every 50 episodes:
action_dist = [c/total_actions for c in action_counts]
print(f"Actions: fix={action_dist[0]:.2f} L={action_dist[1]:.2f} R={action_dist[2]:.2f}")
```

**Impact**: Early detection of degenerate policies during training

## Why This Matters for Your Task

### MultiSensoryIntegration Task Structure
```
Time:    0-300ms     300-1050ms        1050-1150ms
Period:  Fixation    Stimulus          Decision
Action:  Fixate (0)  Fixate (0)        Left (1) or Right (2)
```

**Without temporal context**:
- Model sees one timestep at a time
- Can't accumulate evidence from stimulus period
- Defaults to always choosing same action (e.g., always Left)
- Gets ~33% reward (sometimes guesses right by chance)

**With temporal context**:
- Model sees full history from fixation → stimulus → decision
- Can integrate evidence over time
- Can learn to fixate during fixation period
- Can learn to choose based on accumulated evidence
- Should get >80% reward

## Expected Behavior After Fixes

### Training Progress
You should now see:
```
Ep   50 | Avg reward:  0.30 | ε=0.987 | Actions: fix=0.91 L=0.05 R=0.04  ← Early: mostly fixating
Ep  100 | Avg reward:  0.45 | ε=0.975 | Actions: fix=0.88 L=0.07 R=0.05  ← Learning to fixate
Ep  500 | Avg reward:  0.65 | ε=0.875 | Actions: fix=0.82 L=0.10 R=0.08  ← Better fixation
Ep 1000 | Avg reward:  0.78 | ε=0.750 | Actions: fix=0.78 L=0.12 R=0.10  ← Starting to choose
Ep 2000 | Avg reward:  0.85 | ε=0.500 | Actions: fix=0.75 L=0.13 R=0.12  ← Good performance
Ep 3000 | Avg reward:  0.90 | ε=0.150 | Actions: fix=0.73 L=0.14 R=0.13  ← Converged
```

**Action distribution should be**:
- ~70-80% fixate (most of trial is fixation + stimulus periods)
- ~10-15% left (decision period when left is correct)
- ~10-15% right (decision period when right is correct)

**NOT**: 100% of one action!

### Confusion Matrix (After Fixing)
Instead of:
```
        Pred: Fix   Left  Right
True:
Fix     [   0      0      0   ]  ← No fixation predictions
Left    [   0    100      0   ]  ← Always predict Left
Right   [   0    100      0   ]  ← Always predict Left (WRONG!)
```

You should see:
```
        Pred: Fix   Left  Right
True:
Fix     [ 190      5      5   ]  ← Mostly correct fixation
Left    [  10     85      5   ]  ← Mostly correct left
Right   [  10      5     85   ]  ← Mostly correct right
```

## How to Verify the Fix

### 1. Check Training Output
```bash
python code/question_2d_MSI_simple_dqn.py
```

Look for:
- ✅ Action distribution changes over time (not stuck at 100% one action)
- ✅ Rewards increasing (should reach >0.8 by end of training)
- ✅ Balanced actions: ~75% fix, ~12% left, ~12% right

### 2. Check Analysis
```bash
python code/question_2d_MSI_simple_dqn_analysis.py
```

Look for:
- ✅ Confusion matrices with diagonal dominance
- ✅ Accuracy >70% (ideally >80%)
- ✅ Balanced accuracy >70%

### 3. Visual Inspection
Check `question_2D_MSI_example_predictions.png`:
- ✅ P(fixate) high during fixation/stimulus periods
- ✅ P(left) or P(right) rises during decision period
- ✅ Different trials show different predictions (not all the same!)

## Technical Deep Dive

### Why RNNs Need Temporal Context

**RNN Architecture**:
```python
h_t = f(h_{t-1}, x_t)  # Hidden state depends on previous state
```

**With single timestep**:
```python
h_0 = 0  # Initial state (no memory)
h_1 = f(h_0, x_1)  # Only sees current observation
Q_values = g(h_1)  # Decision based on single timestep
```
→ **Can't integrate over time!**

**With full sequence**:
```python
h_0 = 0
h_1 = f(h_0, x_1)
h_2 = f(h_1, x_2)  # Now has context from x_1
h_3 = f(h_2, x_3)  # Now has context from x_1, x_2
...
h_T = f(h_{T-1}, x_T)  # Full accumulated evidence
Q_values = g(h_T)  # Decision based on integrated evidence
```
→ **Can integrate over time!** ✅

### Why This Was Hard to Debug

1. **Training seemed to work**: Loss decreased, rewards increased slightly
2. **Model wasn't "broken"**: It converged to a strategy (always choose left)
3. **Rewards weren't zero**: ~33% accuracy by random chance on 3-way choice
4. **Action distribution hidden**: Without monitoring, you don't see the problem

The confusion matrix was the **smoking gun** that revealed the degenerate policy!

## Comparison with Supervised Learning

**Supervised learning** (Question 2a):
- Always sees full trials: `inputs = torch.from_numpy(ob[:, np.newaxis, :])` 
- Shape: `[T, 1, obs]` where T is full trial length
- Backprop through time over full sequence
- **Never had this problem!**

**DQN (before fix)**:
- Only saw single timestep
- Shape: `[1, 1, obs]` - just current observation
- RNN couldn't use temporal dynamics
- **Degenerate policy!**

**DQN (after fix)**:
- Now sees full history
- Shape: `[T, 1, obs]` - same as supervised!
- RNN can use temporal dynamics
- **Should work properly** ✅

## Performance Expectations

### Before Fix:
- **Accuracy**: ~33% (chance level for 3-way choice)
- **Confusion matrix**: ((200, 0), (200, 0)) - all one action
- **Action distribution**: 100% one action (fixate, left, OR right)
- **Reward**: ~0.3-0.4 (mostly from abort penalties avoided)

### After Fix:
- **Accuracy**: >80% (task is easy with temporal integration)
- **Confusion matrix**: Diagonal dominance
- **Action distribution**: ~75% fix, ~12% left, ~12% right
- **Reward**: >0.85 (high success rate)

## Next Steps

1. **Re-train all models**:
   ```bash
   python code/question_2d_MSI_simple_dqn.py
   ```
   
2. **Monitor action distribution** during training - should be balanced!

3. **Re-run analysis**:
   ```bash
   python code/question_2d_MSI_simple_dqn_analysis.py
   ```

4. **Verify confusion matrices** now show proper diagonal structure

5. **Compare with supervised learning** - now on equal footing!

## Lessons Learned

### For RL with RNNs:
1. ✅ **Always provide temporal context** for sequence tasks
2. ✅ **Use final timestep Q-values** for decisions
3. ✅ **Monitor action distributions** to detect degenerate policies
4. ✅ **Pad sequences properly** for batching
5. ✅ **Maintain sufficient exploration** to escape local optima

### For Debugging:
1. ✅ **Confusion matrices are crucial** for classification tasks
2. ✅ **Action distribution monitoring** catches degenerate policies early
3. ✅ **Compare with supervised learning** as sanity check
4. ✅ **Visualize example trials** to see actual behavior

This was a **subtle but critical bug** - the model was "learning" but learning the wrong thing! The temporal context fix should resolve it completely.
