# MultiSensoryIntegration Task Structure

## Task Design

### Trial Timeline
```
Time Period:        Fixation         Stimulus          Decision
Duration:           0-300ms         300-1050ms        1050-1150ms
Required Action:    Fixate (0)      Fixate (0)        Left (1) or Right (2)
                    └────────────────────┘              └──────┘
                    Must fixate here                    Must choose here
```

### Action Space
- **Action 0: Fixate** - Required during fixation and stimulus periods
- **Action 1: Left** - Choose left target (decision period only)
- **Action 2: Right** - Choose right target (decision period only)

### Critical Task Constraints

1. **During Fixation + Stimulus (0-1050ms)**:
   - Model MUST choose action 0 (fixate)
   - Choosing left/right too early = abort penalty (-0.1)
   - This is ~91% of trial duration

2. **During Decision (1050-1150ms)**:
   - Model MUST choose action 1 (left) or 2 (right)
   - Continuing to fixate = no reward (trial ends)
   - Only ~9% of trial duration

3. **Trial Ending**:
   - Trial ends when model chooses left or right
   - **Trial should NEVER end on fixation action**
   - If it does, model failed to make a decision

## Expected Action Distribution

### Healthy Model
```
Action Distribution (across all timesteps):
- Fixate: ~73-80% (most of trial is fixation + stimulus)
- Left:   ~10-15% (decision period when left is correct)
- Right:  ~10-15% (decision period when right is correct)
```

### Degenerate Models

#### Problem 1: Always Fixate (>90%)
```
Actions: fix=0.95 L=0.03 R=0.02
```
**Problem**: Model never learns to make decisions
**Symptom**: Low reward (~0.0), trials timeout
**Cause**: Model learns fixation is "safe" but never explores choices

#### Problem 2: Always Choose One Side (from your confusion matrix)
```
Actions: fix=0.00 L=1.00 R=0.00  OR  fix=0.00 L=0.00 R=1.00
```
**Problem**: Model makes premature choices every timestep
**Symptom**: Confusion matrix ((200,0),(200,0))
**Cause**: Without temporal context, model can't learn when to fixate vs when to choose

#### Problem 3: Never Choose (Your "should never end in fixation")
```
Final action distribution shows fixation as most common
```
**Problem**: Model completes trial without making left/right choice
**Symptom**: Trials end without decision
**Cause**: Model doesn't understand it must choose during decision period

## Reward Structure

### From Environment
```python
rewards = {
    "abort": -0.1,    # Breaking fixation too early
    "correct": +1.0,  # Correct left/right choice during decision period
    "wrong": 0.0      # Wrong left/right choice during decision period
}
```

### Added Penalty (in our fix)
```python
if done and action == 0:
    reward -= 0.5  # Penalty for ending trial on fixation
```

This penalty discourages the model from:
- Never making a decision
- Ending trials by fixating through decision period

## Why Temporal Context is Critical

### Without Temporal Context (BROKEN)
```python
state = current_observation_only  # [1, 1, obs]

# Model sees:
Timestep 1: obs[0] → doesn't know it's fixation period → guesses
Timestep 2: obs[1] → doesn't know it's stimulus period → guesses  
Timestep 3: obs[2] → doesn't know it's decision period → guesses
```

**Result**: Model can't learn "fixate until 1050ms, then choose based on evidence"

### With Temporal Context (FIXED)
```python
state = full_observation_history  # [T, 1, obs]

# Model sees:
Timestep 1: [obs[0]] → learns this is early (fixate)
Timestep 2: [obs[0], obs[1]] → learns still accumulating evidence (fixate)
Timestep 10: [obs[0]...obs[9]] → learns decision time approaching (prepare choice)
Timestep 11: [obs[0]...obs[10]] → makes left/right choice based on accumulated evidence
```

**Result**: Model can learn temporal structure and evidence integration

## Training Progression (Expected)

### Early Training (Ep 0-500)
```
Actions: fix=0.95 L=0.03 R=0.02
```
- Exploring randomly
- Learning that fixation is required early in trial
- Occasional random left/right choices

### Mid Training (Ep 500-1500)
```
Actions: fix=0.85 L=0.08 R=0.07
```
- Learning when to transition from fixate to choose
- Still exploring different timings
- Reward increasing

### Late Training (Ep 1500-3000)
```
Actions: fix=0.75 L=0.13 R=0.12
```
- Learned proper timing: fixate → choose
- Learned which choice based on sensory evidence
- High reward (>0.8)

## Common Issues

### Issue 1: Confusion Matrix ((200,0),(200,0))
**Symptom**: All predictions are same action
**Cause**: No temporal context - model can't integrate evidence
**Fix**: Add observation history tracking ✅

### Issue 2: "Should never end in fixation"
**Symptom**: Trials end without left/right choice
**Cause**: Model doesn't learn to transition to decision
**Fix**: Add penalty for ending on fixation ✅ (just added)

### Issue 3: Premature Choices
**Symptom**: Model chooses left/right during stimulus period
**Cause**: Model hasn't learned temporal structure
**Fix**: Temporal context + sufficient training

### Issue 4: Never Chooses
**Symptom**: Model only fixates, never chooses
**Cause**: Fixation is "safe" (no abort penalty), choosing is risky
**Fix**: 
- Ensure exploration (ε-greedy) ✅
- Penalty for ending on fixation ✅
- Sufficient training time

## Verification Checklist

After training, check:

- [ ] Action distribution: ~75% fix, ~12% left, ~12% right
- [ ] Confusion matrix: Diagonal dominance (not ((200,0),(200,0)))
- [ ] Accuracy: >70% (ideally >80%)
- [ ] Balanced accuracy: >70% (both left and right performed well)
- [ ] Trials end with left/right choice (not fixation)
- [ ] Reward: >0.8

If any of these fail, the model has a degenerate policy!

## Summary

The MultiSensoryIntegration task requires:
1. **Temporal integration**: Accumulate evidence over 750ms
2. **Temporal structure learning**: Fixate early, choose later
3. **Evidence-based decisions**: Choose left/right based on integrated sensory signals

**Without temporal context**, the model cannot learn any of these and develops degenerate policies (always one action, never choosing, etc.).

**With temporal context + fixation penalty**, the model can learn the full task structure.
