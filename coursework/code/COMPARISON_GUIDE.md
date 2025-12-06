# DQN vs Supervised Learning: Comparison Guide

## Quick Reference: Which Files to Compare

### Training Files
| Approach | File | Purpose |
|----------|------|---------|
| **Supervised** | `Question_2_multisensory_train.py` | Train 4 RNN models with supervised learning |
| **DQN (RL)** | `question_2d_MSI_simple_dqn.py` | Train 4 RNN models with DQN reinforcement learning |

### Analysis Files
| Approach | File | Purpose |
|----------|------|---------|
| **Supervised** | `Question_2_multisensory_analysis.py` | Comprehensive analysis of supervised models |
| **DQN (RL)** | `question_2d_MSI_simple_dqn_analysis.py` | Comprehensive analysis of DQN models |

### Checkpoints
| Approach | File | Contents |
|----------|------|----------|
| **Supervised** | `checkpoints/question_2_multisensory_models_and_data.pt` | Trained models + trial data |
| **DQN (RL)** | `checkpoints/question_2d_MSI_dqn.pt` | Trained DQN models |

### Output Images
| Approach | File Pattern | Examples |
|----------|--------------|----------|
| **Supervised** | `images/q2_multisensory_*.png` | `q2_multisensory_performance.png` |
| **DQN (RL)** | `images/question_2D_MSI_*.png` | `question_2D_MSI_performance.png` |

## Analysis-by-Analysis Comparison

### 1. Performance Comparison
**Supervised**: `q2_multisensory_performance.png`
**DQN**: `question_2D_MSI_performance.png`

**What to compare**:
- Overall accuracy (should both be >80%)
- Balanced accuracy (are models biased?)
- Which approach performs better?
- Do biological constraints affect them differently?

**Expected differences**:
- Supervised might be slightly better (direct supervision)
- DQN might have more variability (exploration noise)

---

### 2. Coherence Analysis ⭐
**Supervised**: `q2_multisensory_coherence_analysis.png`
**DQN**: `question_2D_MSI_coherence_analysis.png`

**What to compare**:
- Psychometric curves (accuracy vs difficulty)
- Do both approaches struggle with low coherence?
- Slope of curves (sensitivity to coherence)
- Error bars (which is more consistent?)

**Key question**: Does RL learn the same difficulty gradient as supervised?

---

### 3. Modality Weighting ⭐⭐
**Supervised**: `q2_multisensory_modality_weighting.png`
**DQN**: `question_2D_MSI_modality_weighting.png`

**What to compare**:
- Performance when one modality dominates
- Performance with balanced modalities (0.5)
- Can both approaches flexibly integrate?
- Which is better at multi-sensory integration?

**Key question**: Does RL integrate sensory information differently than supervised?

---

### 4. Confusion Matrices
**Supervised**: `q2_multisensory_confusion_matrices.png`
**DQN**: `question_2D_MSI_confusion_matrices.png`

**What to compare**:
- Error patterns (fixate vs choose errors)
- Left/right confusion rates
- Are errors systematic or random?

**Key question**: Do RL and supervised make the same types of errors?

---

### 5. Decision Confidence ⭐
**Supervised**: `q2_multisensory_decision_confidence.png`
**DQN**: `question_2D_MSI_decision_confidence.png`

**What to compare**:
- Confidence distributions (correct vs incorrect)
- Mean confidence levels
- Calibration (high confidence = high accuracy?)
- Overlap between distributions

**Key question**: Is RL well-calibrated like supervised learning?

**Expected differences**:
- DQN might have lower confidence (Q-values can be noisy)
- Supervised might be over-confident (softmax on logits)

---

### 6. Error Analysis
**Supervised**: `q2_multisensory_error_analysis.png`
**DQN**: `question_2D_MSI_error_analysis.png`

**What to compare**:
- Error rate curves vs coherence
- Where do errors concentrate?
- Error rate trends

**Key question**: Do both approaches fail on the same difficult trials?

---

### 7. Choice Decoding Timecourse ⭐⭐⭐
**Supervised**: `q2_multisensory_choice_decoding.png`
**DQN**: `question_2D_MSI_choice_decoding.png`

**What to compare**:
- When does choice become decodable?
- Speed of information emergence
- Final decoding accuracy
- Architecture differences (vanilla vs bio-realistic)

**Key question**: Do RL and supervised develop different internal representations?

**Expected differences**:
- Supervised might have faster/cleaner decoding (direct gradients)
- DQN might have noisier representations (temporal credit assignment)

---

### 8. Temporal Dynamics ⭐⭐
**Supervised**: `q2_multisensory_temporal_dynamics.png`
**DQN**: `question_2D_MSI_temporal_dynamics.png`

**What to compare**:
- Evidence accumulation curves
- Speed of decision formation
- Variability (SEM bands)
- Separation between left/right choices

**Key question**: Does RL accumulate evidence differently than supervised?

**Expected differences**:
- Supervised might have smoother curves
- DQN might show more "commitment" (harder decisions due to Q-learning)

---

### 9. Radar Comparison
**Supervised**: `q2_multisensory_radar_comparison.png`
**DQN**: `question_2D_MSI_radar_comparison.png`

**What to compare**:
- Overall shape (balanced vs imbalanced performance)
- Which model architecture is best for each approach?
- Left vs right recall balance

**Key insight**: Single plot summarizing all metrics

---

### 10. PCA Trajectories
**Supervised**: `q2_multisensory_pca.png`
**DQN**: `question_2D_MSI_pca_trajectories.png`

**What to compare**:
- Trajectory smoothness
- Separation between choices
- Variance explained (PC1 + PC2)
- Architecture differences

**Key question**: Do RL and supervised use different neural codes?

---

### 11. Activity Heatmaps
**Supervised**: `q2_multisensory_heatmaps.png`
**DQN**: `question_2D_MSI_heatmaps.png`

**What to compare**:
- Temporal patterns of activity
- Unit selectivity
- Overall activity levels
- Architecture differences

---

### 12. Example Predictions
**Supervised**: `q2_multisensory_example_predictions.png`
**DQN**: `question_2D_MSI_example_predictions.png`

**What to compare**:
- Probability timecourses
- Decision timing (when does P cross 0.5?)
- Variability across trials
- Correct vs incorrect patterns

---

## Key Research Questions

### 1. Which learning approach is better?
**Compare**: Overall accuracy, balanced accuracy, all metrics
**Expected**: Supervised slightly better, but both should be >80%

### 2. Do biological constraints affect them differently?
**Compare**: Bio-realistic vs Vanilla in both approaches
**Look for**: 
- Does Feedback Alignment hurt RL more?
- Does L1/L2 regularization affect RL differently?
- Performance gap: (Vanilla - Bio) for each approach

### 3. Do they learn different representations?
**Compare**: Choice decoding, PCA trajectories, heatmaps
**Look for**:
- Different temporal dynamics
- Different hidden state structure
- Different information flow

### 4. Do they integrate multi-sensory information differently?
**Compare**: Modality weighting analysis
**Look for**:
- Different optimal weightings
- Different flexibility across weightings
- Integration strategies

### 5. Are they similarly calibrated?
**Compare**: Decision confidence
**Look for**:
- Confidence distributions
- Over/under-confidence
- Relationship between confidence and accuracy

### 6. Do they make the same errors?
**Compare**: Confusion matrices, error analysis, coherence analysis
**Look for**:
- Same difficult trials
- Same error types (fixate vs choose vs swap)
- Different systematic biases

## Quantitative Comparisons

### Extract Key Metrics from Both

Create a comparison table:

| Metric | Supervised (Vanilla) | DQN (Vanilla) | Supervised (Bio) | DQN (Bio) |
|--------|---------------------|---------------|------------------|-----------|
| Accuracy | ? | ? | ? | ? |
| Balanced Acc | ? | ? | ? | ? |
| Left Recall | ? | ? | ? | ? |
| Right Recall | ? | ? | ? | ? |
| Mean Confidence | ? | ? | ? | ? |
| Decoding (final) | ? | ? | ? | ? |

### Statistical Tests (Optional)
If you want to be rigorous:
- t-test: Supervised vs DQN accuracy
- Effect size: Cohen's d for performance difference
- Chi-square: Error pattern differences

## Writing the Comparison Section

### Structure

1. **Introduction**
   - "We compare supervised learning and reinforcement learning (DQN) on the MultiSensoryIntegration task"
   - "Both use identical RNN architectures with varying biological constraints"
   - "This enables fair comparison of learning paradigms"

2. **Overall Performance**
   - Present performance comparison plot
   - Table of accuracies
   - Statistical comparison

3. **Task Difficulty**
   - Coherence analysis comparison
   - "Both approaches show similar sensitivity to coherence..."
   - OR "RL is more robust to low coherence..."

4. **Multi-Sensory Integration**
   - Modality weighting comparison
   - "Supervised learning shows optimal integration at balanced weighting..."
   - "DQN demonstrates similar/different integration strategy..."

5. **Error Patterns**
   - Confusion matrices
   - "Both approaches primarily confuse left/right rather than choosing over fixating"
   - OR "RL tends to fixate more when uncertain..."

6. **Decision Dynamics**
   - Temporal dynamics comparison
   - Choice decoding comparison
   - "Supervised learning shows faster evidence accumulation..."
   - "DQN representations emerge more slowly but reach similar final performance..."

7. **Confidence Calibration**
   - Decision confidence comparison
   - "Supervised models are well-calibrated with mean confidence of X"
   - "DQN shows lower/higher confidence but similar calibration"

8. **Biological Constraints**
   - Compare Vanilla vs Bio for both approaches
   - "Feedback Alignment reduces performance by X% in supervised, Y% in DQN"
   - "Biological constraints affect both paradigms but impact is larger in RL/supervised"

9. **Conclusion**
   - "Both supervised and RL successfully solve the task (>80% accuracy)"
   - "Supervised shows slight advantage in performance but DQN demonstrates comparable integration"
   - "Key difference: temporal dynamics vs final representations"
   - "Biological constraints affect both but..."

## Quick Visual Inspection Checklist

When you generate all plots, look for:

- [ ] Both approaches achieve >80% accuracy
- [ ] Coherence curves have similar shapes
- [ ] Modality weighting curves show integration ability
- [ ] Confusion matrices show diagonal dominance
- [ ] Confidence distributions separate correct/incorrect
- [ ] Choice decoding reaches >70% by end of trial
- [ ] Temporal dynamics show rising P(correct choice)
- [ ] PCA trajectories separate by choice
- [ ] Heatmaps show temporal structure

## Files to Include in Report

### Must Include:
1. Performance comparison (both)
2. Coherence analysis (both)
3. Modality weighting (both)
4. Choice decoding (both)
5. Temporal dynamics (both)

### Nice to Include:
6. Confusion matrices
7. Decision confidence
8. Radar comparison
9. Example predictions

### Supplementary:
10. PCA trajectories
11. Heatmaps
12. Error analysis

## Timeline

1. **Run DQN training**: ~20-30 minutes
2. **Run DQN analysis**: ~3-4 minutes → 13 plots
3. **Run Supervised analysis**: ~1-2 minutes → 16 plots  
4. **Visual comparison**: ~30 minutes (go through each plot pair)
5. **Write comparison section**: ~2-3 hours

**Total**: ~4-5 hours for complete comparison

## Pro Tips

1. **Use consistent color schemes** across plots for easy comparison
2. **Create a dedicated folder** for final report figures
3. **Keep notes** while comparing (observations, questions, hypotheses)
4. **Focus on the story**: What's the key difference between RL and supervised?
5. **Quantify everything**: Don't just say "better", say "5% better"
6. **Look for interactions**: Do biological constraints affect RL differently?

Good luck with your analysis! 🚀
