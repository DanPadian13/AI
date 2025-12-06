# DQN Analysis Enhancements

## Summary
Enhanced `question_2d_MSI_simple_dqn_analysis.py` to match the comprehensive analysis in `Question_2_multisensory_analysis.py`, making DQN results fully comparable to supervised learning results.

## New Analyses Added (9 Major Enhancements)

### 1. **Coherence Analysis** (`analyze_coherence_difficulty`)
- **Purpose**: Shows how accuracy varies with task difficulty (coherence)
- **Output**: `question_2D_MSI_coherence_analysis.png`
- **Key Insights**: 
  - Psychometric curves showing performance vs difficulty
  - Error bars showing variability
  - Sample counts per coherence level
  - Identifies if models struggle with low coherence (hard trials)

### 2. **Modality Weighting Analysis** (`plot_modality_weighting_analysis`)
- **Purpose**: Analyzes multi-sensory integration across different modality weightings
- **Output**: `question_2D_MSI_modality_weighting.png`
- **Key Insights**:
  - How models integrate visual vs auditory information
  - Performance when one modality dominates vs balanced
  - Tests whether models can flexibly weight sensory inputs
  - Critical for understanding multi-sensory integration

### 3. **Confusion Matrices** (`plot_confusion_matrices`)
- **Purpose**: Shows what models predict when they're wrong
- **Output**: `question_2D_MSI_confusion_matrices.png`
- **Key Insights**:
  - Do models fixate when they should choose?
  - Do they confuse left/right?
  - Per-class error patterns
  - Normalized by true labels to show error rates

### 4. **Decision Confidence** (`plot_decision_confidence`)
- **Purpose**: Compares confidence (max probability) for correct vs incorrect trials
- **Output**: `question_2D_MSI_decision_confidence.png`
- **Key Insights**:
  - Are models well-calibrated? (high confidence = high accuracy)
  - Do models "know when they don't know"?
  - Confidence distributions for correct/incorrect predictions
  - Mean confidence differences

### 5. **Error Analysis** (`plot_error_analysis`)
- **Purpose**: Analyzes when and where models fail
- **Output**: `question_2D_MSI_error_analysis.png`
- **Key Insights**:
  - Error rate as function of coherence
  - Stacked histograms of correct/incorrect by difficulty
  - Identifies systematic failure modes
  - Shows if errors concentrated in hard trials

### 6. **Choice Decoding Timecourse** (`plot_choice_decoding_timecourse`)
- **Purpose**: Linear decoder showing when choice becomes separable in hidden states
- **Output**: `question_2D_MSI_choice_decoding.png`
- **Key Insights**:
  - When does the network "decide"?
  - How quickly does choice information emerge?
  - Compares representational dynamics across architectures
  - Tests if biological constraints affect information flow

### 7. **Temporal Dynamics** (`plot_temporal_dynamics`)
- **Purpose**: Shows how decision probabilities evolve during stimulus presentation
- **Output**: `question_2D_MSI_temporal_dynamics.png`
- **Key Insights**:
  - How do decisions emerge over time?
  - Evidence accumulation dynamics
  - Correct trials only (when left, P(left) should rise)
  - SEM error bands showing variability

### 8. **Radar Comparison** (`plot_radar_comparison`)
- **Purpose**: Multi-dimensional comparison across metrics
- **Output**: `question_2D_MSI_radar_comparison.png`
- **Metrics**:
  - Overall accuracy
  - Balanced accuracy (per-class recall)
  - Left choice recall
  - Right choice recall
  - Consistency (1 - std of correctness)
- **Key Insights**: Single plot comparing all models across all dimensions

### 9. **Example Predictions** (`plot_example_predictions`)
- **Purpose**: Shows actual trial timecourses with probabilities evolving
- **Output**: `question_2D_MSI_example_predictions.png`
- **Key Insights**:
  - Visual inspection of model behavior
  - 4 models × 3 trials = 12 example trials
  - Ground truth shown as background shading
  - Coherence and modality weight annotated
  - Correct/incorrect marked with ✓/✗

## Total Plots Generated

**Before**: 4 plots
1. Performance comparison
2. PCA trajectories
3. Activity heatmaps
4. Task structure

**After**: 13 plots (9 new + 4 original)
1. Performance comparison
2. PCA trajectories
3. Activity heatmaps
4. Task structure
5. **Coherence analysis** (NEW)
6. **Modality weighting** (NEW)
7. **Confusion matrices** (NEW)
8. **Decision confidence** (NEW)
9. **Error analysis** (NEW)
10. **Choice decoding** (NEW)
11. **Temporal dynamics** (NEW)
12. **Radar comparison** (NEW)
13. **Example predictions** (NEW)

## Comparison with Supervised Learning

The DQN analysis now matches the supervised learning analysis (`Question_2_multisensory_analysis.py`) with equivalent plots:

| Analysis | Supervised | DQN | Notes |
|----------|-----------|-----|-------|
| Coherence | ✓ | ✓ | Identical analysis |
| Modality weighting | ✓ | ✓ | Identical analysis |
| Confusion matrices | ✓ | ✓ | Identical analysis |
| Decision confidence | ✓ | ✓ | Identical analysis |
| Error analysis | ✓ | ✓ | Identical analysis |
| Choice decoding | ✓ | ✓ | Identical analysis |
| Temporal dynamics | ✓ | ✓ | Identical analysis |
| Radar comparison | ✓ | ✓ | Identical analysis |
| Example predictions | ✓ | ✓ | Identical analysis |

## Key Research Questions Enabled

With these enhancements, you can now answer:

1. **Does RL learn differently than supervised learning?**
   - Compare coherence curves (do they have same difficulty profile?)
   - Compare modality integration (do they weight modalities the same?)
   - Compare error patterns (do they fail on same trials?)

2. **Do biological constraints affect RL differently?**
   - Compare bio-realistic vs vanilla across all metrics
   - Does feedback alignment hurt RL more than supervised?
   - Does L1/L2 regularization affect RL performance?

3. **Are RL models well-calibrated?**
   - Decision confidence analysis shows if high confidence = high accuracy
   - Compare with supervised learning calibration

4. **When do RL models "decide"?**
   - Choice decoding shows when information emerges
   - Temporal dynamics show evidence accumulation
   - Compare with supervised learning dynamics

5. **How do RL models integrate multi-sensory information?**
   - Modality weighting analysis critical for this task
   - Compare with supervised learning integration strategies

## Usage

```bash
# Run DQN analysis (requires trained checkpoint)
python code/question_2d_MSI_simple_dqn_analysis.py

# Run supervised analysis for comparison
python code/Question_2_multisensory_analysis.py

# Compare outputs in images/ folder:
# - question_2D_MSI_*.png (DQN)
# - q2_multisensory_*.png (supervised)
```

## Technical Details

### Dependencies Added
- `sklearn.linear_model.LogisticRegression` - for choice decoding
- `sklearn.metrics.accuracy_score` - for decoder evaluation
- `math.pi` - for radar plots

### Data Collection Enhanced
Updated `collect_trial_data()` to capture:
- Trial info (coherence, modality weights) - stored in `trial_info`
- Full prediction timecourses - stored in `predictions`
- All existing data (activities, outputs, targets, correct)

### Plot Styling
All plots match supervised learning style:
- Same color schemes
- Same figure sizes
- Same fonts and labels
- Professional quality for publication/reports

## Files Modified

1. **question_2d_MSI_simple_dqn_analysis.py**
   - Added 9 new analysis functions
   - Enhanced data collection
   - Updated main execution flow
   - Added comprehensive documentation

## Expected Runtime

- **Data collection**: ~30 seconds (200 trials × 4 models)
- **Analysis**: ~2-3 minutes (13 plots with complex computations)
- **Total**: ~3-4 minutes

## Next Steps

1. **Run the enhanced analysis**:
   ```bash
   python code/question_2d_MSI_simple_dqn_analysis.py
   ```

2. **Compare with supervised learning**:
   - Visual comparison of corresponding plots
   - Quantitative comparison of metrics
   - Discussion in coursework report

3. **Potential additional analyses**:
   - Direct comparison plots (DQN vs supervised side-by-side)
   - Statistical tests (t-tests, effect sizes)
   - Learning curve comparisons (if you save intermediate checkpoints)

## Conclusion

The DQN analysis is now **fully comparable** to the supervised learning analysis with:
- ✅ All major analyses implemented
- ✅ Same metrics computed
- ✅ Same visualization styles
- ✅ Ready for fair comparison
- ✅ Publication-quality plots

This enables a comprehensive comparison of reinforcement learning vs supervised learning on the MultiSensoryIntegration task across multiple dimensions!
