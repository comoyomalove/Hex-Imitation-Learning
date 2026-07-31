# 11x11 Imitation Training Report

- run_id: imitation_11x11_structured_plus_dagger_round2_combined_20260729_011307
- device: mps
- board_size: 11
- train_examples: 120762
- validation_examples: 25937
- test_examples: 26666
- epochs: 100
- best_epoch: 32
- best_validation_loss: 1.0886
- best_validation_top1_accuracy: 0.7083
- best_validation_top5_accuracy: 0.9289
- best_checkpoint: models/checkpoints/imitation_11x11_structured_plus_dagger_round2_combined_20260729_011307_best.pt
- final_model_copy: models/final/imitation_11x11_best.pt

Interpretation: this trains on saved 11x11 expert demonstrations.
The trained model learns to predict expert moves from board states.
It does not call the expert agent during inference.
