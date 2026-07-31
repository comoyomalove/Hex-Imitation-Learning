# 11x11 Imitation Training Report

- run_id: imitation_11x11_dagger_round2_only_20260729_002533
- device: mps
- board_size: 11
- train_examples: 24716
- validation_examples: 5274
- test_examples: 5291
- epochs: 100
- best_epoch: 72
- best_validation_loss: 0.5861
- best_validation_top1_accuracy: 0.8684
- best_validation_top5_accuracy: 0.9744
- best_checkpoint: models/checkpoints/imitation_11x11_dagger_round2_only_20260729_002533_best.pt
- final_model_copy: models/final/imitation_11x11_best.pt

Interpretation: this trains on saved 11x11 expert demonstrations.
The trained model learns to predict expert moves from board states.
It does not call the expert agent during inference.
