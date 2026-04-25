# The-Self-Pruning-Neural-Network
This repository contains the implementation of a custom "self-pruning" neural network trained on the CIFAR-10 dataset. Instead of using post-training pruning techniques, this network dynamically learns to identify and remove its own weakest connections during the training loop using learnable gate parameters and L1 regularization.

## Overview

This project implements a **self-pruning neural network** using learnable gates applied to weights.
The model automatically learns which connections are unnecessary and suppresses them during training.

The key idea:

* Each weight is multiplied by a **learnable gate**
* Gates are trained with **L1 sparsity pressure**
* Small gates → effectively remove connections

---

## Key Features

###  Learnable Weight Pruning

* Each linear layer is replaced with `PrunableLinear`
* Uses:

  ```
  pruned_weight = weight × sigmoid(gate_scores)
  ```
* Gates are trained jointly with model weights

---

###  Smart Initialization (Critical Fix)

* Gate scores initialized at **0**
* So:

  ```
  sigmoid(0) = 0.5
  ```
* This places gates in the **high-gradient region**, enabling effective pruning

---

###  Warmup Training Phase

* First few epochs train **only for accuracy**
* Sparsity loss added **after warmup**
* Prevents early collapse

---

###  Separate Learning Rates

* Gates use higher LR:

  * Faster response to sparsity pressure
* Weights use standard LR

---

###  CNN + Prunable Head Design

* Backbone: standard CNN (efficient already)
* Head: fully connected layers (high redundancy → pruned)

---

## Architecture

### Backbone (not pruned)

* Conv → BN → ReLU → Pool × 3
* AdaptiveAvgPool → 256-d vector

### Prunable Classifier

* 256 → 512 → 256 → 10
* All layers use `PrunableLinear`

---

## Loss Function

Total loss:

```
Loss = CrossEntropy + λ × SparsityLoss
```

Where:

```
SparsityLoss = sum(sigmoid(gate_scores))
```

---

## Training Strategy

| Phase  | Description               |
| ------ | ------------------------- |
| Warmup | Train only classification |
| Prune  | Add sparsity loss         |

---

## Results

| Lambda | Test Accuracy | Sparsity |
| ------ | ------------- | -------- |
| 1e-3   | ~92.17%       | ~99.8%   |
| 1e-2   | ~92.20%       | 100%     |
| 1e-1   | ~91.71%       | 100%     |

### Key Insight

* Nearly **all classifier weights removed**
* Minimal accuracy drop (~0.5%)

---

## Visualization

The project generates:

* Gate distribution histograms
* Zoomed view near zero
* Comparison across λ values

---

## How to Run

```bash
python main.py
```

---

## Output

* Training logs
* Accuracy & sparsity summary
* Plots:

  * `gate_distribution_best.png`
  * `all_gate_distributions.png`

---

## Why This Works

* Sigmoid gates provide smooth pruning
* L1 pressure pushes gates → 0
* Warmup avoids destroying early learning
* CNN backbone preserves feature quality

---

## Limitations

* Only prunes **fully connected layers**
* Does NOT reduce:

  * CNN computation
  * inference FLOPs significantly

---

## Possible Improvements

* Apply pruning to convolution layers
* Convert soft gates → hard pruning mask
* Export compressed model
* Structured pruning (channels / filters)
* Compare with magnitude pruning

---

## Conclusion

This implementation successfully demonstrates:

* End-to-end **differentiable pruning**
* Extreme sparsity with minimal accuracy loss
* Stable training with proper initialization and scheduling

---

