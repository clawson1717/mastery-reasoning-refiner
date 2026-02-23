# Project: Mastery-Reasoning-Refiner (MRR)

## Concept
An RL-driven reasoning agent that iteratively refines its thought traces using iGRPO, with the training curriculum dynamically adjusted based on Pedagogical Mastery Learning (Bloom's) to maximize information-theoretic "world model" gain.

## Papers & Techniques

- **iGRPO** (NVIDIA): Provides the self-feedback-driven iterative refinement loop for reasoning.
- **Pedagogically-Inspired Data Synthesis** (ICLR 2026): Provides the "Mastery Learning" and "Zone of Proximal Development" (ZPD) framework for curriculum control.
- **Information-theoretic analysis of world models** (José Faustino): Provides the metric for measuring the implicit "world model" capacity (in bits) as the agent learns.

## Implementation Steps

### Step 1: Project Scaffold [DONE]
Initialize Python project with PyTorch and Transformers. Create directory structure for models, data, and training scripts.
**Continuity:** N/A

... (Refer to `memory/project-mastery-reasoning-refiner-plan.md` for full steps)
