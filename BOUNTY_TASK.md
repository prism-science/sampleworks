# Bounty Objective: Reward contract tests for multiple conformations
Target: prism-science/sampleworks Issue #495
URL: https://github.com/prism-science/sampleworks/issues/495

## Issue Description
tests/rewards/test_reward_function_contract.py scores a single conformation. Its batch tests repeat that conformation B times at occupancy 1/B, which checks the batch mechanics but is not an ensemble of distinct conformations. The diffuse term of diffuse_bragg, $\langle |F|^2 \rangle - |\langle F \rangle|^2$, is identically zero for identical copies. Its multi-conformation tests therefore currently live in test_diffuse_bragg_reward.py: test_recovers_the_ensemble_it_was_generated_from, test_loss_increases_with_ensemble_displacement, test_every_configuration_receives_a_gradient and test_diffuse_gradient_points_downhill.

Proposal: let a RewardCase carry an optional multi-conformation input, coordinates [n_conformations, n_atoms, 3] with targets generated from them, and add a TestRewardEnsemble class to the contract with those four checks, run for every bundle that supplies one. diffuse_bragg would register its two-conformation fixtures from tests/rewards/conftest.py, one prepared reward per Bragg weight. The real-space and SF rewards could register the 1vme altloc pair they already generate targets for. Raised by Doris Mai in review of #381.

## Instructions for Kuro (Autonomous Solver):
1. Use skills: `systematic-debugging` and `code-review-and-quality`.
2. Inspect the repository structure, locate relevant files, and understand the bug/feature.
3. Apply the minimal, robust fix or implementation.
4. Run any existing test suites or linters in the workspace.
5. Ensure the fix adheres to the project's existing code style.
6. When done, output a summary of files modified and verification results.
