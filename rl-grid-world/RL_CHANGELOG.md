# Changelog
All notable changes to this project will be documented in this file.

## [Current] - 06-Mar-2026
### Added
- Implemented Temporal Difference TD(0) learning method (`update_td_zero`).
- Added a living penalty (`-0.1` step cost) to `GridWorld.prepare_grid()` to penalize laziness and prevent policy oscillation.

### Fixed
- Resolved the "Invisible Wall" bug in `GridWorld.move()` by ensuring the environment returns the current position and a penalty instead of `None` when hitting boundaries.

## [Initial Setup] - 20-Feb-2026
### Added
- Built the foundational `GridWorld` environment with customizable start/goal states.
- Implemented First-Visit Monte Carlo Prediction (`update_state_values`).
- Created an $\epsilon$-greedy action selection mechanism for the `Agent`.