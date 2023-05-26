# Code base for "Plan-Based Relaxed Reward Shaping for Goal-Directed Tasks"
This repository contains code for reproducing results of the following paper:
```
Schubert, I., Oguz, O.S. and Toussaint, M. (2021). Plan-based relaxed reward shaping for goal-directed tasks. In Proceedings of the International Conference on Learning Representations (ICLR).
```
Specifically, the experiment code in this repo will reproduce Fig. 2 (pushing example) and Figs. 6e and 6f (pick-and-place example).
The rest of the examples can be reproduced analogously.

## Installation
The physX [simulations require](https://github.com/ischubert/fv-rs/blob/dc21b01271fc2a33cc4b3050164f94c7846fcf64/ryenv.py#L8) `rai-python` (imported here). Please follow the installation instructions at https://github.com/MarcToussaint/rai-python.

## Pushing example
![](pushing_setup_annotated.png)

There are 3 versions, for
- No reward shaping: `001_pushing_no_rs.py`
- Potential-based reward shaping: `002_pushing_pb_rs.py`
- Asymptotically equivalent reward shaping: `003_pushing_aseq_rs.py`

## Pick-and-place example
![](pick_and_place_setup_annotated.png)

There are 3 versions, for
- No reward shaping: `004_pick_and_place_no_rs.py`
- Potential-based reward shaping: `005_pick_and_place_pb_rs.py`
- Asymptotically equivalent reward shaping: `006_pick_and_place_aseq_rs.py`
