* Added the ``kuka_cable_hand`` contrib task: a KUKA iiwa 7 with the cable-driven hand of the Newton
  ``exp/cable-driven-hand-series`` branch lifts a 4 cm cube to a target. The task is a Newton-native
  vectorized environment for RSL-RL (VBD with massless routed tendons, prescribed arm) and ships a trained
  policy, training, evaluation and video scripts, and an OVRTX renderer for recorded rollouts.
