gym-qbrush
==========
A drawing environment for OpenAI Gym.

Actions
-------
The agent may move in the cardinal directions or choose to stop drawing.

Rewards
-------
At each step, a reward of -1 is given. Another reward is given when the drawing
is complete based on how many times less error there is between the generated
image and the target image compared to a random image.
