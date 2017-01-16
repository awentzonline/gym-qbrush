gym-qbrush
==========
A drawing environment for OpenAI Gym.

State
-----
The state consists of the channel-wise concatenation of the current canvas,
the target image, and the "position map": a single channel with the same spatial dimensions as the images which represents the history of the write head.

Actions
-------
The agent may move in the cardinal directions or choose to stop drawing.

Rewards
-------
At each step, a reward of -1 is given. Another reward is given when the drawing
is complete based on how many times less error there is between the generated
image and the target image compared to a random image.
