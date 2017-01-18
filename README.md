gym-qbrush
==========
Drawing environments for OpenAI Gym.

Environments
------------
The state consists of the channel-wise concatenation of the current canvas,
the target image, and the "position map": a single channel with the same spatial dimensions as the images which represents the history of the write head.

The agent may move in the cardinal directions or choose to stop drawing.

QBrush-Final-v0
---------------
At each step, a reward of -1 is given. Another reward is given when the drawing
is complete based on how many times less error there is between the generated
image and the target image compared to a random image.

QBrush-Step-v0
---------------
At each step, if the current canvas error is less than the previous best error
then a reward proporitional to the ratio of the previous best to current best error
is given. Otherwise a small negative reward is given.
