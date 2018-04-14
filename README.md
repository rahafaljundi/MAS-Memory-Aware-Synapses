# MAS
A pytorch implementation of Memory Aware Synapses: Learning what not to forget method. It allows a neural network to learn in a continual manner without catastrophic forgetting. Moreover, at deployment time while the system is applied to set of images, the method learns the important parts of the tasks that shouldn't be forgotten allowing extra freedom for later tasks.
The code has a demo file that shows a learning scenario in mnist split set of tasks. Soon, we will a sequence of 8 object recognition tasks as shown in the paper
