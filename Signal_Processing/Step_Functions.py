# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt

# i. Unit step sequence of length 10
unit_step = np.ones(10)

# ii. Discrete-time sinusoidal sequence of length 50, frequency 0.1
n = np.arange(50)
sin_seq = np.sin(0.1 * np.pi * n)

# iii. Exponential decay sequence of length 20
decay = 0.9 ** np.arange(20)

# i. Shift the unit step sequence 3 steps to the right
shifted_unit_step = np.concatenate((np.zeros(3), unit_step))

# ii. Compute the sum of the sinusoidal sequence and the decay sequence
# We need to first zero pad the decay sequence to the length of the sinusoidal sequence
padded_decay = np.concatenate((decay, np.zeros(30)))
sum_seq = sin_seq + padded_decay

# iii. Compute the convolution of the unit step sequence with the decay sequence
convolution = np.convolve(unit_step, decay, 'full')

plt.figure(figsize=(15, 10))

plt.subplot(321)
plt.stem(unit_step)
plt.title('Unit Step Sequence')

plt.subplot(322)
plt.stem(sin_seq)
plt.title('Sinusoidal Sequence')

plt.subplot(323)
plt.stem(decay)
plt.title('Decay Sequence')

plt.subplot(324)
plt.stem(shifted_unit_step)
plt.title('Shifted Unit Step Sequence')

plt.subplot(325)
plt.stem(sum_seq)
plt.title('Sum of Sinusoidal and Decay Sequences')

plt.subplot(326)
plt.stem(convolution)
plt.title('Convolution of Unit Step and Decay Sequences')

plt.tight_layout()
plt.show()
