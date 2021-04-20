import numpy as np
import pylab as plt


def show_simulation (Kplus, Kminus, branchPar):

  fig, ax = plt.subplots(figsize=(14,10))

  x = np.arange(Kplus.size)
  ax.plot(x, Kplus, label='In-degree $<K_{+}>$')
  ax.plot(x, Kminus, label='In-degree $<K_{-}>$')
  ax.plot(x[3000:], branchPar[3000:], label='Branching par $<\lambda>$')

  ax.set_xlabel('Evolution steps', fontsize=14)
  ax.legend(fontsize=12)
  ax.set_axisbelow(True)
  ax.grid(linestyle='dotted')

  plt.show()
