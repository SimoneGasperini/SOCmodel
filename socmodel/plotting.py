import numpy as np
import pylab as plt


def show_simulation (Kplus, Kminus, branchPar, savefig=False):

  fig, ax = plt.subplots(figsize=(8,6))

  x = np.arange(Kplus.size)
  ax.plot(x, Kplus, label=r'In-degree $\langle K_{+} \rangle$')
  ax.plot(x, Kminus, label=r'In-degree $\langle K_{-} \rangle$')
  ax.plot(x[4000:], branchPar[4000:], label=r'Branching par $\langle \lambda \rangle$')

  ax.set_xlabel('Evolution steps', fontsize=16)
  ax.legend(fontsize=16)
  adjust_plot(ax=ax)

  plt.show()

  if savefig:
    fig.savefig('./images/simulation.pdf', bbox_inches='tight', dpi=1200)


def adjust_plot (ax):

  ax.set_axisbelow(True)
  ax.grid(linestyle='dotted')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)

  for tx in ax.xaxis.get_major_ticks():
    tx.label.set_fontsize(12)

  for ty in ax.yaxis.get_major_ticks():
    ty.label.set_fontsize(12)
