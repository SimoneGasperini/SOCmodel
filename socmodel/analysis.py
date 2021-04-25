import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from scipy.special import factorial
from scipy.optimize import curve_fit

from socmodel.source.state import RandomState
from socmodel.source.connectivity import RandomConnectivity
from socmodel.source.network import Network
from socmodel.plotting import adjust_plot


def plot_degree_distribution (savefig=False):

  socmodel = Network(n=2000, alpha=0.2, beta=10., tau=10)
  print(socmodel, flush=True)
  socmodel.run(evolution_steps=20000)

  Kin = np.sum(np.abs(socmodel.C.toarray()), axis=0)
  mean_Kin = round(np.mean(Kin), 3)
  bins = np.arange(Kin.min() - 0.5, Kin.max() + 1.5)

  fig, ax = plt.subplots(figsize=(8,6))

  entries, bin_edges, _ = ax.hist(Kin, bins=bins, color='black', density=True, histtype='bar', rwidth=0.7)
  ax.set_xticks(bins + 0.5)
  ax.set_xlabel('In-degree $K$', fontsize=16)
  ax.set_ylabel('Rel. frequency', fontsize=16)
  ax.text(x=0.7, y=0.9, s=fr'$\langle K \rangle \simeq$ {mean_Kin}', fontsize=16, transform=ax.transAxes)

  poisson = lambda k, lamb : (lamb**k / factorial(k)) * np.exp(-lamb)
  bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
  params, _ = curve_fit(poisson, bin_middles, entries)

  x = np.linspace(start=bin_middles[0], stop=bin_middles[-1], num=100)
  ax.plot(x, poisson(x, *params), color='red')
  adjust_plot(ax=ax)

  plt.show()

  if savefig:
    fig.savefig('./images/K_distribution.pdf', bbox_inches='tight', dpi=1200)


def plot_degree_convergence (savefig=False):

  connectivity = [{'prob':0., 'col':'tab:blue'},
                  {'prob':0.002, 'col':'tab:orange'},
                  {'prob':0.004, 'col':'tab:green'}]

  fig1, ax1 = plt.subplots(figsize=(8,6))
  fig2, ax2 = plt.subplots(figsize=(8,6))

  for c in connectivity:

    socmodel = Network(n=1000, alpha=0.2, beta=10., tau=10,
                       C_init=RandomConnectivity(pPlus=c['prob'], pMinus=c['prob']))
    print(socmodel, flush=True)
    Kplus, Kminus, _ = socmodel.run(evolution_steps=30000)

    k = c['prob'] * socmodel.n
    ax1.plot(Kplus, color=c['col'], label=r'$\langle K_{+} \rangle ^{ini} \simeq$ ' + str(k))
    ax2.plot(Kminus, color=c['col'], label=r'$\langle K_{-} \rangle ^{ini} \simeq$ ' + str(k))

  ax1.set_xlabel('Evolution steps', fontsize=16)
  ax1.set_ylabel(r'In-degree $\langle K_{+} \rangle$', fontsize=16)
  ax1.legend(fontsize=16)
  adjust_plot(ax=ax1)

  ax2.set_xlabel('Evolution steps', fontsize=16)
  ax2.set_ylabel(r'In-degree $\langle K_{-} \rangle$', fontsize=16)
  ax2.legend(fontsize=16)
  adjust_plot(ax=ax2)

  plt.show()

  if savefig:
    fig1.savefig('./images/Kplus_convergence.pdf', bbox_inches='tight', dpi=1200)
    fig2.savefig('./images/Kminus_convergence.pdf', bbox_inches='tight', dpi=1200)


def plot_degree_vs_beta (savefig=False):

  betas = np.linspace(start=0., stop=20., num=60)
  mean_Kplus = np.empty_like(betas)
  std_Kplus = np.empty_like(betas)
  mean_Kminus = np.empty_like(betas)
  std_Kminus = np.empty_like(betas)

  for i in trange(betas.size, desc='Running simulations'):

    socmodel = Network(n=400, alpha=0.2, beta=betas[i], tau=10,
                       sigma_init=RandomState(),
                       C_init=RandomConnectivity(pPlus=0.005, pMinus=0.005))
    Kplus, Kminus, _ = socmodel.run(evolution_steps=10000, progressbar=False)
    mean_Kplus[i] = np.mean(Kplus[-1000:])
    std_Kplus[i] = np.std(Kplus[-1000:])
    mean_Kminus[i] = np.mean(Kminus[-1000:])
    std_Kminus[i] = np.std(Kminus[-1000:])

  fig1, ax1 = plt.subplots(figsize=(8,6))
  fig2, ax2 = plt.subplots(figsize=(8,6))

  ax1.errorbar(x=betas, y=mean_Kplus, yerr=std_Kplus, color='tab:blue',
               fmt='.', elinewidth=1, ecolor='black', capsize=2)
  ax1.set_xlabel('Inverse temp. $\\beta$', fontsize=16)
  ax1.set_ylabel(r'Avg. in-degree $\langle K_{+} \rangle$', fontsize=16)
  adjust_plot(ax=ax1)

  ax2.errorbar(x=betas, y=mean_Kminus, yerr=std_Kminus, color='tab:orange',
               fmt='.', elinewidth=1, ecolor='black', capsize=2)
  ax2.set_xlabel('Inverse temp. $\\beta$', fontsize=16)
  ax2.set_ylabel(r'Avg. in-degree $\langle K_{-} \rangle$', fontsize=16)
  adjust_plot(ax=ax2)

  plt.show()

  if savefig:
    fig1.savefig('./images/Kplus_vs_beta.pdf', bbox_inches='tight', dpi=1200)
    fig2.savefig('./images/Kminus_vs_beta.pdf', bbox_inches='tight', dpi=1200)
