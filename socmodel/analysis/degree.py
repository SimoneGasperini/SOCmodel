import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from scipy.special import factorial
from scipy.optimize import curve_fit

from socmodel.source.state import UniformState
from socmodel.source.connectivity import RandomConnectivity
from socmodel.source.network import Network



def plot_degree_distribution (savefig=False):

  socmodel = Network(n=1000, alpha=0.2, beta=10., T=10)
  print(socmodel, flush=True)
  socmodel.run(evolution_steps=20000)

  Kin = np.sum(np.abs(socmodel.C), axis=0)
  mean_Kin = round(np.mean(Kin), 3)
  bins = np.arange(Kin.min() - 0.5, Kin.max() + 1.5)

  fig, ax = plt.subplots(figsize=(8,6))

  entries, bin_edges, _ = ax.hist(Kin, bins=bins, color='black', density=True, histtype='bar', rwidth=0.7)
  ax.set_xticks(bins + 0.5)
  ax.set_xlabel('In-degree $K$', fontsize=14)
  ax.set_ylabel('Rel. frequency', fontsize=14)
  ax.text(x=0.7, y=0.9, s=f'$<K> \simeq$ {mean_Kin}', fontsize=14, transform=ax.transAxes)
  ax.set_axisbelow(True)
  ax.grid(linestyle='dotted')

  poisson = lambda k, lamb : (lamb**k / factorial(k)) * np.exp(-lamb)
  bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
  params, _ = curve_fit(poisson, bin_middles, entries)

  x = np.linspace(start=bin_middles[0], stop=bin_middles[-1], num=100)
  ax.plot(x, poisson(x, *params), color='red')

  plt.show()

  if savefig:
    fig.savefig('./images/K_distribution.pdf', bbox_inches='tight', dpi=1200)



def plot_degree_convergence (savefig=False):

  connectivity = [{'prob':0., 'col':'tab:blue'},
                  {'prob':0.003, 'col':'tab:orange'},
                  {'prob':0.006, 'col':'tab:green'}]

  fig1, ax1 = plt.subplots(figsize=(8,6))
  fig2, ax2 = plt.subplots(figsize=(8,6))

  for c in connectivity:

    socmodel = Network(n=600, alpha=0.2, beta=10., T=10,
                       C_init=RandomConnectivity(pPlus=c['prob'], pMinus=c['prob']))
    print(socmodel, flush=True)
    Kplus, Kminus, _, _ = socmodel.run(evolution_steps=20000)

    ax1.plot(Kplus, color=c['col'])
    ax2.plot(Kminus, color=c['col'])

  ax1.set_xlabel('Evolution steps', fontsize=14)
  ax1.set_ylabel('In-degree $<K_{+}$>', fontsize=14)
  ax1.set_axisbelow(True)
  ax1.grid(linestyle='dotted')

  ax2.set_xlabel('Evolution steps', fontsize=14)
  ax2.set_ylabel('In-degree $<K_{-}$>', fontsize=14)
  ax2.set_axisbelow(True)
  ax2.grid(linestyle='dotted')

  plt.show()

  if savefig:
    fig1.savefig('./images/Kplus_convergence.pdf', bbox_inches='tight', dpi=1200)
    fig2.savefig('./images/Kminus_convergence.pdf', bbox_inches='tight', dpi=1200)



def plot_degree_vs_beta (savefig=False):

  betas = np.linspace(start=0., stop=20., num=100)
  mean_Kplus = np.empty_like(betas)
  std_Kplus = np.empty_like(betas)
  mean_Kminus = np.empty_like(betas)
  std_Kminus = np.empty_like(betas)

  for i in trange(betas.size, desc='Running simulations'):

    socmodel = Network(n=400, alpha=0.2, beta=betas[i], T=10,
                       sigma_init=UniformState(),
                       C_init=RandomConnectivity(pPlus=0.005, pMinus=0.005))
    Kplus, Kminus, _, _ = socmodel.run(evolution_steps=10000, progressbar=False)
    mean_Kplus[i] = np.mean(Kplus[-1000:])
    std_Kplus[i] = np.std(Kplus[-1000:])
    mean_Kminus[i] = np.mean(Kminus[-1000:])
    std_Kminus[i] = np.std(Kminus[-1000:])

  print(socmodel, flush=True)

  fig1, ax1 = plt.subplots(figsize=(8,6))
  fig2, ax2 = plt.subplots(figsize=(8,6))

  ax1.errorbar(x=betas, y=mean_Kplus, yerr=std_Kplus, color='tab:blue',
               fmt='.', elinewidth=1, ecolor='black', capsize=2)
  ax1.set_xlabel('Inverse temp. $\\beta$', fontsize=14)
  ax1.set_ylabel('Avg. in-degree $<K_{+}$>', fontsize=14)
  ax1.set_axisbelow(True)
  ax1.grid(linestyle='dotted')

  ax2.errorbar(x=betas, y=mean_Kminus, yerr=std_Kminus, color='tab:orange',
               fmt='.', elinewidth=1, ecolor='black', capsize=2)
  ax2.set_xlabel('Inverse temp. $\\beta$', fontsize=14)
  ax2.set_ylabel('Avg. in-degree $<K_{-}$>', fontsize=14)
  ax2.set_axisbelow(True)
  ax2.grid(linestyle='dotted')

  plt.show()

  if savefig:
    fig1.savefig('./images/Kplus_vs_beta.pdf', bbox_inches='tight', dpi=1200)
    fig2.savefig('./images/Kminus_vs_beta.pdf', bbox_inches='tight', dpi=1200)
