import os
import time
import math
import tqdm
import multiprocessing
import multiprocessing.pool
import pathlib

import pickle
import torch
import numpy as np
import scipy
# from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import matplotlib

from operator import mul
from functools import reduce

from malogs import malog
N_CPU = multiprocessing.cpu_count()-2
EPS = np.finfo(np.float64).eps



import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func



def get_values(a, feat='e', stage='attention', use_layers=None, use_abs=True, flatten=True):
    if type(a) == dict:
        data = [(t, a[f'layers.malog_{t}']) for t in feat]
    else:
        data = [('e', a)]
    if use_layers is None:
        use_layers = range(len(data[0][1]))
    logs = {k:[] for k in use_layers}
    for t, layers in data:
        # print(t)
        # try:
        if True:
            for i in use_layers:
                ae = layers[i]
                z = []
                if use_layers is None:
                    use_layers = range(len(ae))
                for n in range(len(ae)):
                    x = ae[n][stage].detach()
                    d = reduce(mul, x.shape[1:], 1)
                    x = x.reshape(-1, d)
                    if use_abs:
                        x = x.abs()
                    logs[i].append(x)
                logs[i] = torch.cat(logs[i])
                if flatten:
                    logs[i] = logs[i].reshape(-1)
    return logs



def load_spec(spec):
    f = malog
    for x in spec:
        if type(x) in [str, int]:
            f = f[x]
        else:
            for y in x:
                z = f.get(y)
                if z is not None:
                    f = z
                    break
    with open(f, 'rb') as infile:
        a = pickle.load(infile)
    return a



def ks_ll_plot(data_raw, key, layer=0, xmin=None, bottom=0.1, fit=True, save=True, dist=scipy.stats.gamma, dist_label='gamma', scale169=4.2, normalize=True, dir='gamma', xlabel='ratio', batch_median=False, rect=None, ann=None, ann_coord=None, dist_params_dict={}):
    res = {}
    try:
        key_layer = tuple(key) + (layer,)
        print(f'start {key} {layer}\n', end='')
        data = data_raw.abs().numpy()
        if normalize:
            if batch_median:
                med = np.median(data)
            else:
                med = np.median(data, axis=1).reshape(-1,1)
            data = data / med
        data = data.reshape(-1)
        data = -np.log10(data+EPS)
        if xmin is not None:
            data = data[data>=xmin]
        # fit dist
        dist_params = dist_params_dict.get(key_layer)
        # if fit:
        if dist_params is None:
            dist_params = dist.fit(data)
            dist_params_dict[key_layer] = dist_params
        # compute dist pdf
        dist_x = np.concatenate([
            np.linspace(dist_params[-2], data.min(), 100, endpoint=False),
            np.linspace(data.min(), data.max(), 900)
        ])
        dist_pdf = dist.pdf(dist_x, *dist_params)
        # compute data loglikelihood
        loglikelihood = np.mean(dist.logpdf(data, *dist_params))
        res['loglikelihood'] = loglikelihood
        # compute Kolmogorov-Smirnov statisic
        ksres = scipy.stats.kstest(data, lambda x:dist.cdf(x,*dist_params))
        res['kstest'] = ksres
        # plot
        k = len(data)*((data.max()-data.min())/1000)
        dist_pdf = dist_pdf*k
        pdf_mask = dist_pdf >= bottom/10
        dist_x = dist_x[pdf_mask]
        dist_pdf = dist_pdf[pdf_mask]
        for shape_i, shape in enumerate([(1.6*scale169,0.9*scale169)]):#, (6,4.5)]):
            fig, ax = plt.subplots(figsize=shape)
            ax.set_yscale('log')
            ax.plot(dist_x, dist_pdf, color='C3')
            ax.hist(data, bins=1000)#, density=True)
            ax.axvline(x=-3, color='#000000', linestyle='dashed', label='MA threshold')
            ax.set_ylim(bottom=bottom)
            ax.legend([
                f'{dist_label} approximation pdf',
                'relative threshold (1000)',
                'ratio histogram'
            ])
            ax.set_xlabel('$-\log(\\text{%s})$'%xlabel)
            ax.set_ylabel('count')
            ax.set_title(' '.join(key) + f' - layer {layer}')# - ks {ksres.statistic:.3f} - avg. loglikelihood {loglikelihood:.3f}')
            ax.annotate(f'loglikelihood (avg): {loglikelihood:.3f}', xy=(shape[0]**2*1, shape[1]*0.4), xycoords='figure pixels')
            ax.annotate(f'ks statistic: {ksres.statistic:.3f}', xy=(shape[0]*52.5, shape[1]*0.4), xycoords='figure pixels')
            if rect is not None:
                ax.add_patch(rect)
            if ann is not None and ann_coord is not None:
                plt.gca().annotate(ann, xy=ann_coord, xycoords='data')
            if save:
                name = '_'.join(list(key)+[str(layer)])
                path = os.path.join('out', 'plots', dir, str(shape_i), *key)
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                fig.savefig(os.path.join(path, f'{name}.png'))
                fig.savefig(os.path.join(path, f'{name}.pdf'))
    except Exception as e:
        print(f'ERROR: {key} {layer} {e}\n', end='')
    print(f'end {key} {layer}\n', end='')
    return (tuple(key) + (layer,), res)

def ks_ll_plot_w(params):
    return ks_ll_plot(*params)



# function to plot base ranges
def plot2a(a, name=None, k=1, feat='he', stage='attention', axis=None, title="batches' maximums - %s", legend=True, batch_median=False, color=None, spec=None, all_feat=False):
    if a is not None and type(a) == dict:
        data = [(t, a[f'layers.malog_{t}']) for t in feat]
    else:
        data = [('e', a)]
    for t, layers in data:
        try:
            if axis is None:
                ax = plt
                ax.figure(figsize=(12.8,7.2))
                ax.ylim(1, 1e6)
                ax.yscale('log')
            else:
                ax = axis
            zz = []
            for i,ae in enumerate(layers):
                z = []
                for n in range(len(ae)):
                    x = ae[n][stage].detach()
                    d = reduce(mul, x.shape[1:], 1)
                    x = x.reshape(-1, d).abs()
                    if batch_median:
                        med = x.median()
                    else:
                        med = x.median(axis=1).values
                    y = torch.topk(x, k, dim=1).values[:,k-1] / med
                    y = y[~torch.isnan(y)]
                    if all_feat:
                        z.extend(list(y.numpy()))
                    else:
                        y = y.sort().values
                        z.append(y[-1].item())
                    if len(z) >= 60000:
                        break
                z.sort()
                zz.append(z)
            zz_min = min(min(z) for z in zz)
            zz_max = max(max(z) for z in zz)
            ax.fill_between(np.arange(max(len(z) for z in zz)), zz_min, zz_max, label='base model\nactivations ratio\n(range)')
            if legend:
                ax.legend()
            if title is not None:
                if title.count('%s') == 1:
                    tit = title%(t)
                else:
                    title = tit
                ax.title(tit)
            if name is not None:
                dir_ = os.path.join('out', 'plots', name)
                os.makedirs(dir_, exist_ok=True)
                ax.savefig(os.path.join(dir_, f'all-{t}.png'))
                ax.show()
        except Exception as e:
            print(e)
        finally:
            if axis is None:
                ax.close()



# function to plot trained model ratios
def plot2(a, name=None, k=1, feat='he', stage='attention', axis=None, title="batches' maximums - %s", legend=True, batch_median=False, color=None, alpha=None, spec=None, all_feat=False, plot_threshold=True):
    if a is not None and type(a) == dict:
        data = [(t, a[f'layers.malog_{t}']) for t in feat]
    else:
        data = [('e', a)]
    for t, layers in data:
        try:
            if axis is None:
                ax = plt
                ax.figure(figsize=(12.8,7.2))
                ax.ylim(1, 1e6)
                ax.yscale('log')
            else:
                ax = axis
            zz = []
            for i,ae in enumerate(layers):
                z = []
                for n in range(len(ae)): # batch #n
                    x = ae[n][stage].detach()
                    d = reduce(mul, x.shape[1:], 1)
                    x = x.reshape(-1, d).abs()
                    if batch_median:
                        med = x.median() # median of whole batch
                    else:
                        med = x.median(axis=1).values # median of single node/edge's features
                    y = torch.topk(x, k, dim=1).values[:,k-1] / med
                    y = y[~torch.isnan(y)]
                    if all_feat:
                        z.extend(list(y.numpy()))
                    else:
                        y = y.sort().values
                        z.append(y[-1].item())
                    if len(z) >= 60000:
                        break
                z.sort()
                zz.append(z)
            for i,z in enumerate(zz):
                if i == 0:
                    if plot_threshold:
                        ax.plot([1000]*len(z), color='#000000', linestyle='dashed', label='MA threshold')
                    ax.plot(z, color=color, alpha=alpha, label=f'trained model\nactivations ratio\n(layers)')
                else:
                    ax.plot(z, color=color, alpha=alpha)
            if legend:
                ax.legend()
            if title is not None:
                if title.count('%s') == 1:
                    tit = title%(t)
                else:
                    title = tit
                ax.title(tit)
            if name is not None:
                dir_ = os.path.join('out', 'plots', name)
                os.makedirs(dir_, exist_ok=True)
                ax.savefig(os.path.join(dir_, f'all-{t}.png'))
                ax.show()
        except Exception as e:
            print(e)
        finally:
            if axis is None:
                ax.close()
