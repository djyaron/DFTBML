#%%
'''Imports, Constants and Dataset Path'''

import os
import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from h5py import File
from random import Random
from sklearn.linear_model import LinearRegression

HARTREE = 627.50947406
ANGSTROM2BOHR = 1.889725989

dataset_path = '../ANI-1ccx_energy_clean.h5'

# Helper for reading df (previously written by Francis)
def flatten(dataset):
    # Determine atom types
    Zs = set()
    for moldata in dataset.values():
        Zs.update(moldata['atomic_numbers'][()])
    Zs = tuple(sorted(Zs))
    ZtoName = {1: "nH",
               6: "nC",
               7: "nN",
               8: "nO",
               79: "nAu"}
    for Z in Zs:
        ZtoName.get(Z, f"Atom type {Z} is not included in ZtoName dictionary")
    # Generate flattened dataset
    df = {}
    for mol, moldata in dataset.items():
        nconfs = len(moldata['coordinates'])
        conf_arr = np.arange(nconfs)
        # Column of molecular formulas
        mol_col = [mol] * nconfs
        try:
            df['mol'].extend(mol_col)
        except KeyError:
            df['mol'] = []
            df['mol'].extend(mol_col)
        # Column of conformation indices
        conf_col = np.arange(nconfs, dtype='int')
        try:
            df['conf'].extend(conf_col)
        except KeyError:
            df['conf'] = []
            df['conf'].extend(conf_col)
        # Columns of atomic numbers
        for Z in Zs:
            nZ = list(moldata['atomic_numbers'][()]).count(Z)
            Z_col = [nZ] * nconfs
            try:
                df[ZtoName[Z]].extend(Z_col)
            except KeyError:
                df[ZtoName[Z]] = []
                df[ZtoName[Z]].extend(Z_col)
        # Column of data
        for entry, data in moldata.items():
            if entry in ('atomic_numbers', 'coordinates'):
                continue
            try:
                df[entry].extend(data)
            except KeyError:
                df[entry] = []
                df[entry].extend(data)
    
    return pd.DataFrame(df)

flatten_path = '../cfe_df.pkl'
df = pd.read_pickle(flatten_path)

#%%
''' Outlier Visualizer (std-based) '''

''' 
Formats and makes removed points summary 
  diccy - Dictionary containing removed points
  means - Dictionary containing mean of each bucket (# heavy atoms)
  stds - Dictionary containing standard deviation of each bucket
  mult - Std multiplier (points above mean + mult*std are removed)
  reverseSearch - Hacky dictionary to turn data points (double-as-string) back to molecules
'''
def formatDictPrint(diccy, means, stds, mult, reverseSearch):
  ret = [] # Stringbuilder, since string concatentation might take a while otherwise
  keys = sorted(list(diccy.keys()))

  for k in keys:
    numOutlier = len(diccy[k])
    if numOutlier != 0:
      arr = [] # Stringbuilder, same as above
      for i in range(numOutlier):
        tmp = str(reverseSearch[str(diccy[k][i])]).strip("[").strip("]")

        # Formatting (Separated into chunks of 5) 
        if i % 5 == 4:
          tmp += "\n"
        elif i < numOutlier - 1:
          tmp += ", "

        arr.append(tmp)
      arrStr = ''.join(arr)

      summary = "Bucket " + str(k) + " removed points greater than " + "{:.5f}".format(means[k]) + " + " + str(mult)  + "*" + "{:.5f}".format(stds[k]) + ", removing " + str(numOutlier)

      if numOutlier == 1:
        summary += " point:\n"
      else:
        summary += " points:\n"

      ret.append(summary + arrStr + "\n")
  return ''.join(ret)
  
'''
Plots abs(difference between energies_of_a_method)/normMetric vs normMetric.
Returns the number of removed points, and formatted txt summary of removed points
  df - Pandas dataframe (?) in dict form
  entry1, entry2 - names of respective energy_of_a_methods
  norm - [H, C, N, O] weights to create the normalization metric
  normMetric - Name of the normalization metric (Heavy Atoms, etc.)
  stdVar - Standard deviation constant. Defaults to 10
  mols - Legacy from Francis's work. Select specific molecules
  save - Saves to path. By default, will not save anywhere
  path - file path to save plots to
'''
def genCompPlot(df, entry1, entry2, norm, normMetric, stdVar = 10, mols='all', save=False, path = ""):
  # Copy pasted from Francis's work
  if mols == 'all':
      moldata = df
  elif isinstance(mols, str):
      moldata = df.loc[df['mol'] == mols]
  else:
      mol_mask = np.logical_or.reduce([df['mol'] == mol for mol in mols])
      moldata = df.loc[mol_mask]
  e1 = moldata[entry1]
  e2 = moldata[entry2]
  names = ('nH', 'nC', 'nN', 'nO', 'nAu')
  Zs = [Z for Z in names if Z in df.columns]
  nZ = moldata[Zs]

  s1 = LinearRegression()
  s1.fit(nZ, e1 - e2)
  error = (e1 - e2) - s1.predict(nZ)
  error = error * HARTREE

  hZNum = list(nZ[names[0]])
  cZNum = list(nZ[names[1]])
  nZNum = list(nZ[names[2]])
  oZNum = list(nZ[names[3]])

  # Make weighted normMetric parameter
  totZNum = [hZNum[i]*norm[0] + cZNum[i]*norm[1] + nZNum[i]*norm[2] + oZNum[i]*norm[3] for i in range(len(hZNum))]

  molID = moldata['mol']
  confID = moldata['conf']

  # Make abs(difference between energies_of_a_method)/sqrt(normMetric) 
  ''' IMPORTANT: THIS LINE SHOULD BE MODIFIED IF YOU DONT WANT THE SQRT '''
  arr = [abs(error[i]/(totZNum[i])**0.5) for i in range(len(hZNum))]
  
  # Make the hacky dict that finds the molecule/conformation related to a point
  reverseSearch = dict()
  for i in range(len(hZNum)):
      lookup = str(arr[i])
      if lookup not in reverseSearch:
          reverseSearch[lookup] = [(molID[i], confID[i])]
      else:
          print("dupe")
          reverseSearch[lookup].append((molID[i], confID[i]))

  # Place data in buckets
  bPlots = dict()
  for i in range(len(hZNum)):
    idx = totZNum[i]
    if idx not in bPlots:
      bPlots[idx] = [arr[i]]
    else:
      bPlots[idx].append(arr[i])

  keys = sorted(list(bPlots.keys()))

  # Pruning to remove extreme outliers, also recording them 
  stds = dict()
  means = dict()
  medQ1Q3 = dict()
  bPlotOutliers = dict()
  numOutliers = 0
  for k in keys:
    (t1, t2) = np.percentile(bPlots[k], [25,75])
    t3 = np.median(bPlots[k])
    medQ1Q3[k] = (t1, t3, t2)
    means[k] = np.mean(bPlots[k])
    stds[k] = np.std(bPlots[k])
    rejectVal = stds[k]*stdVar + means[k]
    bPlotOutliers[k] = list(filter(lambda x: x > rejectVal, bPlots[k]))
    numOutliers += len(bPlotOutliers[k])
    bPlots[k] = list(filter(lambda x: x <= rejectVal, bPlots[k]))

  # Plot title, size
  fig = plt.figure(figsize=(10, 10)) 
  ax = fig.add_subplot(111)
  ax.grid(axis = 'x', linewidth=0)
  plt.title("abs(" + entry1 + "-" + entry2 + ")" + '/sqrt(' + normMetric + ")" + " vs. " + normMetric)
  plt.xlabel(normMetric)
  plt.ylabel("abs(" + entry1 + "-" + entry2 + ")" + '/sqrt(' + normMetric + ")")

  # Making the boxplots look pretty 
  flierprops = dict(marker='.', markerfacecolor='none', markersize=5,
                linestyle='none', markeredgecolor='grey')
  boxes = ax.boxplot([bPlots[k] for k in keys], flierprops = flierprops)
  plt.setp(boxes['medians'], color = 'cornflowerblue')

  # Outlier Summary
  txt = "===============================================================================\n"
  for k in keys:
    (Q1, median, Q3) = medQ1Q3[k]
    txt += "Heavy atoms #: " + str(k) + "\n"
    txt += "\tMedian: " + "{:.5f}".format(median) + "\n"
    txt += "\tQ1: " + "{:.5f}".format(Q1) + "\n"
    txt += "\tQ3: " + "{:.5f}".format(Q3) + "\n"
  if numOutliers == 0:
    txt += "No points removed.\n"
  else:
    txt += str(numOutliers) + " extreme outliers:\n" 
    txt += formatDictPrint(bPlotOutliers, means, stds, stdVar, reverseSearch)
  txt += "===============================================================================\n"

  filename = "(" + normMetric + ")" + entry1[:-7] + "--" + entry2[:-7] + ".png"
  print(filename, "generated")

  # If generating multiple plots and need to save somewhere
  if save == True:
    plt.savefig(path + filename, bbox_inches='tight')
    plt.close('all')

  return numOutliers, txt

#%%
''' Test Cell ''' 
# a,b = genCompPlot(df, "wb97x_dz.energy", "mp2_dz.energy", [0,1,1,1], "Heavy Atoms")
# print(b)

#%%
''' Bulk Running ''' 

# Helper method for filtering energy terms
def getEnergyTerms(x):
  if ".energy" in x:
    return True
  return False
  
# Written as method so it can be easily be commented to not run  
def bulkRun():
  with File(dataset_path, 'r') as dataset:
    mols = list(dataset.keys())
    mol_0 = mols[0]
    moldata_0 = list(dataset[mol_0].keys())
    eList = list(filter(getEnergyTerms, moldata_0))
    eListLen = len(eList)
    totOutliers = 0
    mxOutliers = 0

    arr = []
    stdmulty = 10
    outlierMoleSet = set()

    # Generates all pairwise comparisons between the methods (no duplicates)
    for i in range(eListLen):
      for j in range(i+1, eListLen):
        currOut, stats = genCompPlot(df, eList[i], eList[j], [0,1,1,1], "Heavy Atoms", save = True, path = "../Graphs/")
        mxOutliers = max(mxOutliers, currOut)
        totOutliers += currOut
        arr.append("\n\n-------\t\t" + eList[i] + " vs. " + eList[j] + "\t\t-------\n")
        arr.append(stats)

    # Formatting for summary    
    s = "Heavy Atoms.\n"
    s += "total number of extreme outliers: " + str(totOutliers) + "\n"
    s += "max rejected: " + str(mxOutliers) + "\n"
    s += "Rejection metric was per bucket, using mean + " + str(stdmulty) + "std.\n"
    s += ''.join(arr)

    with open("../Graphs/stats.txt", "w") as text_file:
        print(s, file=text_file)

    print("done")

# bulkRun()
# %%
