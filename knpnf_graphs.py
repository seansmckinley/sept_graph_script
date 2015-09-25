import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import statsmodels

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2




known_list, nov_list, prox_list = [], [], []

for files in glob.glob("*.csv"):
    if "Known" in files:
        known_list.append(files) 
    if "Nov" in files:
        nov_list.append(files) 
    if "Prox" in files:
        prox_list.append(files) 

known_frames, nov_frames, prox_frames = [], [], []



for known in known_list:
    known_file = pd.read_csv(known)
    class_name = known.lstrip("ThetaKnown_")
    class_name = class_name.rstrip(".csv")
    ax = known_file.plot(x="Theta-K",y="%CClass", legend=False)
    ax.set_ylim(0,100)
    plt.ylabel("percentage correct")
    title = class_name + " known"
    fig_file = class_name + "_known.png"
    plt.title(title)
    plt.savefig(fig_file)
    plt.close("all")

	
for nov in nov_list:
    nov_file = pd.read_csv(nov)
    acopy = nov_file.copy()
    old_cols = nov_file.columns
    acopy.columns = ["c1","c2","c3","c4","c5","c6","c7","c8","c9"]
    acopy.ix[acopy.c6 == 0, "c4" ] = np.nan
    acopy.ix[acopy.c7 == 0, "c5" ] = np.nan
    nov_file = acopy
    nov_file.columns = old_cols
   
    class_name = nov.lstrip("ThetaNov_")
    class_name = class_name.rstrip(".csv")
    ax = nov_file[nov_file["Theta-P"]==nov_file["Theta-P"].unique()[0]].plot(x="Theta-N",y="Clust:ClassifiedWklds", legend=False, kind="scatter")
    ax.set_ylim(0, 1.05)
    colors = ["cyan","purple","orange"]
    equal = " = "
    for idx, i in enumerate(nov_file["Theta-P"].unique()):
        nov_file[nov_file["Theta-P"]==nov_file["Theta-P"].unique()[idx]].plot(x="Theta-N",y="Clust:ClassifiedWklds", kind="scatter", ax=ax, 
                                                                              color = colors[idx], label=equal.join(["p-threshold", str(i)]) )
    plt.ylabel("cluster accuracy classified workloads")
    title = class_name + " novelty 2a"
    fig_file = class_name + "_novelty2a.png"
    plt.title(title)
    plt.savefig(fig_file)
    plt.close("all")

    ax2 = nov_file[nov_file["Theta-P"]==nov_file["Theta-P"].unique()[0]].plot(x="Theta-N",y="Cluster:PendingWklds", legend=False, kind="scatter")
    ax.set_ylim(0, 1.05)
    for idx, i in enumerate(nov_file["Theta-P"].unique()):
        nov_file[nov_file["Theta-P"]==nov_file["Theta-P"].unique()[idx]].plot(x="Theta-N",y="Cluster:PendingWklds", kind="scatter", ax=ax2, 
                                                                              color = colors[idx], label=equal.join(["p-threshold", str(i)]) )
    plt.ylabel("cluster accuracy pending workloads")
    title = class_name + " novelty 2b"
    fig_file = class_name + "_novelty2b.png"
    plt.title(title)
    plt.savefig(fig_file)
    plt.close("all")

for prox in prox_list:
    prox_file = pd.read_csv(prox)
    acopy = prox_file.copy()
    old_cols = prox_file.columns
    acopy.columns = ["c1","c2","c3","c4","c5","c6","c7","c8","c9"]
    acopy.ix[acopy.c6 == 0, "c4" ] = np.nan
    acopy.ix[acopy.c7 == 0, "c5" ] = np.nan
    prox_file = acopy
    prox_file.columns = old_cols

    class_name = prox.lstrip("ThetaProx_")
    class_name = class_name.rstrip(".csv")
    ax = prox_file[prox_file["Theta-N"]==prox_file["Theta-N"].unique()[0]].plot(x="Theta-P",y="Clust:ClassifiedWklds", legend=False, kind="scatter")
    ax.set_ylim(0, 1.1)
    colors = ["red","blue","green"]
    equal = " = "
    for idx, i in enumerate(prox_file["Theta-N"].unique()):
        prox_file[prox_file["Theta-N"]==prox_file["Theta-N"].unique()[idx]].plot(x="Theta-P",y="Clust:ClassifiedWklds", kind="scatter", ax=ax, 
                                                 color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
    plt.ylabel("cluster accuracy classified workloads")
    title = class_name + " proximity 2a"
    fig_file = class_name + "_proximity2a.png"
    plt.title(title)
    plt.savefig(fig_file)
    plt.close("all")  

    ax = prox_file[prox_file["Theta-N"]==prox_file["Theta-N"].unique()[0]].plot(x="Theta-P",y="Cluster:PendingWklds", legend=False, kind="scatter")
    ax.set_ylim(0, 1.1)
    for idx, i in enumerate(prox_file["Theta-N"].unique()):
        prox_file[prox_file["Theta-N"]==prox_file["Theta-N"].unique()[idx]].plot(x="Theta-P",y="Cluster:PendingWklds", kind="scatter", ax=ax, 
                                                 color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
    plt.ylabel("cluster accuracy pending workloads")
    title = class_name + " proximity 2b"
    fig_file = class_name + "_proximity2b.png"
    plt.title(title)
    plt.savefig(fig_file)
    plt.close("all")

for i in known_list:
    known_frames.append(pd.read_csv(i))
for i in nov_list:
    nov_frames.append(pd.read_csv(i))
for i in prox_list:
    prox_frames.append(pd.read_csv(i))

all_known = pd.concat(known_frames)
all_prox = pd.concat(prox_frames)
all_nov = pd.concat(nov_frames)
    
by_prox_index = all_prox.groupby(all_prox.index)
prox_means = by_prox_index.mean()

by_nov_index = all_nov.groupby(all_nov.index)
nov_means = by_nov_index.mean()

by_known_index = all_known.groupby(all_known.index)
known_means = by_known_index.mean()

all_prox = prox_means
all_nov = nov_means
all_known = known_means

all_p = all_prox[all_prox["Theta-N"]==all_prox["Theta-N"].unique()[0]].plot(x="Theta-P",y="Cluster:PendingWklds", legend=False,kind="scatter")
all_p.set_ylim(0, 1.1)
colors = ["red","blue","green"]
equal = " = "
for idx, i in enumerate(all_prox["Theta-N"].unique()):
    all_prox[all_prox["Theta-N"]==all_prox["Theta-N"].unique()[idx]].plot(x="Theta-P",y="Cluster:PendingWklds", kind="scatter", ax=all_p, color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
plt.ylabel("cluster accuracy pending workloads")
title ="all proximity 1b"
fig_file = "all_proximity1b.png"
plt.title(title)
plt.savefig(fig_file)

all_p2 = all_prox[all_prox["Theta-N"]==all_prox["Theta-N"].unique()[0]].plot(x="Theta-P",y="Clust:ClassifiedWklds",kind="scatter", legend=False)
all_p2.set_ylim(0, 1.1)
colors = ["red","blue","green"]
equal = " = "
for idx, i in enumerate(all_prox["Theta-N"].unique()):
    all_prox[all_prox["Theta-N"]==all_prox["Theta-N"].unique()[idx]].plot(x="Theta-P",y="Clust:ClassifiedWklds", kind="scatter", ax=all_p2, color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
plt.ylabel("cluster accuracy classified workloads")
title ="all proximity 1a"
fig_file = "all_proximity1a.png"
plt.title(title)
plt.savefig(fig_file)

all_n = all_nov[all_nov["Theta-P"]==all_nov["Theta-P"].unique()[0]].plot(x="Theta-N",y="Cluster:PendingWklds", legend=False)
all_n.set_ylim(0, 1.05)
colors = ["cyan","purple","orange"]
equal = " = "
for idx, i in enumerate(all_nov["Theta-P"].unique()):
    all_nov[all_nov["Theta-P"]==all_nov["Theta-P"].unique()[idx]].plot(x="Theta-N",y="Cluster:PendingWklds", kind="scatter", ax=all_n, color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
plt.ylabel("cluster accuracy pending workloads")
title ="all novelty 2b"
fig_file = "all_novelty2b.png"
plt.title(title)
plt.savefig(fig_file)

all_n2 = all_nov[all_nov["Theta-P"]==all_nov["Theta-P"].unique()[0]].plot(x="Theta-N",y="Clust:ClassifiedWklds", legend=False)
all_n2.set_ylim(0, 1.05)
colors = ["cyan","purple","orange"]
equal = " = "
for idx, i in enumerate(all_nov["Theta-P"].unique()):
    all_nov[all_nov["Theta-P"]==all_nov["Theta-P"].unique()[idx]].plot(x="Theta-N",y="Clust:ClassifiedWklds", kind="scatter", ax=all_n2, color = colors[idx], label=equal.join(["n-threshold", str(i)]) )
plt.ylabel("cluster accuracy classified workloads")
title ="all novelty 2a"
fig_file = "all_novelty2a.png"
plt.title(title)
plt.savefig(fig_file)

all_k = all_known.plot(x="Theta-K",y="%CClass", legend=False)
all_k.set_ylim(0,100)
plt.ylabel("percentage correct")
plt.title("all known")
plt.savefig("all_known.png")


