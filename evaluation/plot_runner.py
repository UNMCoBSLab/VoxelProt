#!/usr/bin/env python
# coding: utf-8

# In[1]:


from VoxelProt.evaluation.eval_helpers import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib import colors
def plot_multiple_curves(
    dcc_lists,
    labels,
    title,
    save_path=None,
    thresholds=np.linspace(0, 10, 100),
    colors=None,
    markers=None,
    reference_line=4.0,
):
    
    plt.figure(figsize=(7, 5))
    
    # defaults
    if colors is None:
        colors = plt.cm.tab10.colors
    if markers is None:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    
    for i, (dcc_list, lbl) in enumerate(zip(dcc_lists, labels)):
        dcc_array = np.array(dcc_list)
        dcc_array = dcc_array[~np.isnan(dcc_array)]  # drop NaNs
        
        # compute cumulative success rate
        cumulative = [np.mean(dcc_array <= t) for t in thresholds]
        
        # plot
        plt.plot(
            thresholds, cumulative,
            lw=2, 
            marker=markers[i % len(markers)], 
            markevery=max(1, len(thresholds)//20),  # fewer markers for clarity
            markersize=6,
            color=colors[i % len(colors)],
            label=lbl,
        )
    
    # reference line (optional)
    if reference_line is not None:
        plt.axvline(reference_line, color="gray", linestyle="--", alpha=0.7)
        plt.text(reference_line+0.1, 0.95, f"{reference_line} Å", 
                 rotation=90, va="top", ha="left", fontsize=10, color="gray")
    
    # labels and aesthetics
    plt.xlabel("Distance to the binding site ($\mathrm{\AA}$)", fontsize=13)
    plt.ylabel("Success rate (%)", fontsize=13)
    plt.title(title, fontsize=14, weight="bold")
    plt.ylim(0, 1.05)
    plt.xlim(thresholds[0], thresholds[-1])
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # legend outside for clarity
    plt.legend(loc="upper left", fontsize=11, frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()


def scatter_with_zoom(
    dca, dcc, dvo,title,inset_title,
    zoom=(0, 4, 0, 4), figsize=(7, 5), s=10, alpha=0.8,
    inset_size="40%", inset_loc="upper left", borderpad=1.2,
    cmap="viridis", draw_box=True,
    x_label="DCA ($\\mathrm{\\AA}$)",y_label="DCC ($\\mathrm{\\AA}$)", color_label="DVO",
    label_size=12, title_size=16,
    vmin=None, vmax=None, norm=None, robust=None,
    cbar_orientation="vertical", cbar_ticks=None, cbar_fmt="%.2f",
    cbar_fraction=0.06, cbar_pad=0.02, cbar_shrink=0.9,
    # inset labels
    show_inset_axes_labels=True, inset_label_size=8,
    # NEW: fine control of inset position (axes coords; default no shift)
    inset_offset=(0.0, 0.0),
):
    # --- prep data ---
    dca = np.asarray(dca, float); dcc = np.asarray(dcc, float); dvo = np.asarray(dvo, float)
    if not (len(dca) == len(dcc) == len(dvo)):
        raise ValueError("dca, dcc, dvo must have same length")
    m = ~np.isnan(dca) & ~np.isnan(dcc) & ~np.isnan(dvo)
    dca, dcc, dvo = dca[m], dcc[m], dvo[m]

    # --- decide scaling ---
    if norm is None:
        if vmin is None or vmax is None:
            if robust is not None and dvo.size:
                lo, hi = np.percentile(dvo, robust[0]), np.percentile(dvo, robust[1])
                if not np.isfinite(lo): lo = np.nanmin(dvo)
                if not np.isfinite(hi): hi = np.nanmax(dvo)
                if hi <= lo:
                    eps = max(1e-12, abs(lo) * 1e-6); hi = lo + eps
                vmin = lo if vmin is None else vmin
                vmax = hi if vmax is None else vmax
        if vmin is None: vmin = np.nanmin(dvo) if dvo.size else 0.0
        if vmax is None: vmax = np.nanmax(dvo) if dvo.size else 1.0
        if vmax <= vmin:
            eps = max(1e-12, abs(vmin) * 1e-6); vmax = vmin + eps
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin = vmax = None

    # --- figure ---
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sc = ax.scatter(dca, dcc, c=dvo, s=s, alpha=alpha, cmap=cmap, norm=norm)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)

    # colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation=cbar_orientation,
                        fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)
    if cbar_ticks is not None:
        v0, v1 = float(norm.vmin), float(norm.vmax)
        ticks = np.asarray(cbar_ticks, float)
        ticks = ticks[(ticks >= v0) & (ticks <= v1)]
        cbar.set_ticks(ticks if ticks.size >= 2 else np.linspace(v0, v1, 6))
    else:
        cbar.locator = MaxNLocator(nbins=6); cbar.update_ticks()
    cbar.formatter = FormatStrFormatter(cbar_fmt); cbar.update_ticks()
    cbar.set_label(color_label, fontsize=label_size)

    # inset
    x0, x1, y0, y1 = zoom
    if draw_box:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, lw=1))

    base_anchor = {
        "upper left":  (0.0, 1.0),
        "upper right": (1.0, 1.0),
        "lower left":  (0.0, 0.0),
        "lower right": (1.0, 0.0),
    }.get(inset_loc, (0.0, 1.0))

    bx = base_anchor[0] + inset_offset[0]
    by = base_anchor[1] + inset_offset[1]

    axins = inset_axes(
        ax, width=inset_size, height=inset_size,
        loc=inset_loc,
        bbox_to_anchor=(bx, by, 1, 1),    # keep width=1, height=1
        bbox_transform=ax.transAxes,
        borderpad=borderpad
    )



    axins.scatter(dca, dcc, c=dvo, s=s, alpha=alpha, cmap=cmap, norm=norm)
    axins.set_xlim(x0, x1); axins.set_ylim(y0, y1)

    # ticks/labels in inset
    axins.set_xticks([0, 1, 2, 3, 4]); axins.set_yticks([0, 1, 2, 3, 4])
    axins.tick_params(axis='both', labelsize=inset_label_size, length=2, pad=1)
    if show_inset_axes_labels:
        axins.set_xlabel("DCA", fontsize=inset_label_size, labelpad=1)
        axins.set_ylabel("DCC", fontsize=inset_label_size, labelpad=1)

    axins.set_title(inset_title, fontsize=10)
    return fig, ax, axins
def cal_sta(dcc,dca,dvo,dcc_t = 4 , dca_t = 4):
    dvo_rv = []
    num = 0
    for each in range(len(dcc)):
        if dcc[each]<dcc_t and dca[each]<dca_t:
            num=num+1
            dvo_rv.append(dvo[each])
    mean = np.mean(dvo_rv)
    std = np.std(dvo_rv, ddof=1)   
    return num, mean, std


# In[7]:


path = "/home/jingbo/HOLO4Kresults/"
par = "dcc"
group = "B"
a_e ="excluded"
for fold_n in range(0,1):
    for et in [0,2]:
        csv_path_voxelprot = f'{path}voxelprot/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_fpocket =f'{path}fpocket/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_p2rank =f'{path}p2rank/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_kalasanty =f'{path}kalasanty/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_deepsurf =f'{path}deepsurf/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        
        dcc_voxelprot = read_dcc_list(csv_path_voxelprot)
        dcc_fpocket = read_dcc_list(csv_path_fpocket)
        dcc_p2rank= read_dcc_list(csv_path_p2rank)
        dcc_kalasanty = read_dcc_list(csv_path_kalasanty)
        dcc_deepsurf = read_dcc_list(csv_path_deepsurf)
        if et ==0:
            plot_multiple_curves(
                [dcc_voxelprot, dcc_fpocket, dcc_p2rank,dcc_kalasanty,dcc_deepsurf],
                ['VoxelProt', 'Fpocket', 'P2Rank','Kalasanty','Deepsurf'],
                title=f'Group {group} of HOLO4K (Top-n)',
                save_path=f"/home/jingbo/{par.upper()} HOLO4K-{a_e} n+{et}.png"
            )

        else:
            plot_multiple_curves(
                [dcc_voxelprot, dcc_fpocket, dcc_p2rank,dcc_kalasanty,dcc_deepsurf],
                ['VoxelProt', 'Fpocket', 'P2Rank','Kalasanty','Deepsurf'],
                title=f'Group {group} of HOLO4K (Top-(n+2))',
                save_path=f"/home/jingbo/{par.upper()} HOLO4K-{a_e} n+{et}.png"
            )            


# In[44]:


path =  "/home/jingbo/HOLO4Kresults/deepsurf/"
dataset = "HOLO4K"
group= "A"
g = "all"
method = "Deepsurf"
for et in [0,2]:
    path_dcc = f"{path}{g}/dcc_fold0_extraTop{et}.csv"
    path_dca = f"{path}{g}/dca_fold0_extraTop{et}.csv"
    path_dvo = f"{path}{g}/dvo_fold0_extraTop{et}.csv"
    
    dcc = read_dcc_list(path_dcc)
    dca = read_dcc_list(path_dca)
    dvo = read_dcc_list(path_dvo)

    num, mean, std = cal_sta(dcc,dca,dvo,dcc_t = 4 , dca_t = 4)
    if et == 0:
        fig, ax, axins = scatter_with_zoom(
            dca, dcc, dvo,
            zoom=(0,4,0,4),
            color_label="DVO",
            vmin=0.0, vmax=0.75,
            cbar_ticks=np.linspace(0, 0.75, 1),   
            cbar_fmt="%.2f", cbar_fraction=0.04, cbar_shrink=0.8,
            inset_size="40%",
            inset_loc="upper left",
            inset_offset=(0.05, -1.05),
            title=f'{method}',
            inset_title =f'Success rate: {num/len(dcc)*100:.2f}% ({mean*100:.2f} $\pm$ {std*100:.2f}%)'
        )
        fig.savefig(f"{method} on Group {group} of {dataset} (Top-n))", dpi=600, bbox_inches="tight")
    else:
        fig, ax, axins = scatter_with_zoom(
            dca, dcc, dvo,
            zoom=(0,4,0,4),
            color_label="DVO",
            vmin=0.0, vmax=0.75,
            cbar_ticks=np.linspace(0, 0.75, 1),   
            cbar_fmt="%.2f", cbar_fraction=0.04, cbar_shrink=0.8,
            inset_size="40%",
            inset_loc="upper left",
            inset_offset=(0.05, -1.05),
            title=f'{method}',
            inset_title = f'Success rate: {num/len(dcc)*100:.2f}% ({mean*100:.2f} $\pm$ {std*100:.2f}%)'
        )
        fig.savefig(f"{method} on Group {group} of {dataset} (Top-(n+2)))", dpi=600, bbox_inches="tight")
    plt.show()


methods = ["voxelprot", "fpocket", "p2rank", "kalasanty", "deepsurf"]
path="/home/jingbo/HOLO4Kresults/"
dataset = "HOLO4K"
a_e = "excluded"
group= "B"
fold_n = 0
for et in [0, 2]:
    pars = ['dcc','dca','dvo']
    for par in pars:
        csv_path_voxelprot = f'{path}voxelprot/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_fpocket   = f'{path}fpocket/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_p2rank    = f'{path}p2rank/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_kalasanty = f'{path}kalasanty/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
        csv_path_deepsurf  = f'{path}deepsurf/{a_e}/{par}_fold{fold_n}_extraTop{et}.csv'
    
        vars()[f"{par}_voxelprot"] = read_dcc_list(csv_path_voxelprot)
        vars()[f"{par}_fpocket"]   = read_dcc_list(csv_path_fpocket)
        vars()[f"{par}_p2rank"]    = read_dcc_list(csv_path_p2rank)
        vars()[f"{par}_kalasanty"] = read_dcc_list(csv_path_kalasanty)
        vars()[f"{par}_deepsurf"]  = read_dcc_list(csv_path_deepsurf)


    method_style = {
        "voxelprot":  {"color": "tab:blue",   "marker": "o"},
        "fpocket":    {"color": "tab:orange", "marker": "s"},
        "p2rank":     {"color": "tab:green",  "marker": "D"},
        "kalasanty":  {"color": "tab:red",    "marker": "^"},
        "deepsurf":   {"color": "tab:purple", "marker": "v"},
    }

    plt.figure(figsize=(7, 6))
    for method in methods:
        dcc_arr = np.asarray(vars()[f"dcc_{method}"])
        dca_arr = np.asarray(vars()[f"dca_{method}"])
        dvo_arr = np.asarray(vars()[f"dvo_{method}"])

        
        mask = (dcc_arr < 4) & np.isfinite(dca_arr) & np.isfinite(dvo_arr)

        plt.scatter(
            dca_arr[mask],
            dvo_arr[mask],
            label=method,
            s=8,
            alpha=0.8,
            linewidths=0.5,
            edgecolors='none',
            **method_style[method]
        )

    plt.xlabel(r"DCA ($\AA$)")
    plt.ylabel(r"DVO")
    if et == 0:
        
        plt.title(f"Group {group} of {dataset} (Top-n)")
    else:
        plt.title(f"Group {group} of {dataset} (Top-(n+2))")
        
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"dca_dvo_{group}_{dataset}_extraTop{et}.png", dpi=300)
    plt.show()
         
