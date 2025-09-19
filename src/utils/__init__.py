import matplotlib.pyplot as plt

# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fi
# vethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seabo
# rn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-pa
# per', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-w
# hitegrid', 'tableau-colorblind10']

plt.style.use("_mpl-gallery")
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "axes.labelcolor": "black",
        "figure.facecolor": "white",
        "figure.figsize": (7, 3),
        "grid.color": "black",
        "patch.edgecolor": "black",
        "patch.force_edgecolor": True,
        "text.color": "black",
        "xtick.color": "dimgray",
        "ytick.color": "dimgray",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.format": "png",
        "grid.linestyle": "-",
        "grid.alpha": 0.1,
        "font.size": 14,
    }
)
