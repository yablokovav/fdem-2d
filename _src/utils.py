import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter 
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def load_displacement(path_dir, barrier_type, component='zz'):
    return np.load(path_dir + barrier_type + '.npz')[component]

def get_sum_displacement_by_all_receivers(data, depth_index):
    return np.array([np.sum(data[f_i, :]) for f_i in range(data.shape[0])])
    
def vs2vp_via_puasson(vs, puasson):
    return vs*np.sqrt(2*(1-puasson)/(1-2*puasson))
    
def rho_via_gardner(vp):
    return (0.31*vp**0.25)*1e3

def prepare_referent_model(nz, nx, vs_ref, puasson, qp_ref, qs_ref, nfreq):
    vs = np.zeros((nz, nx)) + vs_ref 
    vp = vs2vp_via_puasson(vs, puasson)
    rho = rho_via_gardner(vp)
    qp = np.zeros((nfreq, nz, nx)) + qp_ref
    qs = np.zeros((nfreq, nz, nx)) + qs_ref
    return vp, vs, rho, qp, qs

def add_barrier2reference_model(h, vp, vs, rho, qp, qs, nfreq, vp_barrier, vs_barrier, rho_barrier, qp_barrier, qs_barrier, barrier_length, barrier_width, x_center_barrier, y_center_barrier):

    # Filling elastic parameters for the barrier 
    ind_top_barrier_coord = int((y_center_barrier - barrier_width/2)/h)
    ind_bottom_barrier_coord = int((y_center_barrier + barrier_width/2)/h) 
    ind_left_barrier_coord = int((x_center_barrier - barrier_length/2)/h)
    ind_right_barrier_coord = int((x_center_barrier + barrier_length/2)/h)
    
    vp[ind_top_barrier_coord : ind_bottom_barrier_coord, ind_left_barrier_coord : ind_right_barrier_coord] = vp_barrier
    vs[ind_top_barrier_coord : ind_bottom_barrier_coord, ind_left_barrier_coord : ind_right_barrier_coord] = vs_barrier
    rho[ind_top_barrier_coord : ind_bottom_barrier_coord, ind_left_barrier_coord : ind_right_barrier_coord] = rho_barrier
    
    # Preparation of Q-models for each frequency and for each type of barrier
    for freq_i in range(nfreq):
        qp[freq_i, ind_top_barrier_coord : ind_bottom_barrier_coord, ind_left_barrier_coord : ind_right_barrier_coord] = qp_barrier[freq_i]
        qs[freq_i, ind_top_barrier_coord : ind_bottom_barrier_coord, ind_left_barrier_coord : ind_right_barrier_coord] = qs_barrier[freq_i]  

    return vp, vs, rho, qp, qs

def extract_and_save_displacement(data, nz, npml, rec_indexes, nfreq, barrier_type, barrier_length, barrier_width, dir4result_save):
    # saving displacement in recievers point
    depth_index = 0
    data_zz = np.abs(np.array([data[0][f_i][depth_index, rec_indexes] for f_i in range(nfreq)]))
    #data_zx = np.array([data[1][f_i][:nz-npml, rec_indexes] for f_i in range(nfreq)])
    
    if barrier_type == "homogeneous":
        np.savez_compressed(dir4result_save + barrier_type, zz=data_zz)
        print(dir4result_save + barrier_type)
    else:
        np.savez_compressed(dir4result_save + barrier_type + f"_barrier_{barrier_width}_{barrier_length}", zz=data_zz)
        print(dir4result_save + barrier_type + f"_barrier_{barrier_width}_{barrier_length}")

def calc_velocities(E: float, ρ: float, puas: float) -> Tuple[float, float]:
    """
    Calculate P-wave and S-wave velocities from elastic parameters.
    
    This function computes the seismic wave velocities based on the 
    elastic modulus, density, and Poisson's ratio of a material.
    
    Parameters:
    -----------
    E : float
        Young's modulus (Pa) - measure of material stiffness
    ρ : float
        Density (kg/m³) - mass per unit volume
    puas : float
        Poisson's ratio (dimensionless) - ratio of transverse to axial strain
        
    Returns:
    --------
    vp : float
        P-wave (primary/compressional) velocity (m/s)
    vs : float
        S-wave (secondary/shear) velocity (m/s)
        
    Notes:
    ------
    Formulas used:
    - Shear modulus: G = E / (2 * (1 + ν))
    - Effective modulus for P-waves: E_eff = E * (1 - ν) / ((1 + ν) * (1 - 2ν))
    - P-wave velocity: vp = √(E_eff / ρ)
    - S-wave velocity: vs = √(G / ρ)
    
    Where ν (nu) is Poisson's ratio (puas parameter)
    """
    
    # Calculate shear modulus (G) from Young's modulus and Poisson's ratio
    G = E / (2 * (1 + puas))
    
    # Calculate effective modulus for P-wave propagation
    Eef = E * (1 - puas) / ((1 + puas) * (1 - 2 * puas))
    
    # Calculate P-wave velocity (compressional wave)
    vp = (Eef / ρ) ** 0.5
    
    # Calculate S-wave velocity (shear wave)
    vs = (G / ρ) ** 0.5
    
    return vp, vs

def draw_models(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    x: float,
    z: float,
    pml: float,
    signature_language: str = "eng",  # "eng" for English, "ru" for Russian
    sou: list = [],
    rec: list = [],
    sm_size: int = 200,
    rm_size: int = 150,
    clip_on: bool = False,
    figsize: Tuple[float, float] = (30, 5),
    fontsize: int = 18,
    cmap: str = 'RdYlBu'
) -> None:
    """
    Visualize 2D models of medium parameters (Vp, Vs, density) with PML regions highlighted.

    Parameters:
    ----------
    vp : np.ndarray
        2D array of P-wave velocities (m/s)
    vs : np.ndarray
        2D array of S-wave velocities (m/s)
    rho : np.ndarray
        2D array of densities (kg/m³)
    x : float
        Width of the main model domain (m)
    z : float
        Depth of the main model domain (m)
    pml : float
        Thickness of PML layer (m)
    signature_language : str, optional
        Language for labels and legends: "eng" for English, "ru" for Russian. Default: "eng"
    sou : list, optional
        List of source coordinates [x_positions, y_positions]
    rec : list, optional
        List of receiver coordinates [x_positions, y_positions]
    sm_size : int, optional
        Marker size for sources. Default: 200
    rm_size : int, optional
        Marker size for receivers. Default: 150
    clip_on : bool, optional
        Whether to clip markers at axes boundaries. Default: False
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches. Default: (30, 5)
    fontsize : int, optional
        Font size for labels. Default: 18
    cmap : str, optional
        Colormap for display. Default: 'RdYlBu'

    Returns:
    -------
    None
        Function only displays plots, does not return values.
    """

    # Validate language parameter
    if signature_language not in ["eng", "ru"]:
        raise ValueError("signature_language must be either 'eng' or 'ru'")
    
    # Language-specific labels
    labels = {
        "eng": {
            "titles": ["Vp (m/s)", "Vs (m/s)", "Density (kg/m³)"],
            "sources": "Sources",
            "receivers": "Receivers",
            "y_label": "Depth (m)",
            "x_label": "Distance (m)"
        },
        "ru": {
            "titles": ["Vp (м/с)", "Vs (м/с)", "Плотность (кг/м³)"],
            "sources": "Источники",
            "receivers": "Приемники",
            "y_label": "Глубина (м)",
            "x_label": "Расстояние (м)"
        }
    }
    
    # Get labels for selected language
    lang_labels = labels[signature_language]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        facecolor='none',
        constrained_layout=True  # Automatic padding adjustment
    )

    # Parameters for each subplot
    titles = lang_labels["titles"]
    data_arrays = [vp, vs, rho]
    extent = [0, x + 2*pml, z + pml, 0]  # Display boundaries [xmin, xmax, ymin, ymax]

    # Optimization: precompute PML coordinates
    pml_left = pml
    pml_right = x + pml
    pml_bottom = z

    for ax, data, title in zip(axes, data_arrays, titles):
        # Display data
        im = ax.imshow(
            data,
            extent=extent,
            interpolation='none',
            cmap=cmap,
            aspect='auto'  # Automatic aspect ratio adjustment
        )

        # Draw PML boundaries
        ax.plot([pml_left, pml_left], [0, pml_bottom], 
                color='gray', linestyle='--', linewidth=1)
        ax.plot([pml_right, pml_right], [0, pml_bottom],
                color='gray', linestyle='--', linewidth=1)
        ax.plot([pml_left, pml_right], [pml_bottom, pml_bottom],
                color='gray', linestyle='--', linewidth=1)

        # PML region labels (remain in English as technical terms)
        ax.text(pml_left/2, pml_bottom/2, 'PML',
               color='gray', ha='center', va='center',
               rotation=90, fontsize=fontsize)
        ax.text(pml_right + pml/2, pml_bottom/2, 'PML',
               color='gray', ha='center', va='center',
               rotation=270, fontsize=fontsize)
        ax.text((x + 2*pml)/2, pml_bottom + pml/2, 'PML',
               color='gray', ha='center', va='center',
               rotation=0, fontsize=fontsize)

        # Plot sources (black inverted triangles)
        if sou:
            ax.scatter(sou[0], sou[1], 
                       marker='v', 
                       color='black', 
                       s=sm_size,
                       label=lang_labels["sources"],
                       edgecolors='black',
                       linewidths=1.5,
                       zorder=3,  # Top layer
                       clip_on=clip_on)

        # Plot receivers (red triangles)
        if rec:
            ax.scatter(rec[0], rec[1], 
                       marker='^', 
                       color='red', 
                       s=rm_size,
                       label=lang_labels["receivers"],
                       edgecolors='black',
                       linewidths=1.5,
                       zorder=3,  # Top layer
                       clip_on=clip_on)

        # Add legend if sources or receivers are present
        if sou or rec:
            ax.legend(fontsize=fontsize, loc='lower right')

        # Formatting settings
        ax.set_title(title, fontsize=fontsize, pad=20)
        ax.set_ylabel(lang_labels["y_label"], fontsize=fontsize)
        ax.set_xlabel(lang_labels["x_label"], fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=fontsize)

    plt.show()

def draw_real_snapshots(
    data_yy: np.ndarray,
    x: float,
    z: float,
    pml: float,
    signature_language: str = "eng",  # "eng" for English, "ru" for Russian
    sou: list = [],
    rec: list = [],
    sm_size: int = 200,
    rm_size: int = 150,
    clip_on: bool = False,
    scale: float = 0.5,
    figsize: Tuple[float, float] = (15, 5),
    fontsize: int = 20,
    cmap: str = 'RdYlBu',
    title: Optional[str] = None,
    figname: Optional[str] = None,
    cbarfigname: Optional[str] = None,
    ylabelflag: bool = True,
    xlabelflag: bool = True,
    barrier_xy: list = [],
    barrier_wh: list = [],
    barrier_xy_2: list = [],
    barrier_wh_2: list = []
) -> None:
    """
    Visualize the real part of wavefield snapshot for the vertical component.
    Colorbar is displayed in a separate figure with a label on the side.

    Parameters:
    ----------
    data_yy : np.ndarray
        2D array of vertical component wavefield data
    x : float
        Width of the main model domain (m)
    z : float
        Depth of the main model domain (m)
    pml : float
        Thickness of PML layer (m)
    signature_language : str, optional
        Language for labels and legends: "eng" for English, "ru" for Russian. Default: "eng"
    sou : list, optional
        Source coordinates [x_positions, y_positions] (default: [])
    rec : list, optional
        Receiver coordinates [x_positions, y_positions] (default: [])
    sm_size : int, optional
        Source marker size (default: 200)
    rm_size : int, optional
        Receiver marker size (default: 150)
    clip_on : bool, optional
        Clip markers at axes boundaries (default: False)
    scale : float, optional
        Scale for colorbar range (default: 0.5)
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches (default: (15, 5))
    fontsize : int, optional
        Font size for labels (default: 20)
    cmap : str, optional
        Colormap (default: 'RdYlBu')
    title : str, optional
        Plot title. If None, uses language-specific default
    figname : str, optional
        Filename for saving main plot (default: None)
    cbarfigname : str, optional
        Filename for saving colorbar (default: None)
    ylabelflag : bool, optional
        Whether to show y-axis label (default: True)
    xlabelflag : bool, optional
        Whether to show x-axis label (default: True)
    barrier_xy : list, optional
        Barrier center coordinates [x, y] (default: [])
    barrier_wh : list, optional
        Barrier width and height [width, height] (default: [])
    barrier_xy_2 : list, optional
        Second barrier center coordinates [x, y] (default: [])
    barrier_wh_2 : list, optional
        Second barrier width and height [width, height] (default: [])

    Returns:
    -------
    None
        Function displays main plot and colorbar in separate figures.
    """

    # Validate language parameter
    if signature_language not in ["eng", "ru"]:
        raise ValueError("signature_language must be either 'eng' or 'ru'")
    
    # Language-specific labels
    labels = {
        "eng": {
            "default_title": "Vertical component wavefield",
            "sources": "Sources",
            "receivers": "Receivers",
            "y_label": "Depth (m)",
            "x_label": "Distance (m)",
            "cbar_label": "Wavefield amplitude (rel. units)"
        },
        "ru": {
            "default_title": "Вертикальная компонента волнового поля",
            "sources": "Источники",
            "receivers": "Приемники",
            "y_label": "Глубина (м)",
            "x_label": "Расстояние (м)",
            "cbar_label": "Амплитуда волнового поля (отн. ед.)"
        }
    }
    
    # Get labels for selected language
    lang_labels = labels[signature_language]

    # Extract real part of the data
    real_yy = np.real(data_yy)

    # Default settings
    if title is None:
        title = lang_labels["default_title"]

    # Precompute coordinates
    extent = [0, x + 2*pml, z + pml, 0]
    pml_left = pml
    pml_right = x + pml
    pml_bottom = z

    # Create main figure
    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor='none',
        constrained_layout=True
    )

    # Display data
    im = ax.imshow(
        real_yy,
        extent=extent,
        interpolation='none',
        cmap=cmap,
        vmin=-scale,
        vmax=scale,
        aspect='equal'
    )

    # Draw barrier
    if len(barrier_xy) != 0:
        center_x, center_y = barrier_xy[0], barrier_xy[1]
        width, height = barrier_wh[0], barrier_wh[1]
        ax.plot([center_x - width/2, center_x - width/2], [center_y - height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x + width/2, center_x + width/2], [center_y - height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x - width/2, center_x + width/2], [center_y - height/2, center_y - height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x - width/2, center_x + width/2], [center_y + height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)

    # Draw second barrier
    if len(barrier_xy_2) != 0:
        center_x, center_y = barrier_xy_2[0], barrier_xy_2[1]
        width, height = barrier_wh_2[0], barrier_wh_2[1]
        ax.plot([center_x - width/2, center_x - width/2], [center_y - height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x + width/2, center_x + width/2], [center_y - height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x - width/2, center_x + width/2], [center_y - height/2, center_y - height/2], 
                color='k', linestyle='--', linewidth=1)
        ax.plot([center_x - width/2, center_x + width/2], [center_y + height/2, center_y + height/2], 
                color='k', linestyle='--', linewidth=1)
        
    
    # Draw PML boundaries
    ax.plot([pml_left, pml_left], [0, pml_bottom],
            color='gray', linestyle='--', linewidth=1)
    ax.plot([pml_right, pml_right], [0, pml_bottom],
            color='gray', linestyle='--', linewidth=1)
    ax.plot([pml_left, pml_right], [pml_bottom, pml_bottom],
            color='gray', linestyle='--', linewidth=1)

    # PML labels (remain in English as they are technical terms)
    ax.text(pml_left/2, pml_bottom/2, 'PML',
           color='gray', ha='center', va='center',
           rotation=90, fontsize=fontsize-4)
    ax.text(pml_right + pml/2, pml_bottom/2, 'PML',
           color='gray', ha='center', va='center',
           rotation=270, fontsize=fontsize-4)
    ax.text((x + 2*pml)/2, pml_bottom + pml/2, 'PML',
           color='gray', ha='center', va='center',
           fontsize=fontsize-4)

    # Draw sources and receivers
    if sou:
        ax.scatter(sou[0], sou[1], 
                   marker='v', 
                   color='black', 
                   s=sm_size,
                   label=lang_labels["sources"],
                   edgecolors='black',
                   linewidths=1.5,
                   zorder=3,
                   clip_on=clip_on)

    if rec:
        ax.scatter(rec[0], rec[1], 
                   marker='^', 
                   color='red', 
                   s=rm_size,
                   label=lang_labels["receivers"],
                   edgecolors='black',
                   linewidths=1.5,
                   zorder=3,
                   clip_on=clip_on)

    if sou or rec:
        ax.legend(fontsize=fontsize-4, loc='lower right')
        
    # Formatting settings
    ax.set_title(title, fontsize=fontsize, pad=15)
    if ylabelflag:
        ax.set_ylabel(lang_labels["y_label"], fontsize=fontsize)
    if xlabelflag:
        ax.set_xlabel(lang_labels["x_label"], fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)

    # Save main figure
    if figname:
        fig.savefig(figname, dpi=300, bbox_inches="tight")
    
    plt.show()

    # Create separate figure for colorbar using the same image data
    # Create a new ScalarMappable with the same normalization and colormap    
    fig_cbar = plt.figure(figsize=(figsize[0]*10, 0.25))
    ax_cbar = fig_cbar.add_axes([0.1, 0.1, 0.2, 0.8])
    
    # Create a new mappable with the same normalization and colormap
    norm = Normalize(vmin=-scale, vmax=scale)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # Empty array since we're not using actual data
    
    cbar = fig_cbar.colorbar(mappable, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(lang_labels["cbar_label"], fontsize=fontsize, labelpad=15)
    cbar.formatter = ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    
    if cbarfigname:
        fig_cbar.savefig(cbarfigname, dpi=300, bbox_inches="tight")
    plt.show()

def plot_wavefield_components(
    fd: list,
    eigf: list,
    depth: np.ndarray,
    signature_language: str = "eng",  # "eng" for English, "ru" for Russian
    figsize: Tuple[float, float] = (7, 7),
    fontsize: int = 20,
    linewidth: int = 2,
    colors: Tuple[str, str] = ('darkblue', 'darkred'),
    linestyles: Tuple[str, str] = ('--', '-'),
    figfilename: str = "",
) -> None:
    """
    Plot comparison of wavefield components from finite difference and eigenfunction methods.
    
    This function creates a depth profile comparison of vertical (UZ) and horizontal (UX)
    wavefield components computed using different numerical methods.
    
    Parameters:
    -----------
    fd : list
        List containing [UZ_fd, UX_fd] - wavefield components from finite difference method
    eigf : list
        List containing [UZ_eigf, UX_eigf] - wavefield components from eigenfunction method
    depth : np.ndarray
        Depth array (y-axis values), typically normalized by Rayleigh wavelength
    signature_language : str, optional
        Language for labels and legends: "eng" for English, "ru" for Russian. Default: "eng"
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches. Default: (7, 7)
    fontsize : int, optional
        Font size for labels and ticks. Default: 20
    linewidth : int, optional
        Line width for plots. Default: 2
    colors : Tuple[str, str], optional
        Colors for UZ and UX components. Default: ('darkblue', 'darkred')
    linestyles : Tuple[str, str], optional
        Line styles for finite difference and eigenfunction methods. Default: ('--', '-')
    figfilename : str, optional
        Filename to save the figure. If empty string, figure is not saved. Default: ""
    
    Returns:
    --------
    None
        Function displays the plot and optionally saves it to file.
    """
    
    # Validate language parameter
    if signature_language not in ["eng", "ru"]:
        raise ValueError("signature_language must be either 'eng' or 'ru'")
    
    # Language-specific labels
    labels = {
        "eng": {
            "uz_fd": r"$U_Z$ (FDM)",
            "ux_fd": r"$U_X$ (FDM)",
            "uz_eigf": r"$U_Z$ (Eigenfunction)",
            "ux_eigf": r"$U_X$ (Eigenfunction)",
            "y_label": r"Depth / $\lambda_R$ (rel. units)",
            "x_label": r"Normalized amplitude (rel. units)"
        },
        "ru": {
            "uz_fd": r"$U_Z$ (МКР)",
            "ux_fd": r"$U_X$ (МКР)",
            "uz_eigf": r"$U_Z$ (disba)",
            "ux_eigf": r"$U_X$ (disba)",
            "y_label": r"Глубина / $\lambda_R$ (отн. ед.)",
            "x_label": r"Нормированная амплитуда (отн. ед.)"
        }
    }
    
    # Get labels for selected language
    lang_labels = labels[signature_language]
    
    # Configure matplotlib settings
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['font.size'] = fontsize
    
    # Create figure and axis
    fig, ax = plt.subplots(
        figsize=figsize,
        facecolor='none',
        constrained_layout=True
    )

    # Plot wavefield components
    ax.plot(fd[0], depth, color=colors[0], linestyle=linestyles[1], 
            linewidth=linewidth, label=lang_labels["uz_fd"])
    ax.plot(fd[1], depth, color=colors[1], linestyle=linestyles[1],
            linewidth=linewidth, label=lang_labels["ux_fd"])
    
    ax.plot(eigf[0], depth, color=colors[0], linestyle=linestyles[0],
            linewidth=linewidth, label=lang_labels["uz_eigf"])
    ax.plot(eigf[1], depth, color=colors[1], linestyle=linestyles[0],
            linewidth=linewidth, label=lang_labels["ux_eigf"])

    # Zero reference line
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # Custom formatter for axis labels
    class CustomFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            if abs(x) < 1e-12:  # Handle zero values
                return '0'
            return super().__call__(x, pos)
    
    formatter = CustomFormatter()
    formatter.set_powerlimits((-3, 3))  # Automatic scientific notation outside this range
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
    # Axis configuration
    ax.set_ylim([max(depth), min(depth)])  # Invert y-axis for depth profile
    ax.set_ylabel(lang_labels["y_label"], fontsize=fontsize)
    ax.set_xlabel(lang_labels["x_label"], fontsize=fontsize)
    
    # Legend
    ax.legend(fontsize=fontsize, loc='lower right', fancybox=True, framealpha=0.1)
    
    # Grid and styling
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Save figure if filename provided
    if figfilename != "":
        fig.savefig(figfilename, dpi=300, bbox_inches="tight", transparent=True)
    
    plt.show()


def plot_legend(legend_elements, labels, fontsize=20, figsize=(12, 0.5), figtitle=""):
    fig_legend = plt.figure(figsize=figsize)
    fig_legend.legend(handles=legend_elements, 
                    labels=labels,
                    fontsize=fontsize-2, 
                    loc='center', 
                    ncol=len(labels))
    fig_legend.tight_layout()
    plt.axis('off')
    if figtitle != "":
        fig_legend.savefig(figtitle, dpi=300, bbox_inches = "tight")
    plt.show()
    plt.close(fig_legend)

def plot_efficiency_for_all_barrier_configurations(path_dir, hom_name, bar_type, frequencies, bar_length_all,  bar_width_all, figsize=(9, 5), fontsize=20, figtitle="", ifig=1):

    f_picked, w_picked, l_picked = 100, 0.6, 3
    
    zz_homogen_all = load_displacement(path_dir, hom_name)
    zz_homogen = get_sum_displacement_by_all_receivers(zz_homogen_all, 0)
    ratio_all, ratio_picked = [], []
    for bar_l in bar_length_all:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        legend_elements = []
        for bar_w in bar_width_all:
            zz_all = load_displacement(path_dir, bar_type + f"_barrier_{bar_w}_{bar_l}")
            zz = get_sum_displacement_by_all_receivers(zz_all, 0)
            ratio = zz / zz_homogen
            line, = ax.plot(frequencies, ratio, linestyle='--', linewidth=2.5) 
            legend_elements.append(line)
            ratio_all.append(ratio)
            if bar_w == w_picked and bar_l == l_picked:
                ratio_picked.append(ratio[frequencies == f_picked])

                
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.grid(True, which='both')
        ax.set_xlim([0, 250])
        ax.set_ylim([1e-4, 1e1])
        ax.set_ylabel(r"$A^{барьер} / A^{референтная}$ (отн. ед.)", fontsize=fontsize)
        ax.set_xlabel(r"$f$ (Гц)", fontsize=fontsize)
    
        ax2 = ax.twiny()
        ax2.set_xscale("linear")
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.linspace(1, 250, 7))
        ax2.set_xticklabels([f"{round(val, 2)}" for val in 100 / np.linspace(1, 400, 7)])
        ax2.set_xlabel(r"$\lambda$ (м)", fontsize=fontsize)
        ax2.tick_params(axis='x', labelsize=fontsize)

        if figtitle != "":
            fig.savefig(figtitle+str(ifig), dpi=300, bbox_inches = "tight")
        ifig+=1
        
        plt.show()
    ratio_all = np.array(ratio_all)
    print(f"Median ratio for {bar_type} on all frequencies and all barrier configurations is {np.round(np.median(ratio_all), 3)}")
    print(f"Ratio for {bar_type} on {f_picked} frequency with {w_picked} width and {l_picked} length barrier is {np.squeeze(np.round(ratio_picked, 3))}")

    return legend_elements, ifig