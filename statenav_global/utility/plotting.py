"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib
import sys
import os
import warnings
# Backend is selected PER-PROCESS, and the default is now the headless 'Agg' in EVERY process:
# no matplotlib window is ever popped up (savefig still works; show()/pause() become no-ops).
# To opt a single process back into interactive GUI figures, export STATENAV_MPL_BACKEND=QtAgg
# (or matplotlib's own MPLBACKEND) before starting it. force=True applies the choice even if
# another module imported pyplot first (e.g. the local planner imports pyplot before importing
# this package).
try:
    matplotlib.use(os.environ.get('STATENAV_MPL_BACKEND')
                   or os.environ.get('MPLBACKEND', 'Agg'), force=True)
except Exception:
    pass
# Silence the "non-GUI backend cannot show the figure" warnings that the viz helpers'
# repeated plt.show(block=False)/plt.pause() calls emit under a headless backend.
# Two message texts are needed: matplotlib <= 3.7 says "Matplotlib is currently using agg,
# which is a non-GUI backend...", while >= 3.8 says "FigureCanvasAgg is non-interactive, and
# thus cannot be shown". The first pattern alone silently stopped matching on the newer
# matplotlib, which is how these warnings started leaking into the planner console.
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
warnings.filterwarnings("ignore", message=".* is non-interactive, and thus cannot be shown")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.collections import LineCollection
import numpy as np
import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

# Canonical style constants — default values match globalmap.py.
# Pass per-figure overrides to MapFigure() as needed.
_FONT_PATH = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
if os.path.exists(_FONT_PATH):
    FONT_PROP = fm.FontProperties(fname=_FONT_PATH)
else:
    fm._load_fontmanager(try_read_cache=False)
    FONT_PROP = fm.FontProperties(family='serif')

FS_TICK:   int = 15
FS_LABEL:  int = 15
FS_LEGEND: int = 15
FS_TITLE:  int = 20
PLOT_DPI:  int = 1200
PLOT_FORMAT: str = 'pdf'
RC_PARAMS: dict = {
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Nimbus Roman'],
}

# Private aliases — keep existing Plotting class methods unchanged
_fp        = FONT_PROP
_FS_TICK   = FS_TICK
_FS_LABEL  = FS_LABEL
_FS_LEGEND = FS_LEGEND
_FS_TITLE  = FS_TITLE
_RC_PARAMS = RC_PARAMS


class BaseFigure:
    """
    General-purpose persistent matplotlib figure.

    Manages figure lifecycle, RC_PARAMS styling, font, colorbar, legend,
    save-once, and show — usable for any plot type (images, charts, maps).

    Instantiate once per figure number and reuse across update calls.
    Font sizes default to the canonical module-level values but can be overridden
    per-instance to allow different sizes in different callers.

    MapFigure subclasses this for gridmap-specific world-coordinate axis styling.
    """

    def __init__(self, fig_num, geometry, figsize=None,
                 fs_tick=None, fs_label=None,
                 fs_legend=None, fs_title=None,
                 font_prop=None):
        """
        Parameters
        ----------
        fig_num  : int        — matplotlib figure number (persistent across calls)
        geometry : tuple      — (x, y, w, h) window position and size in pixels
        figsize  : tuple|None — (width, height) in inches; None uses matplotlib default
        fs_tick  : int or None — tick label font size; defaults to FS_TICK
        fs_label : int or None — axis label font size; defaults to FS_LABEL
        fs_legend: int or None — legend/colorbar font size; defaults to FS_LEGEND
        fs_title : int or None — title font size; defaults to FS_TITLE
        font_prop: FontProperties or None — defaults to FONT_PROP (Times New Roman / serif)
        """
        self.fig_num   = fig_num
        self.geometry  = geometry
        self.figsize   = figsize
        self.fs_tick   = fs_tick   if fs_tick   is not None else FS_TICK
        self.fs_label  = fs_label  if fs_label  is not None else FS_LABEL
        self.fs_legend = fs_legend if fs_legend is not None else FS_LEGEND
        self.fs_title  = fs_title  if fs_title  is not None else FS_TITLE
        self.font_prop = font_prop if font_prop  is not None else FONT_PROP
        self._fig      = None

    def get_or_create(self):
        """
        Create the figure if it does not exist yet; otherwise clear it for redraw.
        Applies RC_PARAMS to matplotlib rcParams.
        Returns the figure object.
        """
        if self._fig is None or not plt.fignum_exists(self.fig_num):
            kw = {}
            if self.figsize is not None:
                kw['figsize'] = self.figsize
            self._fig = plt.figure(self.fig_num, **kw)
            # setGeometry is Qt-specific; guard against non-Qt backends silently
            try:
                manager   = self._fig.canvas.manager
                x, y, w, h = self.geometry
                manager.window.setGeometry(x, y, w, h)
            except Exception:
                pass
        else:
            self._fig.clf()
        # Explicitly make this the current figure so plt.xxx calls below land here
        plt.figure(self.fig_num)
        plt.rcParams.update(RC_PARAMS)
        return self._fig

    def apply_colorbar(self, label):
        """
        Add a styled colorbar to the current axes.
        Returns the colorbar object.
        """
        cbar = plt.colorbar()
        cbar.set_label(label, fontsize=self.fs_legend, fontproperties=self.font_prop)
        return cbar

    def apply_legend(self, loc='upper left'):
        """
        Apply font styling to the current legend.
        Returns the legend object.
        """
        legend = plt.legend(loc=loc, prop=self.font_prop, fontsize=self.fs_legend)
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(self.fs_legend)
        return legend

    def save_once(self, figure_path, name, flag):
        """
        Save the current figure to a file if flag is True.

        Parameters
        ----------
        figure_path : str  — directory to save into (created if missing)
        name        : str  — filename without extension
        flag        : bool — True triggers a save; False is a no-op

        Returns False so the caller can reset its flag:
            flag = fig.save_once(path, name, flag)
        """
        if flag:
            os.makedirs(figure_path, exist_ok=True)
            plt.savefig(
                os.path.join(figure_path, name + '.' + PLOT_FORMAT),
                format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight'
            )
        return False

    def save(self, filepath, dpi=None):
        """
        Save the current figure to filepath.

        Parameters
        ----------
        filepath : str      — full output path including filename and extension
        dpi      : int|None — resolution; defaults to PLOT_DPI (high-quality).
                              Pass a lower value (e.g. 150) for debug/preview output.
        """
        _dpi = dpi if dpi is not None else PLOT_DPI
        dirpath = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(dirpath, exist_ok=True)
        plt.savefig(filepath, dpi=_dpi, bbox_inches='tight')

    def show(self, block=False):
        """
        Show the figure.

        Parameters
        ----------
        block : bool — True for blocking show (waits until window is closed);
                       False (default) for non-blocking with a brief Qt event flush.
                       Use block=True for one-shot debug views; False for real-time loops.
        """
        plt.show(block=block)
        if not block:
            plt.pause(0.001)


class MapFigure(BaseFigure):
    """
    Extends BaseFigure with gridmap-specific world-coordinate axis styling.

    Use BaseFigure directly for non-map figures (camera images, tables, charts, etc.).
    """

    def apply_axes_style(self, map_shape, x_range, y_range,
                          xlabel="Y-axis", ylabel="X-axis",
                          grid_kwargs=None):
        """
        Apply tick positions, axis labels, grid, and font styling to the current axes.

        Parameters
        ----------
        map_shape   : (H, W) — pixel shape of the displayed map array (first two dims)
        x_range     : (x_min, x_max) — world X extent (row direction)
        y_range     : (y_min, y_max) — world Y extent (col direction)
        xlabel      : str — horizontal axis label (world Y direction)
        ylabel      : str — vertical axis label (world X direction)
        grid_kwargs : dict or None — extra kwargs forwarded to plt.grid(True, ...).
                      None uses the default dashed style; {} uses matplotlib defaults.
        """
        map_h, map_w = map_shape[:2]
        x_min, x_max = x_range
        y_min, y_max = y_range
        plt.xlabel(xlabel, fontsize=self.fs_label, fontproperties=self.font_prop)
        plt.ylabel(ylabel, fontsize=self.fs_label, fontproperties=self.font_prop)
        plt.xticks(
            np.linspace(0, map_w, int(y_max - y_min) + 1),
            np.linspace(y_max, y_min, int(y_max - y_min) + 1)
        )
        plt.yticks(
            np.linspace(0, map_h, int(x_max - x_min) + 1),
            np.linspace(x_max, x_min, int(x_max - x_min) + 1)
        )
        if grid_kwargs is None:
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        else:
            plt.grid(True, **grid_kwargs)
        plt.tick_params(axis='both', which='major', labelsize=self.fs_tick, width=2)
        for lbl in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            lbl.set_fontproperties(self.font_prop)
            lbl.set_fontsize(self.fs_tick)


class Plotting:
    def __init__(self, x_start, x_goal, task_extent=None):
        self.xI, self.xG = x_start, x_goal
        self.obs_bound = []
        self.obs_circle = []
        self.obs_rectangle = []
        self.task_extent = task_extent  # [x_min, x_max, y_min, y_max]

    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)
    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)


    def animation_graph_path(self, path, name="Nominal Plan",
                             map_extent=None,
                             graph_nodes=None, graph_node_colors=None,
                             graph_edges=None, graph_edge_colors=None):
        """
        Visualize a single nominal graph path over the A* planning-graph
        background.

        Draws the coolwarm_r planning-graph background and axis styling; the
        only overlay is the nominal path polyline plus start/goal markers.
        Uses planning figure 12.

        Parameters
        ----------
        path              : list of [x, y] — nominal graph-node polyline (world coords).
        name              : str            — figure title.
        map_extent        : (xmin, xmax, ymin, ymax) or None — world extent for ticks.
        graph_nodes       : (M, 2) [x, y]  — active graph-node world coords (None → skip).
        graph_node_colors : (M, 4) RGBA    — per-node color (None → gray).
        graph_edges       : (K, 2, 2)      — [[x0,y0],[x1,y1]] world segments (None → skip).
        graph_edge_colors : (K, 4) RGBA    — per-edge color (None → translucent gray).

        World coords are plotted as (world_y, world_x) so the graph, path, and
        axes share the same frame (x-axis = world Y, inverted; y-axis = world X).
        """
        plt.rcParams.update(_RC_PARAMS)
        self.plot_grid_interactive(name, fig_num=12, win_size=(480, 420))

        # Overlay the A* planning graph as background (RViz coolwarm_r style).
        # graph_edges: (K, 2, 2) [[x0,y0],[x1,y1]] → swap last axis to [y, x].
        if graph_edges is not None and len(graph_edges) > 0:
            segs = np.asarray(graph_edges)[:, :, ::-1]   # (K, 2, 2) as [y, x] per endpoint
            edge_c = graph_edge_colors if graph_edge_colors is not None else (0.6, 0.6, 0.6, 0.5)
            self.ax.add_collection(
                LineCollection(segs, colors=edge_c, linewidths=0.5, zorder=0)
            )
        # graph_nodes: (M, 2) [x, y] → scatter at (y, x).
        if graph_nodes is not None and len(graph_nodes) > 0:
            gn = np.asarray(graph_nodes)
            node_c = graph_node_colors if graph_node_colors is not None else (0.5, 0.5, 0.5, 1.0)
            self.ax.scatter(gn[:, 1], gn[:, 0], s=8, c=node_c, zorder=1, linewidths=0)

        # Nominal path polyline (red), drawn as (world_y, world_x) to match the
        # graph/axis convention above.
        if path and len(path) > 0:
            self.ax.plot([pt[1] for pt in path], [pt[0] for pt in path],
                         '-r', linewidth=2.0, zorder=5, label='Nominal path')

        # Start (black square) and goal (yellow star) markers from the path ends.
        if path and len(path) >= 1:
            sx, sy = path[0][0], path[0][1]
            self.ax.plot(sy, sx, 's', color='black', markersize=10, zorder=6, label='Start')
        if path and len(path) >= 2:
            gx, gy = path[-1][0], path[-1][1]
            self.ax.plot(gy, gx, '*', color='yellow', markersize=14, zorder=6, label='Goal')

        # --- Styling to match globalmap.py visualize_maps ---
        self.ax.set_title(name, fontproperties=_fp, fontsize=_FS_TITLE)
        self.ax.set_xlabel('Y-axis', fontproperties=_fp, fontsize=_FS_LABEL)
        self.ax.set_ylabel('X-axis', fontproperties=_fp, fontsize=_FS_LABEL)

        # Tick resolution: 1 tick per metre. x-axis = world Y (inverted), y-axis = world X.
        if map_extent is not None:
            xmin, xmax, ymin, ymax = map_extent
            n_xticks = max(2, int(abs(ymax - ymin) + 1))   # number of ticks along Y span
            n_yticks = max(2, int(abs(xmax - xmin) + 1))   # number of ticks along X span
            self.ax.set_xticks(np.linspace(ymin, ymax, n_xticks))
            self.ax.set_yticks(np.linspace(xmin, xmax, n_yticks))

        self.ax.tick_params(axis='both', which='major', labelsize=_FS_TICK, width=2)
        for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            lbl.set_fontproperties(_fp)
            lbl.set_fontsize(_FS_TICK)

        # Remove grid lines for the graph plot (overrides plot_grid_interactive's grid(True))
        self.ax.grid(False)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.001)

    def animation_interactive(self, nodelist, path, name, animation=False):
        self.plot_grid_interactive(name, fig_num=11)
        self.plot_visited_interactive(nodelist, animation)
        self.plot_path_interactive(path)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_grid_interactive(self, name, fig_num=11, win_size=(480, 400)):
        attr_fig = f'_ifig{fig_num}'
        attr_ax  = f'_iax{fig_num}'
        if not hasattr(self, attr_fig) or not plt.fignum_exists(fig_num):
            fig, ax = plt.subplots(num=fig_num)
            setattr(self, attr_fig, fig)
            setattr(self, attr_ax, ax)
            manager = fig.canvas.manager
            # setGeometry is Qt-specific; guard against non-GUI backends (e.g. Agg headless).
            try:
                manager.window.setGeometry(1440, 0, win_size[0], win_size[1])
            except Exception:
                pass
        else:
            fig = getattr(self, attr_fig)
            fig.clf()                      # clear entire figure (axes + colorbars) — matches visualize_maps pattern
            ax = fig.add_subplot(111)
            setattr(self, attr_ax, ax)
        self.fig = fig
        self.ax  = ax

        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (oy, ox), h, w,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (oy, ox), h, w,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True,
                )
            )


        self.ax.plot(self.xI[1], self.xI[0], "bs", linewidth=0.5, markersize=4) # blue square in the beginning
        self.ax.plot(self.xG[1], self.xG[0], "gs", linewidth=0.5, markersize=4) # green square in the end

        self.ax.set_title(name, fontsize=15)

        if self.task_extent is not None:
            x_min, x_max, y_min, y_max = self.task_extent
            self.ax.set_xlim(y_min, y_max)  # horizontal axis = Y
            self.ax.set_ylim(x_min, x_max)  # vertical axis = X

        # Invert x and y axes
        self.ax.invert_xaxis()
        # self.ax.invert_yaxis()

        # Add grid
        self.ax.grid(True)

        self.ax.set_xlabel('Y Axis')
        self.ax.set_ylabel('X Axis')
        
    def plot_visited_interactive(self, nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.get_parent():
                    self.ax.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g", linewidth=0.5) # green line thcikness add 1 
                    self.ax.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:

                self.ax.plot(node.y, node.x, "Db", markersize=1.7) # blue square size
                if node.get_parent():
                    self.ax.plot([node.get_parent().y, node.y], [node.get_parent().x, node.x], "-g", linewidth=0.5) # green line thcikness add 1


    def plot_path_interactive(self, path):
        if len(path) != 0:
            self.ax.plot([x[1] for x in path], [x[0] for x in path], "Dr", markersize=2) # red square size on the path
            self.ax.plot([x[1] for x in path], [x[0] for x in path], '-r', linewidth=1) # red line on the path
            plt.pause(0.001)

    def plot_grid(self, name):
        self.fig, ax = plt.subplots(figsize=(40,20))

        for (ox, oy, w, h) in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True,
                )
            )

        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.3
                )
            )
            
        ax.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        ax.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        ax.set_title(name)

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.get_parent():
                    plt.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.get_parent():
                    plt.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].get_parent():
                    plt.plot([V1[k].x, V1[k].get_parent().x], [V1[k].y, V1[k].get_parent().y], "-g")
            if k < len2:
                if V2[k].get_parent():
                    plt.plot([V2[k].x, V2[k].get_parent().x], [V2[k].y, V2[k].get_parent().y], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01)
        plt.show(block=False)





