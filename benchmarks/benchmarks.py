import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np

from mplcairo import antialias_t
from mplcairo.base import FigureCanvasCairo


mpl.rcdefaults()


def get_axes():
    axes = Figure().subplots()
    axes.set(xticks=[], yticks=[])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def get_sample_vectors():
    return np.random.RandomState(0).random_sample((2, 10000))


def get_sample_image():
    return np.random.RandomState(0).random_sample((100, 100))


class _TimeBase:
    def setup(self):
        self.axes = get_axes()
        self.axes.figure.canvas = FigureCanvasCairo(self.axes.figure)

    def time_draw(self, *args):
        self.axes.figure.canvas.draw()

    def teardown(self, *args):
        mpl.rcdefaults()


class TimeAxes(_TimeBase):
    pass


class TimeLine(_TimeBase):
    param_names = ("antialiased", "joinstyle")
    params = (list(antialias_t.__members__.values()),
              ["miter", "round", "bevel"])

    def setup(self, antialiased, joinstyle):
        mpl.rcParams["agg.path.chunksize"] = 0
        super().setup()
        self.axes.plot(*get_sample_vectors(),
                       antialiased=antialiased, solid_joinstyle=joinstyle)


# For the marker tests, try both square and round markers, as we have a special
# code path for circles which may not be representative of general performance.


class TimeMarkers(_TimeBase):
    param_names = ("threshold", "marker")
    params = ([1 / 8, 0],
              ["o", "s"])

    def setup(self, threshold, marker):
        mpl.rcParams["path.simplify_threshold"] = threshold
        super().setup()
        self.axes.plot(*get_sample_vectors(), marker=marker)


class TimeScatterMulticolor(_TimeBase):
    param_names = ("thresold", "marker")
    params = ([1 / 8, 0],
              ["o", "s"])

    def setup(self, threshold, marker):
        mpl.rcParams["path.simplify_threshold"] = threshold
        super().setup()
        a, b = get_sample_vectors()
        self.axes.scatter(a, a, c=b, marker=marker)


class TimeScatterMultisize(_TimeBase):
    param_names = ("thresold", "marker")
    params = ([1 / 8, 0],
              ["o", "s"])

    def setup(self, threshold, marker):
        mpl.rcParams["path.simplify_threshold"] = threshold
        super().setup()
        a, b = get_sample_vectors()
        self.axes.scatter(a, a, s=100 * b ** 2, marker=marker)


class TimeImage(_TimeBase):
    def setup(self):
        super().setup()
        self.axes.imshow(get_sample_image())
