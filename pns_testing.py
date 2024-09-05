# General imports
import glob
from IPython.testing.globalipapp import get_ipython
import asyncio
import h5py
import pandas
from dask.distributed import Client
from distributed import Client
import numpy as np
import os
import sys
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
import sgmdata
from lmfit import Model
from sgmdata.load import SGMData
import sgmdata.report
from collections import OrderedDict
# Plotting function imports
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, CustomJS, BooleanFilter, LinearColorMapper, LogColorMapper, ColorBar
from bokeh.io import show

ip = get_ipython()


# GETTING DATA FILES FROM DISK * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

l = []

desired_file = input("Please input the absolute path to your hdf5 file: ")
for filename in glob.iglob(desired_file, recursive=True):
    l.append(filename)
if l.isempty():
    print("There were no files matching your description found in the specified directory.\n")
else:
    print("The following files match your input: " + str(l))


# LOADING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

# Creates a new SGMData object
sgm_data = sgmdata.load.SGMData(l, sample = "Imidazol - C")

table = None
a_dict = OrderedDict()
sample_keys = list(sgm_data.scans.keys())
for sample in sample_keys:
    for entry in sgm_data.scans[sample].__dict__:
        # print("Entry: " + str(entry))
        for section in sgm_data.scans[sample].__dict__[entry]:
            if len(sgm_data.scans[sample].__dict__[entry]) > 0:
                if type(sgm_data.scans[sample].__dict__[entry]) is not str:
                    if type(sgm_data.scans[sample].__dict__[entry][section]) is not int:
                        # print("\tSection: " + str(section))
                        for subsect in sgm_data.scans[sample].__dict__[entry][section]:
                            if "sdd" in subsect:
                                if type(sgm_data.scans[sample].__dict__[entry][section]) is dict:
                                    # print("\t\tSubsect: " + str(subsect))
                                    table = (sgm_data.scans[sample].__dict__[entry][section][subsect].__array__())


# FUNCTIONS * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def plot1d(xarr, yarr):
    """
    Sets the specifications for a graph, then shows graph. Graph represents the data samples, and the predicted data
    samples.
    Variables:
        xarr: a list of two lists containing the x values for the points we will be plotting.
        yarr: a list of two lists containing the y values for the points we will be plotting.
    """
    # A string listing the tools we will have available for our graph.
    TOOLS = 'pan, hover, box_zoom, box_select, crosshair, reset, save'
    # Specifying the appearance of our graph.
    fig = figure(
        tools=TOOLS,
        title="Plot",
        background_fill_color="white",
        background_fill_alpha=1,
        x_axis_label="x",
        y_axis_label="y",
    )
    colors = []
    # For every group of six in yarr (rounded down) and once more, add " 'purple', 'black', 'yellow', 'firebrick',
    # 'red', 'orange' " to the 'colors' list.
    for i in range(np.floor(len(yarr) / 6).astype(int) + 1):
        colors += ['purple', 'yellow', 'black', 'firebrick', 'red', 'orange']
    # Making colors variable into an iterator that can iterate through the previous version of colors.
    colors = iter(colors)
    if not isinstance(xarr, list):
        xarr = [xarr]
    if not len(xarr) == len(yarr):
        yarr = [yarr]
    # For the number of items in xarr, plot a new point on our graph.
    for i, x in enumerate(xarr):
        fig.circle(x=x, y=yarr[i], color=next(colors), legend_label="Curve" + str(i))
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    show(fig)


def lowest_variance(d_list):
    """
    From a list, calculates the lowest level of variance between 5 of the consecutive values in the list. Returns this
    level of variance and the index of the values that make up this level.
    Variables:
        d_list: the list of variances between each consecutive spot on a graph.
    """
    print("\nd_list: \n" + str(d_list))
    pos = 0
    recent_differences = []
    differences = []
    recent_vars = []
    for diff in d_list:
        if len(recent_vars) < 4:
            recent_vars.append(diff)
        elif len(recent_vars) == 4:
            recent_differences.clear()
            pos = 4
            recent_vars.append(diff)
            for var in recent_vars:
                recent_differences.append(((var - np.mean(recent_vars)) ** 2))
            differences.append(np.sum(recent_differences) / len(recent_differences))
        else:
            recent_differences.clear()
            pos = pos + 1
            recent_vars.pop(0)
            recent_vars.append(diff)
            for var in recent_vars:
                recent_differences.append(((var - np.mean(recent_vars)) ** 2))
            differences.append(np.sum(recent_differences) / len(recent_differences))
    # print("difference: \n" + str(differences))
    i = 0
    for boop in differences:
        print(str(i) + "-" + str(i + 4) + " difference is: " + str(boop))
        i += 1
    return "The lowest average variance (of the variance between 2 consecutive variances) between 5 consecutive " \
           "variances is " + str(min(differences)) + ".\nIt is reached with the variances between the " \
            "values within the range: " + str(pos - 4) + " through to position: " + str(pos) + ".\n"


# INTERPOLATING DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def interpolating_data(interp_list_param):
    """Takes the data returned from interpolate and collects the sdd values. Sorts through these sdd values and removes
     unfit values. Keeps a separate list of the indices of the fit values. Deals with nan and infinity values in the
     list of sdd values and returns it to caller as a numpy array."""
    sdd_list = []
    # Checking for items in interp_list containing the characters 'sdd.' those items are appended to sdd_list.
    for df in interp_list_param:
        sdd_list.append(df.filter(regex=("sdd.*"), axis=1).to_numpy())

    prev_mean = sdd_list[0]
    avg_list = [sdd_list[0]]
    diff_list = []
    indices = []
    for i, arr in enumerate(sdd_list[1:]):
        # *Note that arr is the array of arrays representing the each scan's sdd values and i is the count of the number of
        # times the loop has been cycled through, starting at 0.
        avg_list.append(arr)
        cur_mean = np.mean(avg_list, axis=0)
        # * Cur_mean is an array with the mean of all values in the same positions in the in arrays within avg_list. Eg, the
        # average of the item [0] of each of the lists, then the average of the item[1] of each of the lists, etc.
        # Finding the variance between the last scan and the current scan.
        diff_list.append(np.var(np.subtract(cur_mean, prev_mean)))
        if len(diff_list) > 2:
            if diff_list[-2] - diff_list[-1] < -50:
                avg_list = avg_list[:-1]
                diff_list = diff_list[:-1]
                # If the variance between the difference between our current scan's mean and the mean of the scan before it
                # is 51 or more than the variance of the difference between the previous scan's mean and the mean of the
                # scan before it, then don't include the variance of the difference between the current scan's mean and the
                # previous scan's mean in diff list, or the average of this scan in avg_list.
            else:
                indices.append(i)
                prev_mean = cur_mean
        else:
            indices.append(i)
            prev_mean = cur_mean
    # Turns diff_list into and array, then converts NaN values to 0s and deals with infinity values.
    diff_list = np.nan_to_num(np.array(diff_list))
    return diff_list, indices
