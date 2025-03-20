# timelined_array

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJostTim%2Ftimelined_array%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![PyPI - Version](https://img.shields.io/pypi/v/timelined_array)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/JostTim/timelined_array/publish.yml?label=Publishing)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/JostTim/timelined_array/test.yml?label=Testing)
![Codecov](https://img.shields.io/codecov/c/github/JostTim/timelined_array)


<!-- eventually, add ?cache-bust=1 to badges url above, and update the value of cache-bust randomly at git commits (or based on the commit name) so that we always see the latest badge produced, and not the cached badge in the browser.-->

## Overview
The TimelinedArray package provides a set of classes and utilities for working with time-indexed arrays. It extends the functionality of NumPy arrays to include time-based indexing and operations, making it easier to work with time-series data.

## Classes
### Timeline
A subclass of np.ndarray that represents a timeline associated with the array. It includes methods for creating uniformly spaced timelines and calculating time steps.

### Boundary
An enumeration that defines inclusive and exclusive boundaries for time indexing.

### TimeIndexer
A class that provides methods for converting time values to array indices and for indexing arrays based on time.

### TimeMixin
A mixin class that adds time-related methods and properties to arrays, including methods for aligning, transposing, and moving axes.

### TimePacker
A class for packing arrays with their associated timelines. Mostly usefull to plot data fast to matplotlib in case of 1D TimelinedArrays (x and y are unpacked directly from Timeline and the array, into `plt.plot` using `plt.plot(*time_arrray.pack)` )

### TimelinedArray
A subclass of np.ndarray that includes a timeline and a time dimension. It provides methods for time-based indexing and operations.

### MaskedTimelinedArray
A subclass of np.ma.MaskedArray that includes a timeline and a time dimension. It provides methods for time-based indexing and operations on masked arrays.

### Seconds
A simple class for converting seconds to array indices based on a given sampling frequency.

## Installation
To install the TimelinedArray package, simply type in your environment activated console :
```bash
pip install timelined_array
```

The package can be found on PyPI at : https://pypi.org/project/timelined_array/ 

## Usage

### Imports
```python
from timelined_array import TimelinedArray, MaskedTimelinedArray
```

### Creating a TimelinedArray
```python
import numpy as np
from timelined_array import TimelinedArray

data = np.random.rand(100, 10)
timeline = np.linspace(0, 10, 100)
timelined_array = TimelinedArray(data, timeline=timeline, time_dimension=0)
```

### Time-based indexing
Use the `itime` attribute that allows idexing over time (and time only, if you want otherdimensionnal indexing, look below)
```python
# Access data at a specific time
data_at_time = timelined_array.itime[5.0]
 ```
Here we accessed all points in time closest to time unit 5.0. So because our array was shaped (100,10) and time_dimension was 0, we are obtaining an array of 10 elements, and the type of the returned array is a normal np.ndarray (because we selected a single time point on the time_dimension, we lost the timeseries aspect of our data).

or to access a span of time :
```python
data_at_time = timelined_array.itime[5.0:9.0]
 ```
Note that the slice ``start`` is ``including`` and ``stop`` is ``excluding``, when working with ``itime``.

As such, if one wans to get 9.0 time point related data included, he may write :
```python
data_at_time = timelined_array.itime[5.0:9.0]
 ```

(Exclusion / Inclusion handling for start and stop are planned to be tuneable in the future, with a syntax close to :)
```python
data_at_time = timelined_array.itime(start="exclude")[5.0:9.0]
 ```
And exclusion settings set at the level of the array for all future usage might also be implemented. (need to be passed down to child arrays too, wich would imply some reworking of the pickling handling)

### Pickling timelinedarrays :
Timelined arrays and their maked counterpart can be pickled without issue, and the timeline and time dimension is kept, by overriding the __reduce__ and __setstate__ methods of numpy arrays.

### Mixing non-time and time-based indexing

```python
data_at_time = timelined_array[:,2:4].itime[5.0]
 ```
Here, we seleted the whole first dimension (the timed one) and only a span of 2 to 4 on the second dimension. Then, on the returned TimelinedArray, we select only the time_related data closest to point 5.0. This would leave us with an array of shape : (2,) as we lost the time dimension using itime over a single point, and we selected a span of two over the initial second dimension with [:,2:4].

Note that the order doesn't matter, as .itime returns an array over wich you can still iterate normally.
Of course you still need to pay attention to the dimension of the array that the first .itime indexing will yield.

For example, doing this removes the time_dimension, wich is first, so to select a span of the second, we should write :
```python
data_at_time = timelined_array.itime[5.0][2:4]
 ```

But if we selected a span over time, we should write :
```python
data_at_time = timelined_array.itime[5.0:9.0][:, 2:4]
 ```

### Synchronizing two arrays in time:
Is as easy as :

```python
sync_point = 5.0
sample_size = 25

sync_data_1 = timelined_array_1.itime[5.0:][:,:sample_size]
sync_data_2 = timelined_array_2.itime[5.0:][:,:sample_size]
 ```
Where sync_point is the start of you new synchronized arrays, and sample_size is the lendth they will both have on the time_dimension. Using such method, you can stack them easily (however, even if understanding that this is possible and how it works is usefull to beter deal with this package, please see the tutorial step called **Synchronizing timelined array to a new higher dimension** for an easier way to perform such stacking on the time dimension after synchronization, and some details on the caveats to avoid.)


### Performing operations reducing dimensions

```python
# Access data at a specific time
data_at_time = timelined_array.mean(axis = 0)
 ```
This will leave us with a normal np.ndarray as we loose the dimension 0 wich is our time_dimension.

```python
# Access data at a specific time
data_at_time = timelined_array.mean(axis = 1)
 ```
On the other hand, this will leave us with a TimelinedArray, but of only one dimension (the second one collapsed). The timeline didn't change as we didn't touched the time_dimension.

For now, the available collapsing/dimension ordering alterating methods are : 
- sum
- mean
- std
- var
- swapaxes
- transpose
- T
- moveaxis
- rollaxis
  
Feel free to request more ufunc support if you need it.

### Synchronizing timelined array to a new higher dimension
Say you want to create a higner dimensionnality array, with one timeline, from different arrays (a iterable of them) that all have a timeline that is containing a common part (some of them might start earlier, end later in the timeline, but they all must have at least some space in common)
This function provides the necessary help to do that. It does so by checking the first common available time in all the arrays in the iterable. Then, from that common first timepoint, it checks up to how many timepoints it can go so that all arrays have the same number of points in the end. (to stack them to a higher dimension, as the new first dimension).

Because it it's internal working, as just described above, it has two caveats : 
  - it is implied that your timelines must have the same, or almost equivalent, time step between two points in time. This is not enforced for performance reasons, so be aware of what you do and what you work with. If necessary, you might resample your data first to a common time step and then use this method. This resampling might be implemented in a method that is attached to the TimelinedArray (In a close future, not planned at all on the ``MaskedTimelinedArrays`` version of the arrays, as it would imply too complex fillding over the binary mask to choose what should be masked or not, for an ue case i don't need right now). If you want to check the average time step (calculated with a ``.mean`` of the ``.diff`` over the timeline) you have on the timeline of a given array to compare them, you can use ``my_array.timeline.step``. Getting the step_variation (calculated with a ``.std`` of the ``.diff`` over the timeline) may be implemented soon. Feel free to request the feature if you need it.
  - it checks the **first** time point available for all arrays, by performing a simple ``.min()`` on the timeline of all arrays so it implies that all timelines of the arrays in the iterable are rising. For my own purpose, it doesn't happend that arrays that have a decreasing timeline, but an option to state wether the timelines are increasing or decreasing in time might be added in the future to this function (default will be increasing). Feel free to request the feature if you need it.

```python
# Aligning arrays
aligned_array = TimelinedArray.align_from_iterable([timelined_array, another_timelined_array, yet_another_timelined_array])
```
In that example, the new higer dimension of size 3 will be the new first dimension, and the time_dimension of the newly created array will thus be shifted of +1 compared to the time_dimension of the original timelined_arrays in the iterable.

### Masking TimelinedArrays
```python
from timelined_array import MaskedTimelinedArray

masked_data = np.ma.masked_array(data, mask=data > 0.5)
masked_timelined_array = MaskedTimelinedArray(masked_data, timeline=timeline, time_dimension=0)
```

### Converting time location to index location
This code finds the closest index on the `time_dimension`, to the time point at `5.0` (can be seconds, milliseconds, whatever you prefer that the time value represents for your situation)
```python
masked_data.itime.time_to_index(5.0)
```
You can also get a bunch of indexes for a set of time points, in one go :
```python
masked_data.itime.time_to_index([5.0,8.0,12.0,4.0])
```
Note that the set of timepoints doesn't necessarily need to ordered in time in a strictly increasing or decreasing way, allowing you to easily sort an array :
```python
sorted_index = masked_data.itime.time_to_index([5.0,8.0,12.0,4.0])
sorted_masked_data = masked_data[sorted_index]
```
By doing so, the timeline is also sorted at the same time than your data. Coherence is respected. But beware that the ``step`` method of the attached timeline will no longer make sense. (It lonly does when an array is strictly increasing or decreasing in time). Because the `_step` value is cached however, it might not complain about it if you request it, and the value will not reflect the reality. Given how specific this issue is, it might not be dealt with soon.


## Misc

### License
This package is licensed under the MIT License. See the LICENSE file for more details.

### Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

### Contact
For any questions or issues, please contact the package maintainer.
