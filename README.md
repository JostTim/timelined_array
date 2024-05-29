# timelined_array

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
from timelined_array import TimelinedArray, MaskedTimelinedArray, Boundary, Timeline
```

### Creating a TimelinedArray
```python
import numpy as np
from timelined_array import TimelinedArray

data = np.random.rand(100, 10)
timeline = np.linspace(0, 10, 100)
timelined_array = TimelinedArray(data, timeline=timeline, time_dimension=0)
```
### Time-based Indexing
```python
# Access data at a specific time
data_at_time = timelined_array.itime[5.0]

# Aligning arrays
aligned_array = TimelinedArray.align_from_iterable([timelined_array, another_timelined_array])
```


### Masked TimelinedArray
```python
from timelined_array import MaskedTimelinedArray

masked_data = np.ma.masked_array(data, mask=data > 0.5)
masked_timelined_array = MaskedTimelinedArray(masked_data, timeline=timeline, time_dimension=0)
```

### Converting Seconds to Index
```python
masked_data.itime.time_to_index(5.0)
```

## Misc

### License
This package is licensed under the MIT License. See the LICENSE file for more details.

### Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

### Contact
For any questions or issues, please contact the package maintainer.
