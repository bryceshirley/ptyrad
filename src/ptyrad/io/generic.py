"""
Generic file handling (load/save) for raw, npy, tif, and zarr formats

"""

import os
from typing import Any, Optional

import numpy as np
from tifffile import imread, imwrite


def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    """Loads a raw binary file containing interleaved image data and gaps.

    This implementation uses a custom `numpy.dtype` with `np.fromfile` for 
    fast I/O performance, extracting only the valid data regions and skipping 
    the specified byte gaps between frames. Note that custom processed raw 
    data might have a gap of 0.

    Args:
        file_path (str): The path to the raw binary file.
        shape (tuple of int): The expected shape of the data in the format 
            (N, height, width), where N is the number of frames.
        dtype (data-type, optional): The NumPy data type of the image pixels. 
            Defaults to np.float32.
        offset (int, optional): The number of bytes to skip at the beginning 
            of the file. Defaults to 0.
        gap (int, optional): The number of gap bytes to skip between each 
            image frame. Defaults to 1024.

    Returns:
        numpy.ndarray: An array of the extracted data with the specified shape 
        and dtype.

    Raises:
        ValueError: If the actual file size does not match the expected size 
            calculated from the inputs.
    """
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape

    # Verify file size first
    expected_size = offset + N * (height * width * dtype().itemsize + gap)
    actual_size = os.path.getsize(file_path)

    if actual_size != expected_size:
        raise ValueError(f"Mismatch in expected ({expected_size} bytes = offset + N * (height * width * 4 + gap)) vs. actual ({actual_size} bytes) file size! Check your loading configurations!")
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap)  # uint8 means 1 byte per gap element
    ])

    # Read the entire file using the custom dtype
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    return data

def load_tif(file_path):
    """Loads an image array from a TIFF file.

    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: The loaded image data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = imread(file_path)

    return data

def load_npy(file_path):
    """Loads an array from a binary NumPy .npy file.

    Args:
        file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray: The loaded array data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")
    
    data = np.load(file_path)

    return data

def _normalize_zarr_key(key: Optional[str]) -> Optional[str]:
    """Normalize HDF5-like keys for Zarr path access."""
    if key in (None, ""):
        return None
    return key.lstrip("/")

def _normalize_zarr_selection(selection):
    """Convert YAML-friendly slice specs into NumPy/Zarr indexing objects."""
    if selection is None:
        return None
    if isinstance(selection, slice):
        return selection
    if isinstance(selection, int):
        return selection
    if selection is Ellipsis:
        return selection
    if not isinstance(selection, (list, tuple)):
        raise TypeError(
            "`zarr_kwargs['selection']` must be an int, slice, or a list/tuple of "
            "ints, slices, nulls, or [start, stop, step] slice specs."
        )

    normalized = []
    for axis_selection in selection:
        if axis_selection is None:
            normalized.append(slice(None))
        elif isinstance(axis_selection, (int, slice)):
            normalized.append(axis_selection)
        elif axis_selection is Ellipsis:
            normalized.append(axis_selection)
        elif isinstance(axis_selection, (list, tuple)):
            if len(axis_selection) > 3:
                raise ValueError(
                    "Each Zarr slice spec must be [start], [start, stop], or "
                    "[start, stop, step]."
                )
            normalized.append(slice(*axis_selection))
        else:
            raise TypeError(
                "Each Zarr axis selection must be an int, slice, null, ellipsis, "
                "or [start, stop, step] slice spec."
            )
    return tuple(normalized)

def _zarr_array_to_numpy(zarray, selection=None):
    if selection is None:
        selection = Ellipsis
    return np.asarray(zarray[selection])

def _zarr_selection_ndim(ndim, selection=None):
    if selection is None or selection is Ellipsis:
        return ndim
    if isinstance(selection, int):
        return ndim - 1
    if isinstance(selection, slice):
        return ndim

    ellipsis_count = sum(axis_selection is Ellipsis for axis_selection in selection)
    if ellipsis_count > 1:
        raise IndexError("Only one ellipsis is allowed in `zarr_kwargs['selection']`.")

    consumed_axes = sum(axis_selection is not Ellipsis for axis_selection in selection)
    if consumed_axes > ndim:
        return None

    indexed_axes = sum(isinstance(axis_selection, int) for axis_selection in selection)
    return ndim - indexed_axes

def _collect_zarr_array_refs(zgroup, ndims=None, selection=None, _parent_key=None):
    results = {}
    for key in zgroup.keys():
        full_key = f"{_parent_key}/{key}" if _parent_key else key
        value = zgroup[key]
        if hasattr(value, "shape") and hasattr(value, "ndim"):
            selected_ndim = _zarr_selection_ndim(value.ndim, selection)
            if selected_ndim is not None and selected_ndim in ndims:
                results[full_key] = value
        elif hasattr(value, "keys"):
            results.update(
                _collect_zarr_array_refs(
                    value,
                    ndims=ndims,
                    selection=selection,
                    _parent_key=full_key,
                )
            )
    return results

def load_zarr(file_path, key=None, ndims=None, zarr_kwargs: Optional[dict[str, Any]] = None):
    """Loads an array from a Zarr store.

    Args:
        file_path (str): Path to the Zarr store.
        key (str, optional): Internal path to the array inside a Zarr group.
        ndims (list, optional): Desired dimensions when searching a group with no key.
        zarr_kwargs (dict, optional): Optional Zarr load settings. Use
            ``selection`` to slice before converting to NumPy, e.g.
            ``{'selection': [[0, 64], [0, 64], None, None]}``. Remaining entries
            are passed to ``zarr.open``.

    Returns:
        numpy.ndarray: The loaded array data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist. Please check your file path and working directory.")

    import zarr

    if ndims is None:
        ndims = [3, 4]

    open_kwargs = dict(zarr_kwargs or {})
    selection = _normalize_zarr_selection(
        open_kwargs.pop("selection", open_kwargs.pop("slices", None))
    )
    zarr_path = _normalize_zarr_key(key)

    if zarr_path is not None and "path" in open_kwargs:
        raise ValueError(
            "Specify the Zarr array path with either `key` or "
            "`zarr_kwargs['path']`, not both."
        )

    zobj = zarr.open(file_path, mode="r", path=zarr_path, **open_kwargs)
    if hasattr(zobj, "shape") and hasattr(zobj, "ndim"):
        selected_ndim = _zarr_selection_ndim(zobj.ndim, selection)
        if selected_ndim is None:
            raise IndexError(
                f"Selection consumes more axes than the Zarr array has ({zobj.ndim})."
            )
        return _zarr_array_to_numpy(zobj, selection=selection)

    if hasattr(zobj, "keys"):
        if zarr_path is not None:
            raise ValueError(
                f"The returned Zarr value at key '{key}' is a group, not an array. "
                "Please specify the dataset key explicitly."
            )

        valid_datasets = _collect_zarr_array_refs(zobj, ndims=ndims, selection=selection)
        if len(valid_datasets) == 1:
            return _zarr_array_to_numpy(next(iter(valid_datasets.values())), selection=selection)
        if len(valid_datasets) == 0:
            raise ValueError(
                f"No eligible Zarr arrays found with ndims = {ndims}. Please check the store or specify `key`."
            )
        raise ValueError(
            f"Multiple eligible Zarr arrays found: {list(valid_datasets.keys())}. Please specify the dataset key explicitly."
        )

    raise ValueError(f"Unsupported Zarr object type: {type(zobj).__name__}")

def write_tif(file_path, data):
    """Saves a NumPy array as a TIFF file.

    The file is saved with ImageJ compatibility enabled to ensure proper 
    handling of hyperstacks and metadata in common microscopy viewers.

    Args:
        file_path (str): The destination path for the TIFF file.
        data (numpy.ndarray): The array data to save.
    """
    imwrite(file_path, data, imagej=True)

def write_npy(file_path, data):
    """Saves a NumPy array to a binary .npy file.

    Args:
        file_path (str): The destination path for the .npy file.
        data (numpy.ndarray): The array data to save.
    """
    np.save(file_path, data)
