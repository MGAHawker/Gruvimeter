# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 13:50:03 2025

@author: migsh
"""
import numpy as np
import scipy as scipy
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import optoanalysis as opt
import glob
import os
from datetime import timedelta
import datetime
import seaborn as sns
import json
from pathlib import Path
from scipy.io import savemat, loadmat

# Define your colour scheme
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Set as default colour cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOURS)
# =============================================================================
# DATA SAVING/LOADING FUNCTIONS
# =============================================================================

def save_processed_data(data_dict, filepath, compress=True):
    """
    Save processed data to disk with optional compression
    
    Parameters:
    data_dict: Dictionary containing arrays and metadata
    filepath: Path to save file (.pkl, .npz, or .mat)
    compress: Whether to use compression (recommended for large arrays)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.mat':
        # Use MATLAB format - scipy.io.savemat
        # Note: savemat has limitations with nested structures
        # Convert any datetime objects to strings for compatibility
        mat_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, (datetime.datetime, np.datetime64)):
                mat_dict[key] = str(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], datetime.datetime):
                mat_dict[key] = [str(dt) for dt in value]
            else:
                mat_dict[key] = value
        
        # MATLAB format options
        savemat(
            filepath, 
            mat_dict, 
            do_compression=compress,
            oned_as='column'  # Save 1D arrays as column vectors (MATLAB convention)
        )
        print(f"Saved to {filepath} (MATLAB format)")
        
    elif filepath.suffix == '.npz':
        # Use numpy's compressed format
        if compress:
            np.savez_compressed(filepath, **data_dict)
        else:
            np.savez(filepath, **data_dict)
        print(f"Saved to {filepath} (NumPy format)")
        

    else:
        raise ValueError("Filepath must end with .mat or .npz")
    
    # Print file size
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")
def load_processed_data(filepath):
    """
    Load processed data from disk
    
    Returns:
    Dictionary containing arrays and metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.mat':
        # Load MATLAB format
        loaded = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        
        # Remove MATLAB metadata fields
        data_dict = {key: value for key, value in loaded.items() 
                     if not key.startswith('__')}
        
        print(f"Loaded from {filepath} (MATLAB format)")
        
    elif filepath.suffix == '.npz':
        # Load numpy format
        loaded = np.load(filepath, allow_pickle=True)
        # Convert to regular dict for easier access
        data_dict = {key: loaded[key] for key in loaded.files}
        print(f"Loaded from {filepath} (NumPy format)")

    else:
        raise ValueError("Filepath must end with .mat or .npz")
    
    # Print data summary
    print(f"Loaded data contains {len(data_dict)} fields:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    return data_dict
# =============================================================================
# PIPELINE SAVE FUNCTIONS
# =============================================================================

def save_angle_segments_with_datetime(angle_segments, file_datetimes, output_dir, dataset_name):
    """
    Save per-file angle segments with datetime information (Level 1 data)
    
    Parameters:
    angle_segments: List of angle segment dictionaries
    file_datetimes: List of (filepath, datetime) tuples
    output_dir: Output directory path
    dataset_name: Name of the dataset
    """
    output_path = Path(output_dir) / f"{dataset_name}_angle_segments.mat"
    
    # Prepare data for MATLAB format
    # MATLAB doesn't handle nested lists well, so flatten the structure
    save_dict = {
        'dataset_name': dataset_name,
        'num_segments': len(angle_segments),
        # Datetime information
        'file_paths': [str(fp) for fp, dt in file_datetimes],
        'file_datetimes_str': [dt.strftime('%Y-%m-%d %H:%M:%S.%f') for fp, dt in file_datetimes],
        'file_timestamps': np.array([dt.timestamp() for fp, dt in file_datetimes]),
    }
    
    # Store segment data as separate arrays
    for i, segment in enumerate(angle_segments):
        prefix = f'seg{i:03d}_'
        save_dict[f'{prefix}time'] = segment['time']
        save_dict[f'{prefix}angles'] = segment['angles']
        save_dict[f'{prefix}start_time'] = segment['start_time']
        save_dict[f'{prefix}end_time'] = segment['end_time']
        save_dict[f'{prefix}index'] = segment['index']
        save_dict[f'{prefix}ch1_name'] = segment['ch1_name']
        save_dict[f'{prefix}ch2_name'] = segment['ch2_name']
    
    save_processed_data(save_dict, output_path, compress=True)
    
    return output_path

def save_stitched_position_trace(stitched_time, stitched_angles, position, 
                                 velocity, phase_changes, angle_segments,
                                 stitch_info, output_dir, dataset_name,
                                 file_datetimes=None, metadata=None):
    """
    Save full-resolution stitched position trace (Level 2 data - THE KEY ONE)
    
    This is the main reusable asset for different analyses
    """
    # Extract datetime span for filename
    datetime_suffix = ""
    if file_datetimes is not None and len(file_datetimes) > 0:
        start_dt = file_datetimes[0][1]
        end_dt = file_datetimes[-1][1]
        
        # Format: dataset_YYYYMMDD-HHMMSS_to_YYYYMMDD-HHMMSS.mat
        datetime_suffix = (f"_{start_dt.strftime('%Y%m%d-%H%M%S')}_"
                          f"to_{end_dt.strftime('%Y%m%d-%H%M%S')}")
    
    output_path = Path(output_dir) / f"{dataset_name}{datetime_suffix}_full_position_trace.mat"
    # Prepare data dictionary
    data_dict = {
        'time': stitched_time,
        'angles': stitched_angles,
        'position': position,
        'velocity': velocity,
        'phase_changes': phase_changes,
        # Store segment boundaries for edge exclusion
        'segment_start_times': np.array([seg['start_time'] for seg in angle_segments]),
        'segment_end_times': np.array([seg['end_time'] for seg in angle_segments]),
        'segment_indices': np.array([seg['index'] for seg in angle_segments]),
    }
    
    # Add datetime information if provided
    if file_datetimes is not None:
        data_dict['file_datetimes_str'] = [dt.strftime('%Y-%m-%d %H:%M:%S.%f') 
                                            for fp, dt in file_datetimes]
        data_dict['file_timestamps'] = np.array([dt.timestamp() 
                                                  for fp, dt in file_datetimes])
        data_dict['file_paths'] = [str(fp) for fp, dt in file_datetimes]
    
    # Add metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'dataset_name': dataset_name,
        'num_segments': len(angle_segments),
        'total_duration_hours': float((stitched_time[-1] - stitched_time[0]) / 3600),
        'sample_rate_hz': float(1 / np.median(np.diff(stitched_time))),
        'laser_wavelength_nm': 1550,
        'processing_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Add metadata fields directly (MATLAB compatible)
    for key, value in metadata.items():
        if isinstance(value, (int, float, str)):
            data_dict[f'meta_{key}'] = value
        elif isinstance(value, dict):
            # Flatten nested dicts for MATLAB
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float, str)):
                    data_dict[f'meta_{key}_{subkey}'] = subvalue
    
    save_processed_data(data_dict, output_path, compress=True)
    
    # Also save a JSON metadata file for easy inspection
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json_metadata = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict))
                        else v for k, v in metadata.items()}
        json.dump(json_metadata, f, indent=2)
    print(f"Metadata also saved to {metadata_path}")
    
    return output_path
# =============================================================================
# PIPELINE LOADER FUNCTIONS
# =============================================================================

def load_full_position_trace(filepath):
    """
    Load full position trace and reconstruct angle_segments info
    
    Returns:
    Dictionary with all position trace data ready for analysis
    """
    data = load_processed_data(filepath)
    
    # Reconstruct angle_segments structure for compatibility
    angle_segments = []
    for i in range(len(data['segment_start_times'])):
        angle_segments.append({
            'start_time': data['segment_start_times'][i],
            'end_time': data['segment_end_times'][i],
            'index': data['segment_indices'][i]
        })
    
    # Extract metadata if it's stored as object array
    metadata = data.get('metadata')
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item()  # Extract from numpy object array
    
    return {
        'time': data['time'],
        'angles': data['angles'],
        'position': data['position'],
        'velocity': data['velocity'],
        'phase_changes': data['phase_changes'],
        'angle_segments': angle_segments,
        'metadata': metadata
    }

# =============================================================================
# MEMORY-SAVER PIPELINE MANAGER
# =============================================================================
class PositionTraceProcessor:
    """
    Manager class for processing and storing position traces
    """
    
    def __init__(self, output_dir, dataset_name):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialised processor for dataset: {dataset_name}")
        print(f"Output directory: {self.output_dir}")
    
    def save_full_trace(self, stitched_time, stitched_angles, position, 
                       velocity, phase_changes, angle_segments, stitch_info,
                       file_datetimes=None, **metadata_kwargs):
        """Save full position trace with datetime information"""
        self.full_trace_path = save_stitched_position_trace(
            stitched_time, stitched_angles, position, velocity, 
            phase_changes, angle_segments, stitch_info,
            self.output_dir, self.dataset_name, 
            file_datetimes=file_datetimes,
            metadata=metadata_kwargs
        )
        print(f"Full trace saved. Use load_full_trace() to reload.")
        return self.full_trace_path
    
    def save_angle_segments(self, angle_segments, file_datetimes):
        """Save angle segments with datetime information"""
        self.angle_segments_path = save_angle_segments_with_datetime(
            angle_segments, file_datetimes,
            self.output_dir, self.dataset_name
        )
        return self.angle_segments_path
    
    def load_full_trace(self):
        """Load full position trace"""
        if not hasattr(self, 'full_trace_path'):
            # Try to find it
            for ext in ['.mat', '.npz', '.pkl']:
                pattern = f"{self.dataset_name}_full_position_trace{ext}"
                matches = list(self.output_dir.glob(pattern))
                if matches:
                    self.full_trace_path = matches[0]
                    break
            else:
                raise FileNotFoundError(f"No full trace found for {self.dataset_name}")
        
        return load_processed_data(self.full_trace_path)
#==============================================================================    
# =============================================================================
# HEAVY COMPUTATION - RUN ONCE, SAVE RESULTS
# =============================================================================
#==============================================================================
# Configuration
DATASET_NAME = "Processed_Data_Chunk"
OUTPUT_DIR = "D:/USB Drive Backup/Timetraces and grouping notes/processed_data_mats/Chunk 1 Output"
base_dir = "D:/USB Drive Backup/Timetraces and grouping notes/Pipeline trace file chunks sorted/23Jul-06Oct 04328-05244"
# Initialise processor
processor = PositionTraceProcessor(OUTPUT_DIR, DATASET_NAME)
#==============================================================================
#MAIN CONVERSION PIPELINE STEP
#==============================================================================
def correct_symmetric_time_axis(time_data, verbose=True):
    """
    Correct symmetric time axis from oscilloscope to start from 0
    
    Returns:
    corrected_time: Time array starting from 0
    correction_info: Dictionary with correction details
    """
    original_start = time_data[0]
    original_end = time_data[-1]
    original_duration = original_end - original_start
    
    if original_start < 0 and abs(original_start + original_end) < abs(original_start) * 0.1:
        # Symmetric axis detected
        corrected_time = time_data - original_start
        correction_applied = True
        if verbose:
            print(f"  Symmetric time axis detected: {original_start:.3f}s to {original_end:.3f}s")
            print(f"  Corrected to: 0.000s to {corrected_time[-1]:.3f}s")
    
    elif original_start < 0:
        # Negative start but not symmetric
        corrected_time = time_data - original_start
        correction_applied = True
        if verbose:
            print(f"  Negative start time detected: {original_start:.3f}s")
            print(f"  Shifted to start from 0.000s")
    
    else:
        # Time already starts from 0 or positive
        corrected_time = time_data.copy()
        correction_applied = False
        if verbose:
            print(f"  Time axis OK: {original_start:.3f}s to {original_end:.3f}s")
    
    correction_info = {
        'correction_applied': correction_applied,
        'original_start': original_start,
        'original_end': original_end,
        'corrected_start': corrected_time[0],
        'corrected_end': corrected_time[-1],
        'duration': original_duration,
        'time_shift': corrected_time[0] - original_start
    }
    
    return corrected_time, correction_info

def construct_absolute_time_axis_sequential(channel_data, measurement_start_datetime=None,
                                           use_first_file_ctime=True):
    """
    Construct absolute time axis by sequential ordering
    
    This assumes:
    1. Files are numbered sequentially (01982, 01983, 01984...)
    2. Files were recorded back-to-back with teeny gaps
    3. Each file's internal time axis is relative to that file's start
    
    Parameters:
    channel_data: Dictionary with file_paths, data_segments, time_segments
    measurement_start_datetime: Optional datetime object for absolute reference
    use_first_file_ctime: If True and measurement_start_datetime is None, 
                          use first file's creation time as reference
    
    Returns:
    absolute_start_times: Array of absolute start times for each file (seconds from t=0)
    file_datetimes: List of (filepath, datetime) tuples for metadata
    measurement_start_datetime: The actual datetime used as reference
    """
    
    reference_channel = list(channel_data.keys())[0]
    file_paths = channel_data[reference_channel]['file_paths']
    time_segments = channel_data[reference_channel]['time_segments']
    
    print(f"\n{'='*80}")
    print(f"CONSTRUCTING SEQUENTIAL ABSOLUTE TIME AXIS")
    print(f"{'='*80}")
    print(f"Method: Sequential ordering (NO per-file creation times)")
    print(f"Number of files: {len(file_paths)}")
    
    # Calculate duration of each file from its internal time axis
    file_durations = []
    for i, time_seg in enumerate(time_segments):
        duration = time_seg[-1] - time_seg[0]
        file_durations.append(duration)
        if i < 3:  # Show first 3
            print(f"  File {i:04d}: duration = {duration:.3f} seconds ({duration/60:.2f} minutes)")
    
    if len(file_paths) > 3:
        print(f"  ... ({len(file_paths) - 3} more files)")
    
    # Construct absolute start times sequentially
    absolute_start_times = [0.0]  # First file starts at t=0
    
    for i in range(1, len(file_paths)):
        # Each file starts when the previous file ended
        prev_start = absolute_start_times[i-1]
        prev_duration = file_durations[i-1]
        new_start = prev_start + prev_duration
        absolute_start_times.append(new_start)
    
    # Convert to numpy array
    absolute_start_times = np.array(absolute_start_times)
    
    # Calculate absolute end times for verification
    absolute_end_times = absolute_start_times + np.array(file_durations)
    
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL TIME AXIS VERIFICATION")
    print(f"{'='*80}")
    
    # Check for overlaps (should be ZERO with this method dear god please)
    overlaps_found = False
    gaps_found = False
    for i in range(len(file_paths) - 1):
        gap = absolute_start_times[i+1] - absolute_end_times[i]
        if gap < -0.01:  # Small negative tolerance for floating point
            print(f"  ⚠ WARNING: Overlap between files {i} and {i+1}: {gap:.3f} seconds")
            overlaps_found = True
        elif gap > 0.1:  # More than 0.1 second gap
            print(f"  ⚠ WARNING: Gap between files {i} and {i+1}: {gap:.3f} seconds")
            gaps_found = True
    
    if not overlaps_found and not gaps_found:
        print(f"  ✓ No overlaps or gaps detected - sequential ordering is consistent")
    
    # Print time axis summary
    total_duration = absolute_end_times[-1]
    print(f"\nTime axis summary:")
    print(f"  First file starts: {absolute_start_times[0]:.3f} seconds (relative t=0)")
    print(f"  Last file ends: {absolute_end_times[-1]:.3f} seconds")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"  Average file duration: {np.mean(file_durations):.2f} seconds")
    print(f"  Min file duration: {np.min(file_durations):.2f} seconds")
    print(f"  Max file duration: {np.max(file_durations):.2f} seconds")
    
    # ==========================================================================
    # DETERMINE ABSOLUTE TIME REFERENCE
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"ABSOLUTE TIME REFERENCE")
    print(f"{'='*80}")
    
    if measurement_start_datetime is not None:
        # User provided explicit datetime
        print(f"Using user-provided start time: {measurement_start_datetime}")
        print(f"  Source: Manual specification")
        
    elif use_first_file_ctime:
        # Extract creation time from first file
        first_file_path = file_paths[0]
        
        # Try modification time first (more reliable than creation time)
        try:
            first_file_mtime = os.path.getmtime(first_file_path)
            measurement_start_datetime = datetime.datetime.fromtimestamp(first_file_mtime)
            print(f"Using first file's modification time: {measurement_start_datetime}")
            print(f"  Source: {os.path.basename(first_file_path)}")
            print(f"  Note: This assumes file was saved immediately after recording started")
            
            # Sanity check - is this a reasonable date?
            now = datetime.datetime.now()
            if measurement_start_datetime.year < 2025:
                print(f"  ⚠ WARNING: Date is before 2020 -incorrect")
            elif measurement_start_datetime > now:
                print(f"  ⚠ WARNING: Date is in the future - definitely incorrect!")
                print(f"  Falling back to relative times only")
                measurement_start_datetime = None
            else:
                print(f"  ✓ Date looks reasonable")
                
        except Exception as e:
            print(f"  ⚠ ERROR: Could not read file modification time: {e}")
            print(f"  Falling back to relative times only")
            measurement_start_datetime = None
    else:
        print(f"No absolute time reference provided")
        print(f"  Using relative times only (t=0 at first file)")
        measurement_start_datetime = None
    
    # ==========================================================================
    # CREATE FILE DATETIMES
    # ==========================================================================
    file_datetimes = []
    
    if measurement_start_datetime is not None:
        print(f"\nConstructing absolute datetimes for all files:")
        for i, (filepath, start_time) in enumerate(zip(file_paths, absolute_start_times)):
            file_dt = measurement_start_datetime + timedelta(seconds=float(start_time))
            file_datetimes.append((filepath, file_dt))
            if i < 3:  # Show first 3
                print(f"  File {i:04d}: {file_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        if len(file_paths) > 3:
            print(f"  ... ({len(file_paths) - 3} more files)")
            # Show last file too
            print(f"  File {len(file_paths)-1:04d}: {file_datetimes[-1][1].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        # Calculate total duration as datetime span
        measurement_end_datetime = file_datetimes[-1][1] + timedelta(seconds=file_durations[-1])
        total_time_span = measurement_end_datetime - measurement_start_datetime
        
        print(f"\nAbsolute time span:")
        print(f"  Start: {measurement_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End:   {measurement_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {total_time_span.total_seconds()/3600:.2f} hours ({total_time_span.days} days)")
        
    else:
        print(f"\nUsing relative times only - no absolute datetimes available")
        # Create placeholder datetimes for structure compatibility
        placeholder_start = datetime.datetime(2000, 1, 1, 0, 0, 0)
        for filepath, start_time in zip(file_paths, absolute_start_times):
            file_dt = placeholder_start + timedelta(seconds=float(start_time))
            file_datetimes.append((filepath, file_dt))
        print(f"  (Placeholder datetimes created for compatibility)")
    
    print(f"{'='*80}\n")
    
    return absolute_start_times, file_datetimes, file_durations, measurement_start_datetime


# Configuration for channels (WHERE IS THE DATA? WHATS THE NAMING CONVENTION?)
base_dir = "D:/USB Drive Backup/Timetraces and grouping notes/Pipeline trace file chunks sorted/23Jul-06Oct 04328-05244"
channels = {
    'C1': "C1--trace--*.trc",
    'C2': "C2--trace--*.trc"
}

# =============================================================================
# ABSOLUTE TIME REFERENCE OPTIONS
# =============================================================================
# Option 1: If you know the exact start time, specify it here:
MEASUREMENT_START_DATETIME = None  # e.g., datetime(2024, 7, 23, 14, 30, 0)

# Option 2: If None, automatically use first file's creation time:
USE_FIRST_FILE_CTIME = True  # Set to False to use only relative times

print(f"{'='*80}")
print(f"ABSOLUTE TIME CONFIGURATION")
print(f"{'='*80}")
if MEASUREMENT_START_DATETIME is not None:
    print(f"Mode: User-specified absolute time")
    print(f"Start time: {MEASUREMENT_START_DATETIME}")
elif USE_FIRST_FILE_CTIME:
    print(f"Mode: Automatic from first file's creation time")
    print(f"Will extract: First file modification time as reference")
else:
    print(f"Mode: Relative times only")
    print(f"No absolute time reference will be used")
print(f"{'='*80}\n")

# Configuration for channels
channels = {
    'C1': "C1--trace--*.trc",
    'C2': "C2--trace--*.trc"
}

# Dictionary to store data from each channel
channel_data = {}

# Load data for all channels
for channel_name, file_pattern in channels.items():
    print(f"Loading {channel_name} files...")
    
    # Get all files for this channel
    full_pattern = os.path.join(base_dir, file_pattern)
    file_paths = sorted(glob.glob(full_pattern),
                       key=lambda x: int(x.split('--')[-1].split('.')[0]))
    
    if not file_paths:
        print(f"Warning: No files found for {channel_name} with pattern {file_pattern}")
        continue
    
    # Lists to store data from each file for this channel
    data_segments = []
    time_segments = []
    
    # Load all the data segments for this channel
    for i, file_path in enumerate(file_paths):
        print(f"  Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load data from files
        data = opt.load_data(file_path)
        time_data, voltage = data.get_time_data()
        
        # CORRECT THE SYMMETRIC TIME AXIS (if present)
        corrected_time, time_correction_info = correct_symmetric_time_axis(
            time_data, verbose=(i < 3)  # Only verbose for first 3 files
        )
        
        # Store corrected data
        data_segments.append(voltage)
        time_segments.append(corrected_time)
    
    # Store channel data
    channel_data[channel_name] = {
        'file_paths': file_paths,
        'data_segments': data_segments,
        'time_segments': time_segments
    }
    
    print(f"Loaded {len(file_paths)} files for {channel_name}\n")

# =============================================================================
# CONSTRUCT ABSOLUTE TIME AXIS USING SEQUENTIAL ORDERING
# =============================================================================

absolute_start_times, file_datetimes, file_durations, measurement_start_datetime = \
    construct_absolute_time_axis_sequential(
        channel_data, 
        measurement_start_datetime=MEASUREMENT_START_DATETIME,
        use_first_file_ctime=USE_FIRST_FILE_CTIME
    )

# Store the determined start time for later use
if measurement_start_datetime is not None:
    print(f"✓ Absolute time reference established: {measurement_start_datetime}")
else:
    print(f"⚠ Using relative times only - absolute time reference not available")

# Verification output
print(f"\n{'='*80}")
print(f"FILE TIMING SUMMARY")
print(f"{'='*80}")
print(f"First 5 file durations: {[f'{d:.2f}s' for d in file_durations[:5]]}")
print(f"First 5 absolute start times: {[f'{t:.2f}s' for t in absolute_start_times[:5]]}")

print(f"\nDetailed verification (first 3 files):")
reference_channel = list(channel_data.keys())[0]
for i in range(min(3, len(channel_data[reference_channel]['time_segments']))):
    seg = channel_data[reference_channel]['time_segments'][i]
    abs_start = absolute_start_times[i]
    abs_end = abs_start + file_durations[i]
    
    print(f"\n  File {i:04d}: {os.path.basename(file_datetimes[i][0])}")
    print(f"    Internal time: {seg[0]:.3f}s → {seg[-1]:.3f}s (duration: {seg[-1]-seg[0]:.3f}s)")
    print(f"    Absolute time: {abs_start:.3f}s → {abs_end:.3f}s")
    
    if measurement_start_datetime is not None:
        file_start_dt = file_datetimes[i][1]
        file_end_dt = file_start_dt + timedelta(seconds=file_durations[i])
        print(f"    Datetime: {file_start_dt.strftime('%Y-%m-%d %H:%M:%S')} → "
              f"{file_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n{'='*80}\n")

def smart_downsample_for_correction(ch1_data, ch2_data, max_samples=50000):
    """
    Downsample data for quadrature correction to prevent memory issues
    
    Parameters:
    ch1_data, ch2_data: Input channel data
    max_samples: Maximum number of samples to use for correction fitting
    
    Returns:
    ch1_subset, ch2_subset: Downsampled data for fitting
    downsample_factor: Factor used for downsampling
    """
    data_length = len(ch1_data)
    
    if data_length <= max_samples:
        return ch1_data, ch2_data, 1
    
    # Calculate downsampling factor
    downsample_factor = int(np.ceil(data_length / max_samples))
    
    # Use systematic sampling to maintain temporal distribution
    indices = np.arange(0, data_length, downsample_factor)
    indices = indices[:max_samples]  # Ensure we don't exceed max_samples
    
    ch1_subset = ch1_data[indices]
    ch2_subset = ch2_data[indices]
    
    print(f"Downsampled from {data_length} to {len(ch1_subset)} samples for correction fitting (factor: {downsample_factor})")
    
    return ch1_subset, ch2_subset, downsample_factor

def find_ellipse_center(ch1_data, ch2_data):
    """
    Find center of ellipse using direct least squares ellipse fitting
    More accurate than circle fitting for elliptical data
    
    Returns:
    xc, yc: Center coordinates
    """
    
    # Direct least squares ellipse fit
    X = ch1_data
    Y = ch2_data
    
    # Design matrix for ellipse: ax² + bxy + cy² + dx + ey + f = 0
    D = np.column_stack([X**2, X*Y, Y**2, X, Y, np.ones_like(X)])
    
    # Constraint matrix for ellipse (b² - 4ac < 0)
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    
    # Solve generalised eigenproblem
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)
        
        # Find positive eigenvalue
        pos_eigval_idx = np.where(eigvals > 0)[0]
        if len(pos_eigval_idx) == 0:
            # Fallback to median if fitting fails
            print("  Warning: Ellipse fit failed, using median fallback")
            return np.median(X), np.median(Y)
        
        params = eigvecs[:, pos_eigval_idx[0]]
        
        # Extract center from ellipse parameters
        a, b, c, d, e, f = params
        
        # Center coordinates: xc = (be - 2cd)/(4ac - b²), yc = (bd - 2ae)/(4ac - b²)
        denominator = 4*a*c - b**2
        
        if abs(denominator) < 1e-10:
            print("  Warning: Degenerate ellipse, using median fallback")
            return np.median(X), np.median(Y)
            
        xc = (b*e - 2*c*d) / denominator
        yc = (b*d - 2*a*e) / denominator
        
        return xc, yc
        
    except np.linalg.LinAlgError:
        print("  Warning: Ellipse fit failed, using median fallback")
        return np.median(X), np.median(Y)
def correct_quadrature_data_simple(ch1_data, ch2_data, remove_dc=True, 
                                  show_correction=False, method='ellipse'):
    """
    Simple quadrature correction with geometric centering
    
    Parameters:
    method: 'mean', 'median', 'algebraic' (circle), or 'ellipse'
    """
    ch1_orig = ch1_data.copy()
    ch2_orig = ch2_data.copy()
    
    correction_info = {'method': f'geometric_center_{method}'}
    
    if remove_dc:
        if method == 'mean':
            ch1_offset = np.mean(ch1_data)
            ch2_offset = np.mean(ch2_data)
        elif method == 'median':
            ch1_offset = np.median(ch1_data)
            ch2_offset = np.median(ch2_data)
        elif method == 'ellipse':
            ch1_offset, ch2_offset = find_ellipse_center(ch1_data, ch2_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ch1_corrected = ch1_data - ch1_offset
        ch2_corrected = ch2_data - ch2_offset
        
        correction_info.update({
            'ch1_offset': ch1_offset,
            'ch2_offset': ch2_offset,
            'centering_applied': True,
            'centering_method': method
        })
        print(f"  Applied {method} center correction: ({ch1_offset:.4f}, {ch2_offset:.4f})")
    else:
        ch1_corrected = ch1_data
        ch2_corrected = ch2_data
        correction_info['centering_applied'] = False
    
    if show_correction:
        plot_simple_correction_results(ch1_orig, ch2_orig, ch1_corrected, ch2_corrected, correction_info)
    
    return ch1_corrected, ch2_corrected, correction_info

# CONSERVATIVE PHASE JUMP HANDLING FUNCTIONS (the angle jumps are driving me insane)
def ultra_minimal_phase_jump_correction(angles, time_array, jump_threshold=6.28, 
                                       show_corrections=False, require_2pi_multiple=True):
    """
    ULTRA-MINIMAL phase jump detection - only fixes the most obvious discontinuities
    
    Parameters:
    angles: Array of phase angles
    time_array: Time array (for plotting)
    jump_threshold: Very high threshold (10+ rad) - only fix massive jumps
    require_2pi_multiple: Only correct jumps that are very close to 2π multiples
    show_corrections: Whether to show correction plots
    
    Returns:
    corrected_angles: Minimally corrected angles
    correction_info: Information about corrections applied
    """
    
    print(f"=== PHASE JUMP CORRECTION ===")
    
    original_angles = angles.copy()
    corrected_angles = angles.copy()
    
    # Calculate phase differences
    angle_diffs = np.diff(angles)
    
    # Use very high threshold - only catch truly massive jumps
    massive_jumps = np.abs(angle_diffs) > jump_threshold
    
    if not np.any(massive_jumps):
        print(f"✓ No massive phase jumps detected above {jump_threshold:.1f} rad threshold")
        return angles, {'method': 'ultra_minimal', 'jumps_corrected': 0, 'total_correction': 0}
    
    jump_indices = np.where(massive_jumps)[0] + 1
    jump_magnitudes = angle_diffs[massive_jumps]
    
    print(f"Found {len(jump_indices)} potential massive jumps above {jump_threshold:.1f} rad")
    
    corrections_applied = 0
    total_correction = 0
    
    # Process each potential jump with extreme caution
    for i, (jump_idx, jump_mag) in enumerate(zip(jump_indices, jump_magnitudes)):
        
        if require_2pi_multiple:
            # Only correct if it's VERY close to a 2π multiple
            n_periods = jump_mag / (2 * np.pi)
            nearest_integer = np.round(n_periods)
            
            # Very strict criterion - must be within 0.1 of integer multiple
            residual = abs(n_periods - nearest_integer)
            
            if residual < 0.1 and abs(nearest_integer) >= 1:
                # This is almost certainly a 2π phase wrap error
                correction = -nearest_integer * 2 * np.pi
                
                # Apply correction to all subsequent points
                corrected_angles[jump_idx:] += correction
                corrections_applied += 1
                total_correction += correction
                
                print(f"  CORRECTED jump at index {jump_idx}: {jump_mag:.2f} rad → {correction:.2f} rad")
            else:
                # Could be real motion - absolutely do not touch it
                print(f"  PRESERVED jump at index {jump_idx}: {jump_mag:.2f} rad (not precise 2π multiple)")
        else:
            # Even without 2π requirement, be conservative
            # Only correct jumps that are really big (>6.5 rad)
            if abs(jump_mag) > 6.2:
                # Assume it's probably a phase wrap and correct by the nearest 2π
                n_periods = jump_mag / (2 * np.pi)
                nearest_integer = np.round(n_periods)
                correction = -nearest_integer * 2 * np.pi
                
                corrected_angles[jump_idx:] += correction
                corrections_applied += 1
                total_correction += correction
                
                print(f"  CORRECTED massive jump at index {jump_idx}: {jump_mag:.2f} rad → {correction:.2f} rad")
            else:
                print(f"  PRESERVED jump at index {jump_idx}: {jump_mag:.2f} rad (could be real)")
    
    print(f"Applied {corrections_applied} corrections out of {len(jump_indices)} candidates")
    print(f"Preserved {len(jump_indices) - corrections_applied} potential real motions")
    
    if show_corrections and corrections_applied > 0:
        plot_ultra_minimal_corrections(original_angles, corrected_angles, time_array, 
                                     jump_indices, jump_magnitudes, jump_threshold)
    
    return corrected_angles, {
        'method': 'ultra_minimal',
        'jumps_corrected': corrections_applied,
        'total_correction': total_correction,
        'candidates_preserved': len(jump_indices) - corrections_applied,
        'jump_threshold': jump_threshold
    }

def minimal_trend_preserving_unwrap(angles, show_validation=False):
    """Just use scipy unwrap - trust it!"""
    print("=== UNWRAPPING WITH SCIPY (FORCED) ===")
    
    unwrapped_scipy = np.unwrap(angles)
    
    angle_change = unwrapped_scipy[-1] - unwrapped_scipy[0]
    print(f"Angle change: {angle_change:.2f} rad ({angle_change/(2*np.pi):.2f} rotations)")
    
    return unwrapped_scipy, {'method': 'scipy_unwrap_forced'}

def plot_ultra_minimal_corrections(original_angles, corrected_angles, time_array, 
                                 jump_indices, jump_magnitudes, jump_threshold):
    """Plot results of ultra-minimal correction"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Original angles with jump locations
    axes[0].plot(time_array, original_angles, 'b-', linewidth=0.8, alpha=0.7)
    axes[0].scatter(time_array[jump_indices], original_angles[jump_indices], 
                   color='red', s=20, alpha=0.8, label=f'Potential jumps ({len(jump_indices)})')
    axes[0].set_title(f'Original Angles with Potential Jumps (>{jump_threshold:.1f} rad)')
    axes[0].set_ylabel('Angle (rad)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Corrected angles
    axes[1].plot(time_array, corrected_angles, 'g-', linewidth=0.8)
    axes[1].set_title('After Ultra-Minimal Correction')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].grid(True, alpha=0.3)
    
    # Show difference between original and corrected
    difference = corrected_angles - original_angles
    axes[2].plot(time_array, difference, 'r-', linewidth=1)
    axes[2].set_title('Correction Applied (Corrected - Original)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Correction (rad)')
    axes[2].grid(True, alpha=0.3)
    
    # Mark points where correction was applied
    correction_applied = np.abs(difference) > 0.1
    if np.any(correction_applied):
        correction_indices = np.where(correction_applied)[0]
        axes[2].scatter(time_array[correction_indices], difference[correction_indices], 
                       color='red', s=10, alpha=0.8, label='Corrections applied')
        axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
def unwrap_with_validation(angles_wrapped, max_phase_change_per_sample=1):
    """
    Unwrap angles but validate that no single step is too large
    
    Parameters:
    max_phase_change_per_sample: Maximum rad/sample (default 1.0 rad ≈ 0.12 µm at 1550nm)
    """
    unwrapped = np.unwrap(angles_wrapped)
    
    # Check for suspicious large jumps
    deltas = np.diff(unwrapped)
    large_jumps = np.abs(deltas) > max_phase_change_per_sample
    
    if np.any(large_jumps):
        n_large = np.sum(large_jumps)
        print(f"  ⚠️ WARNING: {n_large} large phase changes detected in this file")
        print(f"      Max: {np.max(np.abs(deltas)):.2f} rad (should be < {max_phase_change_per_sample})")
        
        # Find and correct these jumps
        jump_indices = np.where(large_jumps)[0]
        
        corrected = unwrapped.copy()
        for idx in jump_indices:
            # This jump is probably a 2π error
            jump = deltas[idx]
            n_wraps = np.round(jump / (2*np.pi))
            
            if abs(n_wraps) >= 1:
                # Remove the wrap from this point forward
                correction = -n_wraps * 2*np.pi
                corrected[idx+1:] += correction
                print(f"      Fixed: removed {n_wraps:.0f}×2π at index {idx}")
        
        return corrected
    
    return unwrapped
def extract_angles_from_quadrature_simple(ch1_data, ch2_data, unwrap=True, 
                                         apply_centering=True, show_correction=False):
    """
    Extract angles - simple scipy unwrap, no forcing continuity
    """
    correction_info = None
    
    if apply_centering:
        ch1_corrected, ch2_corrected, correction_info = correct_quadrature_data_simple(
            ch1_data, ch2_data, 
            remove_dc=True,
            show_correction=show_correction
        )
    else:
        ch1_corrected = ch1_data
        ch2_corrected = ch2_data
    
    # Calculate angles using atan2
    raw_angles = np.arctan2(ch2_corrected, ch1_corrected)
    
    if unwrap:
        # Simple scipy unwrap - let it do its job
        angles = unwrap_with_validation(raw_angles)
        
        if correction_info is None:
            correction_info = {}
        correction_info['unwrap_info'] = {
            'method': 'scipy_unwrap',
            'angle_range': angles[-1] - angles[0]
        }
    
    return angles, correction_info

def count_problematic_boundaries(angle_segments, position_jump_threshold_um=0.5):
    """
    How many boundaries have jumps > threshold?
    """
    wavelength_m = 1550e-9
    angle_segments = sorted(angle_segments, key=lambda x: x['start_time'])
    
    large_jumps = 0
    near_2pi_jumps = 0
    
    for i in range(1, len(angle_segments)):
        boundary_delta = angle_segments[i]['angles'][0] - angle_segments[i-1]['angles'][-1]
        position_jump_um = abs(boundary_delta * wavelength_m / (4*np.pi) * 1e6)
        
        if position_jump_um > position_jump_threshold_um:
            large_jumps += 1
            
            # Is it near 2π?
            n_wraps = np.round(boundary_delta / (2*np.pi))
            residual = abs(boundary_delta - n_wraps * 2*np.pi)
            if residual < 0.5 and abs(n_wraps) >= 1:
                near_2pi_jumps += 1
    
    total_boundaries = len(angle_segments) - 1
    print(f"Total boundaries: {total_boundaries}")
    print(f"Large jumps (>{position_jump_threshold_um} µm): {large_jumps} ({large_jumps/total_boundaries*100:.1f}%)")
    print(f"Of those, near 2π multiples: {near_2pi_jumps} ({near_2pi_jumps/total_boundaries*100:.1f}%)")
    return large_jumps, near_2pi_jumps

def delta_theta_segment_stitching_with_boundaries(angle_segments, 
                                                    wrap_threshold=0.05,
                                                    reference_segment=0):
    """
    Improved boundary handling with self-consistency checks
    """
    
    if not angle_segments:
        return np.array([]), np.array([]), np.array([]), {}
    
    print("="*80)
    print("SMART BOUNDARY STITCHING")
    print("="*80)
    
    angle_segments = sorted(angle_segments, key=lambda x: x['start_time'])
    wavelength_m = 1550e-9
    
    all_delta_theta = []
    all_times = []
    boundary_info = []
    
    # First pass: collect all boundary deltas to understand the distribution
    raw_boundary_deltas = []
    for i in range(1, len(angle_segments)):
        boundary_delta = angle_segments[i]['angles'][0] - angle_segments[i-1]['angles'][-1]
        raw_boundary_deltas.append(boundary_delta)
    
    # Normalize all boundaries by removing 2π multiples
    normalised_deltas = []
    for bd in raw_boundary_deltas:
        n = np.round(bd / (2*np.pi))
        normalised_deltas.append(bd - n * 2*np.pi)
    
    # Statistics of normalised deltas
    median_normalised = np.median(normalised_deltas)
    std_normalised = np.std(normalised_deltas)
    
    print(f"Boundary delta statistics (after normalizing to [-π, π]):")
    print(f"  Median: {median_normalised:.4f} rad")
    print(f"  Std dev: {std_normalised:.4f} rad")
    print(f"  Range: [{np.min(normalised_deltas):.4f}, {np.max(normalised_deltas):.4f}] rad")
    print(f"\nWrap detection threshold: {wrap_threshold} rad\n")
    
    # Reference angle
    reference_angle = angle_segments[0]['angles'][0]
    
    # Process segments
    for i, segment in enumerate(angle_segments):
        seg_time = segment['time']
        seg_angles = segment['angles']
        
        print(f"Segment {i}: {len(seg_angles)} points")
        
        # Internal delta-thetas
        seg_delta_theta = np.diff(seg_angles)
        seg_time_midpoints = (seg_time[:-1] + seg_time[1:]) / 2
        
        if i == 0:
            # First segment
            all_delta_theta.extend(seg_delta_theta.tolist())
            all_times.extend(seg_time_midpoints.tolist())
            print(f"  First segment: {len(seg_delta_theta)} internal deltas\n")
            
        else:
            # Calculate boundary delta
            prev_segment = angle_segments[i-1]
            
            boundary_delta_theta = seg_angles[0] - prev_segment['angles'][-1]
            boundary_time = (prev_segment['time'][-1] + seg_time[0]) / 2
            time_gap = seg_time[0] - prev_segment['time'][-1]
            
            # Position jump
            position_jump_um = boundary_delta_theta * wavelength_m / (4*np.pi) * 1e6
            
            # Wrap detection
            n_wraps = np.round(boundary_delta_theta / (2*np.pi))
            wrap_residual = abs(boundary_delta_theta - n_wraps * 2*np.pi)
            
            # Decision logic with multiple checks
            should_correct = False
            reason = ""
            
            if abs(n_wraps) >= 1:
                # Potential wrap candidate
                
                # Check 1: Close to 2π multiple?
                if wrap_residual < wrap_threshold:
                    
                    # Check 2: Would correction make jump smaller?
                    corrected_delta = boundary_delta_theta - n_wraps * 2*np.pi
                    corrected_position_um = corrected_delta * wavelength_m / (4*np.pi) * 1e6
                    
                    if abs(corrected_position_um) < abs(position_jump_um):
                        
                        # Check 3: Is the corrected value consistent with other boundaries?
                        # (i.e., close to the median normalized delta)
                        deviation_from_median = abs(corrected_delta - median_normalised)
                        
                        if deviation_from_median < 3 * std_normalised:
                            should_correct = True
                            reason = (f"Wrap detected: {n_wraps:.0f}×2π removed, "
                                    f"residual={wrap_residual:.3f}, "
                                    f"jump: {position_jump_um:+.2f}→{corrected_position_um:+.2f} µm")
                        else:
                            should_correct = False
                            reason = (f"Looks like wrap but corrected value is outlier "
                                    f"(dev={deviation_from_median:.3f} > 3σ={3*std_normalised:.3f})")
                    else:
                        should_correct = False
                        reason = f"Correction would worsen jump: {position_jump_um:+.2f}→{corrected_position_um:+.2f} µm"
                else:
                    should_correct = False
                    reason = f"Residual too large: {wrap_residual:.3f} > {wrap_threshold}"
            else:
                should_correct = False
                reason = f"Small jump: {position_jump_um:+.2f} µm (no wrap)"
            
            # Apply decision
            if should_correct:
                corrected_boundary_delta = boundary_delta_theta - n_wraps * 2*np.pi
                print(f"  ⚠️  CORRECTED: {reason}")
                
                boundary_info.append({
                    'segment_index': i,
                    'time': boundary_time,
                    'time_gap': time_gap,
                    'original_delta': boundary_delta_theta,
                    'corrected_delta': corrected_boundary_delta,
                    'n_wraps_removed': n_wraps,
                    'wrap_residual': wrap_residual,
                    'original_position_jump_um': position_jump_um,
                    'corrected_position_jump_um': corrected_position_um,
                    'method': 'wrap_corrected',
                    'reason': reason
                })
                
                all_delta_theta.append(corrected_boundary_delta)
                
            else:
                print(f"  ✓ PRESERVED: {reason}")
                
                boundary_info.append({
                    'segment_index': i,
                    'time': boundary_time,
                    'time_gap': time_gap,
                    'boundary_delta': boundary_delta_theta,
                    'position_jump_um': position_jump_um,
                    'method': 'preserved',
                    'reason': reason
                })
                
                all_delta_theta.append(boundary_delta_theta)
            
            all_times.append(boundary_time)
            
            # Add segment internal deltas
            all_delta_theta.extend(seg_delta_theta.tolist())
            all_times.extend(seg_time_midpoints.tolist())
            print()
    
    # Convert to arrays and sort
    all_delta_theta = np.array(all_delta_theta)
    all_times = np.array(all_times)
    
    sort_indices = np.argsort(all_times)
    all_delta_theta = all_delta_theta[sort_indices]
    all_times = all_times[sort_indices]
    
    # Reconstruct angles
    final_angles = np.zeros(len(all_delta_theta) + 1)
    final_angles[0] = reference_angle
    final_angles[1:] = reference_angle + np.cumsum(all_delta_theta)
    
    final_time = np.zeros(len(final_angles))
    final_time[0] = angle_segments[0]['time'][0]
    final_time[1:] = all_times
    
    # Summary
    total_change = final_angles[-1] - final_angles[0]
    n_corrected = sum(1 for b in boundary_info if b['method'] == 'wrap_corrected')
    n_preserved = sum(1 for b in boundary_info if b['method'] == 'preserved')
    
    print("="*80)
    print("STITCHING COMPLETE")
    print("="*80)
    print(f"Total angle change: {total_change:.2f} rad ({total_change/(2*np.pi):.1f} rotations)")
    print(f"Total points: {len(final_angles)}")
    print(f"Boundaries: {len(boundary_info)}")
    print(f"  Corrected: {n_corrected} ({n_corrected/len(boundary_info)*100:.1f}%)")
    print(f"  Preserved: {n_preserved} ({n_preserved/len(boundary_info)*100:.1f}%)")
    print("="*80)
    
    stitch_info = {
        'method': 'smart_boundary_detection',
        'num_segments': len(angle_segments),
        'boundary_info': boundary_info,
        'reference_angle': reference_angle,
        'wrap_threshold': wrap_threshold,
        'boundary_statistics': {
            'median_normalised_delta': median_normalised,
            'std_normalised_delta': std_normalised
        }
    }
    
    return final_angles, final_time, all_delta_theta, stitch_info

def calculate_position_from_delta_theta(delta_theta, time_array, wavelength_nm=1550, 
                                       initial_position=0.0):
    """
    Calculate position directly from delta-theta values (most efficient?)
    
    Pipeline: Δθ -> Δx -> integrate -> x(t)
    
    Parameters:
    -----------
    delta_theta : np.ndarray
        Array of phase differences in radians (length N-1)
    time_array : np.ndarray
        Time array (length N)
    wavelength_nm : float
        Laser wavelength in nanometers
    initial_position : float
        Starting position in meters (default: 0.0)
    
    Returns:
    --------
    position : np.ndarray
        Position array in meters (length N)
    velocity : np.ndarray
        Velocity array in m/s (length N)
    """
    
    wavelength_m = wavelength_nm * 1e-9
    conversion_factor = wavelength_m / (4 * np.pi)
    
    print(f"=== DIRECT POSITION CALCULATION FROM Δθ ===")
    print(f"Wavelength: {wavelength_nm} nm")
    print(f"Conversion factor: λ/(4π) = {conversion_factor*1e9:.6f} nm/rad")
    print(f"Input: {len(delta_theta)} Δθ values for {len(time_array)} time points")
    
    # Verify array sizes match
    if len(delta_theta) != len(time_array) - 1:
        raise ValueError(f"Delta-theta length ({len(delta_theta)}) should be "
                        f"time_array length - 1 ({len(time_array) - 1})")
    
    # Step 1: Δθ -> Δx (convert phase changes to position changes)
    delta_x = delta_theta * conversion_factor
    
    print(f"Step 1: Converted Δθ to Δx")
    print(f"        Δx range: {np.min(delta_x)*1e9:.4f} to {np.max(delta_x)*1e9:.4f} nm")
    print(f"        Mean |Δx|: {np.mean(np.abs(delta_x))*1e9:.4f} nm")
    
    # Step 2: Integrate to get position x(t) = x₀ + Σ(Δx)
    position = np.zeros(len(time_array))
    position[0] = initial_position
    position[1:] = initial_position + np.cumsum(delta_x)
    
    print(f"Step 2: Integrated to position x(t)")
    
    # Step 3: Calculate velocity
    velocity = np.gradient(position, time_array)
    
    # Statistics and verification
    total_displacement = position[-1] - position[0]
    total_phase_change = np.sum(delta_theta)
    total_rotations = total_phase_change / (2 * np.pi)
    expected_displacement = total_rotations * (wavelength_m / 2)
    
    print(f"\n=== RESULTS ===")
    print(f"Total Δθ: {total_phase_change:.4f} rad = {total_rotations:.6f} rotations")
    print(f"Expected displacement: {expected_displacement*1e6:.6f} μm")
    print(f"Actual displacement: {total_displacement*1e6:.6f} μm")
    print(f"Error: {abs(expected_displacement - total_displacement)*1e9:.4f} nm")
    
    print(f"\nPosition statistics:")
    print(f"  Range: {np.min(position)*1e6:.4f} to {np.max(position)*1e6:.4f} μm")
    print(f"  Std dev: {np.std(position)*1e6:.4f} μm")
    print(f"  Mean velocity: {np.mean(velocity)*1e9:.4f} nm/s")
    print(f"  Velocity std dev: {np.std(velocity)*1e9:.4f} nm/s")
    
    # Check for any suspicious large steps
    large_steps = np.abs(delta_x) > 3.0 * conversion_factor  # > 3 rad equivalent
    if np.any(large_steps):
        n_large = np.sum(large_steps)
        max_step = np.max(np.abs(delta_x))
        print(f"\nINFO: {n_large} large position steps detected")
        print(f"      Max step: {max_step*1e9:.2f} nm")
    
    return position, velocity

def process_file_pairs_to_angles(channel_data, downsample_factor=2, 
                                            apply_correction=True, 
                                            show_first_correction=True, 
                                            max_correction_samples=50000,
                                            memory_efficient=True):
    """
    Process each file pair to extract angles
    
    Parameters:
    channel_data: Dictionary containing data for all channels
    downsample_factor: Factor to downsample data
    apply_correction: Whether to apply quadrature correction
    show_first_correction: Whether to show correction plot for first file
    max_correction_samples: Maximum samples for correction fitting
    memory_efficient: Whether to use memory-efficient processing
    
    Returns:
    angle_segments: List of dictionaries containing angle data for each file
    """
    channel_names = list(channel_data.keys())
    
    if len(channel_names) < 2:
        raise ValueError("Need at least 2 channels for quadrature analysis")
    
    ch1_name = channel_names[0]
    ch2_name = channel_names[1]
    
    ch1_segments = channel_data[ch1_name]['data_segments']
    ch2_segments = channel_data[ch2_name]['data_segments']
    time_segments = channel_data[ch1_name]['time_segments']
    
    # Ensure we have the same number of files for both channels
    min_files = min(len(ch1_segments), len(ch2_segments))
    
    angle_segments = []
    correction_stats = []  # Store correction information for analysis
    
    for i in range(min_files):
        print(f"Processing file pair {i+1}/{min_files}...")
        
        ch1_data = ch1_segments[i]
        ch2_data = ch2_segments[i]
        time_data = time_segments[i]
        
        # Ensure both channels have same length
        min_length = min(len(ch1_data), len(ch2_data), len(time_data))
        ch1_data = ch1_data[:min_length]
        ch2_data = ch2_data[:min_length]
        time_data = time_data[:min_length]
        
        print(f"  File {i}: Processing {min_length} samples...")
        
        # Apply initial downsampling to manage memory
        if memory_efficient and downsample_factor > 1:
            # Downsample early to reduce memory footprint
            new_length = int(np.ceil(len(ch1_data) / downsample_factor))
            downsampled_ch1 = np.zeros(new_length)
            downsampled_ch2 = np.zeros(new_length)
            downsampled_time = np.zeros(new_length)
            
            for j in range(new_length):
                start_idx = j * downsample_factor
                end_idx = min((j + 1) * downsample_factor, len(ch1_data))
                
                if start_idx < end_idx:
                    downsampled_ch1[j] = np.mean(ch1_data[start_idx:end_idx])
                    downsampled_ch2[j] = np.mean(ch2_data[start_idx:end_idx])
                    downsampled_time[j] = np.mean(time_data[start_idx:end_idx])
            
            ch1_data = downsampled_ch1
            ch2_data = downsampled_ch2
            time_data = downsampled_time
            
            print(f"  File {i}: Downsampled to {len(ch1_data)} samples (factor: {downsample_factor})")
        
        # Apply centering correction
        show_correction = show_first_correction and (i == 0)
        if apply_correction:
            ch1_corrected, ch2_corrected, correction_info = correct_quadrature_data_simple(
                ch1_data, ch2_data, 
                remove_dc=True,
                show_correction=show_correction,
                method='ellipse'
            )
        else:
            ch1_corrected = ch1_data
            ch2_corrected = ch2_data
            correction_info = {}
        
        # Calculate angles using atan2 (gives [-π, π])
        raw_angles = np.arctan2(ch2_corrected, ch1_corrected)
        
        # Unwrap angles using scipy (this is the key fix!)
        unwrapped_angles = np.unwrap(raw_angles)
        
        # Statistics
        angle_change = unwrapped_angles[-1] - unwrapped_angles[0]
        angle_range = np.max(unwrapped_angles) - np.min(unwrapped_angles)
        n_rotations = angle_change / (2 * np.pi)
        
        print(f"  File {i} angle extraction:")
        print(f"    Raw angles: [{np.min(raw_angles):.2f}, {np.max(raw_angles):.2f}] rad")
        print(f"    After unwrap: [{np.min(unwrapped_angles):.2f}, {np.max(unwrapped_angles):.2f}] rad")
        print(f"    Net change: {angle_change:.2f} rad ({n_rotations:.2f} rotations)")
        print(f"    Total range: {angle_range:.2f} rad")
        
        # Store unwrap info
        if correction_info is None:
            correction_info = {}
        correction_info['unwrap_info'] = {
            'method': 'scipy_unwrap',
            'angle_change': angle_change,
            'n_rotations': n_rotations,
            'angle_range': angle_range
        }
        
        correction_stats.append(correction_info)
        
        # Create absolute time array for this segment
        absolute_time = time_data + absolute_start_times[i]
        
        angle_segments.append({
            'time': absolute_time,
            'angles': unwrapped_angles,
            'start_time': absolute_start_times[i],
            'end_time': absolute_start_times[i] + file_durations[i],
            'index': i,
            'ch1_name': ch1_name,
            'ch2_name': ch2_name,
            'correction_info': correction_info
        })
        
        print(f"  ✓ File {i} complete: {len(unwrapped_angles)} angle points\n")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"ANGLE EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(angle_segments)}")
    
    total_rotations = sum(c.get('unwrap_info', {}).get('n_rotations', 0) 
                         for c in correction_stats)
    print(f"Total rotations across all files: {total_rotations:.1f}")
    
    avg_rotations = total_rotations / len(correction_stats) if correction_stats else 0
    print(f"Average rotations per file: {avg_rotations:.2f}")
    
    print(f"{'='*80}\n")
    
    return angle_segments

def plot_simple_correction_results(ch1_orig, ch2_orig, ch1_corrected, ch2_corrected, correction_info):
    """Plot simple correction results (centering only)"""
    plot_downsample = max(1, len(ch1_orig) // 10000)
    plot_indices = np.arange(0, len(ch1_orig), plot_downsample)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before correction
    axes[0].scatter(ch1_orig[plot_indices], ch2_orig[plot_indices], s=1, alpha=0.6)
    axes[0].set_title('Before Correction')
    axes[0].set_xlabel('Channel 1 (V)')
    axes[0].set_ylabel('Channel 2 (V)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # After correction (centered, ellipse preserved)
    axes[1].scatter(ch1_corrected[plot_indices], ch2_corrected[plot_indices], s=1, alpha=0.6)
    axes[1].set_title('After Centering (Ellipse Preserved)')
    axes[1].set_xlabel('Channel 1 (centered)')
    axes[1].set_ylabel('Channel 2 (centered)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_conservative_quadrature_plots(angle_segments, show_plots=True, max_plots=6):
    """
    Create quadrature plots for individual file pairs
    """
    if not show_plots:
        return
        
    num_plots = min(len(angle_segments), max_plots)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))
    
    if num_plots > 0:
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if num_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i in range(num_plots):
            segment = angle_segments[i]
            
            # Get the original voltage data for this file
            ch1_name = segment['ch1_name']
            ch2_name = segment['ch2_name']
            file_idx = segment['index']
            
            ch1_data = channel_data[ch1_name]['data_segments'][file_idx]
            ch2_data = channel_data[ch2_name]['data_segments'][file_idx]
            
            # Apply the same corrections that were used
            correction_info = segment.get('correction_info', {})
            if correction_info and 'ch1_offset' in correction_info:
                ch1_data = ch1_data - correction_info['ch1_offset']
                ch2_data = ch2_data - correction_info['ch2_offset']
            
            # Ensure same length and downsample for plotting
            min_length = min(len(ch1_data), len(ch2_data))
            plot_step = max(1, min_length // 5000)
            plot_indices = np.arange(0, min_length, plot_step)
            
            ch1_plot = ch1_data[plot_indices]
            ch2_plot = ch2_data[plot_indices]
            
            # Get corresponding angles
            if len(segment['angles']) == min_length:
                angles_plot = segment['angles'][plot_indices]
            else:
                original_time = channel_data[ch1_name]['time_segments'][file_idx][:min_length]
                angles_plot = np.interp(original_time[plot_indices], segment['time'], segment['angles'])
            
            # Ensure consistent lengths
            min_plot_length = min(len(ch1_plot), len(ch2_plot), len(angles_plot))
            ch1_plot = ch1_plot[:min_plot_length]
            ch2_plot = ch2_plot[:min_plot_length]
            angles_plot = angles_plot[:min_plot_length]
            
            # Create quadrature plot
            ax = axes[i] if num_plots > 1 else axes[0]
            scatter = ax.scatter(ch1_plot, ch2_plot, 
                               s=1, alpha=0.6, c=angles_plot, 
                               cmap='hsv', vmin=-np.pi, vmax=np.pi)
            
            ax.set_xlabel(f"{ch1_name} Voltage (V)")
            ax.set_ylabel(f"{ch2_name} Voltage (V)")
            
            # Title
            title = f"File {file_idx} Quadrature)"
            if correction_info and 'unwrap_info' in correction_info:
                unwrap_info = correction_info['unwrap_info']
                if isinstance(unwrap_info, dict) and 'jumps_corrected' in unwrap_info:
                    corrected = unwrap_info['jumps_corrected']
                    preserved = unwrap_info.get('candidates_preserved', 0)
                    title += f"\nCorrected: {corrected}, Preserved: {preserved}"
            
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Angle (rad)')
        
        # Hide unused subplots
        for i in range(num_plots, len(axes) if isinstance(axes, np.ndarray) else 1):
            if isinstance(axes, np.ndarray) and i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

def visualise_conservative_results(angle_segments, stitched_angles, stitched_time, 
                                 position, velocity, phase_changes, plot_downsample=10):
    """
    Visualise the processing results
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    
    # Memory-efficient plotting
    plot_step = max(1, len(stitched_time) // 50000)
    
    # Plot 1: Individual angle segments
    ax1 = axes[0, 0]
    ax1.set_title("Individual Angle Segments")
    colors = plt.cm.tab10
    for i, segment in enumerate(angle_segments[:min(10, len(angle_segments))]):
        color = colors(i % 10)
        seg_plot_step = max(1, len(segment['time']) // 1000)
        ax1.plot(segment['time'][::seg_plot_step], segment['angles'][::seg_plot_step], 
                color=color, alpha=0.7, linewidth=1, label=f"File {segment['index']}")
    
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle (rad)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final stitched angles
    ax2 = axes[0, 1]
    ax2.set_title("Stitched Angles")
    ax2.plot(stitched_time[::plot_step], stitched_angles[::plot_step], 'b-', linewidth=0.8)
    
    # Mark segment boundaries
    for segment in angle_segments:
        ax2.axvline(x=segment['start_time'], color='r', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (rad)")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Angle differences (showing preserved motion)
    ax3 = axes[1, 0]
    angle_diffs = np.diff(stitched_angles)
    ax3.plot(stitched_time[1::plot_step], angle_diffs[::plot_step], 'g-', linewidth=0.5, alpha=0.7)
    ax3.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, label='±5 rad (info threshold)')
    ax3.axhline(y=-5.0, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=10.0, color='red', linestyle='--', alpha=0.5, label='±10 rad (correction threshold)')
    ax3.axhline(y=-10.0, color='red', linestyle='--', alpha=0.5)
    ax3.set_title("Phase Differences (Large Changes Preserved)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Δθ (rad)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Interferometric position
    ax4 = axes[1, 1]
    ax4.set_title("Interferometric Position")
    ax4.plot(stitched_time[::plot_step], position[::plot_step] * 1e6, 'g-', linewidth=0.8)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position (μm)")
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity
    ax5 = axes[2, 0]
    ax5.set_title("Velocity (All Motion Preserved)")
    ax5.plot(stitched_time[::plot_step], velocity[::plot_step] * 1e6, 'r-', linewidth=0.8)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Velocity (μm/s)")
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Position vs angle (phase space)
    ax6 = axes[2, 1]
    ax6.set_title("Phase Space (Position vs Angle)")
    scatter = ax6.scatter(stitched_angles[::plot_step*5], position[::plot_step*5] * 1e6, 
                         s=1, alpha=0.6, c=stitched_time[::plot_step*5], cmap='viridis')
    ax6.set_xlabel("Angle (rad)")
    ax6.set_ylabel("Position (μm)")
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time (s)')
    
    plt.tight_layout()
    plt.show()

def create_position_only_plot(stitched_time, position):
    """
    Create a clean position vs time plot with 0.1 micrometer minor grid lines
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot position
    plot_step = max(1, len(stitched_time) // 50000)  # Downsample for plotting
    ax.plot(stitched_time[::plot_step], position[::plot_step] * 1e6, 'b-', linewidth=0.8)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (µm)')
    ax.set_title('Conservative Interferometric Position Measurement (1550 nm laser)')
    
    # Set up major and minor grid lines
    # Major grid (default spacing)
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    
    # Minor grid at 0.1 µm intervals on y-axis
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # 0.1 µm minor ticks
    ax.grid(True, which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Optional major y-axis ticks to nice intervals (e.g., 0.5 or 1.0 µm)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 0.5 µm major ticks
    # ax.yaxis.set_major_locator(MultipleLocator(1.0))  # 1.0 µm major ticks
    
    # Add measurement statistics
    total_disp = (position[-1] - position[0]) * 1e6
    duration_hrs = (stitched_time[-1] - stitched_time[0]) / 3600
    max_pos = np.max(np.abs(position)) * 1e6
    std_pos = np.std(position) * 1e6

    stats_text = (f'Duration: {duration_hrs:.2f} hours\n'
                  f'Total displacement: {total_disp:.3f} μm\n'
                  f'Maximum position: {max_pos:.3f} μm\n'
                  f'Position std dev: {std_pos:.3f} μm\n'
                  f'Data points: {len(position):,}')

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN CONVERSION EXECUTION - CONSERVATIVE VERSION
# =============================================================================

print("="*80)
print("CONSERVATIVE INTERFEROMETRY PROCESSING")
print("="*80)

# Sample rates and downsampling
downsampling_factor = 10  # Original
original_sample_rate = 1000
downsampled_sample_rate = original_sample_rate / downsampling_factor

print(f"Using downsampling factor of {downsampling_factor} (effective sample rate: {downsampled_sample_rate} Hz)")

# Configuration for CONSERVATIVE processing
SHOW_FIRST_CORRECTION = True
MAX_CORRECTION_SAMPLES = 10000

print(f"\nSIMPLIFIED PROCESSING SETTINGS:")
print(f"• Centering: Yes (DC offset removal)")
print(f"• Require 2π multiples: Yes (within 0.1 tolerance)")
print(f"• Boundary correction threshold: 15+ rad")

# Process with simplified approach
print("Processing file pairs to extract angles...")
angle_segments = process_file_pairs_to_angles(
    channel_data, 
    downsample_factor=downsampling_factor,
    apply_correction=True,
    show_first_correction=SHOW_FIRST_CORRECTION,
    max_correction_samples=50000,
    memory_efficient=True
    )

# Create conservative quadrature plots
print("Creating conservative quadrature plots...")
create_conservative_quadrature_plots(angle_segments, show_plots=True, max_plots=6)

## Stitching and angle change bit
print("Stitching angle segments with new approach...")
stitched_angles, stitched_time, stitched_delta_theta, stitch_info = delta_theta_segment_stitching_with_boundaries(angle_segments)

print(f"Final stitched angle data: {len(stitched_angles)} points")
print(f"Angle time range: {stitched_time[0]:.2f}s to {stitched_time[-1]:.2f}s")
print(f"Total angle change: {stitched_angles[-1] - stitched_angles[0]:.2f} rad")
print(f"Full rotations: {(stitched_angles[-1] - stitched_angles[0])/(2*np.pi):.2f}")

count_problematic_boundaries(angle_segments, position_jump_threshold_um=0.5)        
# Calculate interferometric position
print("Calculating interferometric position...")
LASER_WAVELENGTH_NM = 1550

position, velocity = calculate_position_from_delta_theta(
   stitched_delta_theta, stitched_time, 
    wavelength_nm=LASER_WAVELENGTH_NM 
)

phase_changes = np.sum(stitched_delta_theta)  # Total phase change

# Conservative visualisation
print("Generating conservative visualisation...")
visualise_conservative_results(angle_segments, stitched_angles, stitched_time, 
                             position, velocity, phase_changes)

# Create clean position summary plot
print("Creating position summary plot...")
create_position_only_plot(stitched_time, position,)

print("\n" + "="*80)
print("CONSERVATIVE INTERFEROMETRY SUMMARY")
print("="*80)
print(f"Number of file pairs processed: {len(angle_segments)}")
print(f"Total time span: {stitched_time[-1] - stitched_time[0]:.2f} seconds ({(stitched_time[-1] - stitched_time[0])/3600:.2f} hours)")
print(f"Total angular displacement: {stitched_angles[-1] - stitched_angles[0]:.2f} rad")
print(f"Total rotations: {(stitched_angles[-1] - stitched_angles[0])/(2*np.pi):.2f}")
print(f"Laser wavelength: {LASER_WAVELENGTH_NM} nm")
print(f"Position conversion factor: {LASER_WAVELENGTH_NM/(4*np.pi):.4f} nm/radian")
print(f"Final interferometric position: {position[-1]*1e6:.3f} μm")
print(f"Total position displacement: {(position[-1] - position[0])*1e6:.3f} μm")
print(f"Maximum position excursion: {np.max(np.abs(position))*1e6:.3f} μm")
print(f"Position standard deviation: {np.std(position)*1e6:.3f} μm")
print(f"Mean velocity: {np.mean(velocity)*1e6:.4f} μm/s")
print(f"Velocity standard deviation: {np.std(velocity)*1e6:.4f} μm/s")
print(f"Effective sample rate after processing: {downsampled_sample_rate:.1f} Hz")

# Calculate and display conservative processing statistics
total_jumps_corrected = sum(
    seg.get('correction_info', {}).get('unwrap_info', {}).get('jumps_corrected', 0) 
    for seg in angle_segments
)
total_jumps_preserved = sum(
    seg.get('correction_info', {}).get('unwrap_info', {}).get('candidates_preserved', 0) 
    for seg in angle_segments
)

print(f"\nCONSERVATIVE PROCESSING STATISTICS:")
print(f"Total phase jumps corrected: {total_jumps_corrected}")
print(f"Total phase jumps preserved: {total_jumps_preserved}")
if total_jumps_corrected + total_jumps_preserved > 0:
    conservation_ratio = total_jumps_preserved / (total_jumps_corrected + total_jumps_preserved)
    print(f"Conservation ratio: {conservation_ratio:.1%} (higher is more conservative)")
else:
    print("Conservation ratio: N/A (no jumps detected)")

print(f"\nDATA INTEGRITY:")
print("✓ No phase offset corrections applied (artifact prevention)")
print("✓ Minimal algorithmic intervention")
print("✓ Natural trends and drifts preserved")
print("✓ Ultra-gentle segment stitching")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("Data ready for binning and downsampling")
print("="*80)
#==============================================================================
#(SAVE THE DAMN THING!)
# Save angle segments first (with datetime info)
print("\n" + "="*80)
print("SAVING INTERMEDIATE DATA")
print("="*80)

#processor.save_angle_segments(angle_segments, file_datetimes)

#Save full trace (with datetime info)
processor.save_full_trace(
    stitched_time, stitched_angles, position, velocity, 
    phase_changes, angle_segments, stitch_info,
    file_datetimes=file_datetimes,  # Make sure this is passed!
    base_directory=base_dir,
    downsampling_factor=downsampling_factor,
    laser_wavelength_nm=1550,
    notes="N/A"
)

print("="*80)
print("FULL POSITION TRACE SAVED")
print("="*80)