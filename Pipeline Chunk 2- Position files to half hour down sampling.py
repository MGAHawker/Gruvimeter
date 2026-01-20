# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:08:52 2025

@author: migsh
"""
import numpy as np
import scipy as scipy
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
from scipy.io import savemat, loadmat
from datetime import timedelta
import datetime

# Define your colour scheme
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOURS)

# =============================================================================
# HELPER FUNCTION TO ENSURE ARRAYS
# =============================================================================

def ensure_1d_array(arr):
    """Ensure input is a 1D numpy array, even if it's a scalar or multi-dimensional"""
    if arr is None:
        return None
    arr = np.atleast_1d(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr

# =============================================================================
# DATA SAVING/LOADING FUNCTIONS
# =============================================================================

def save_processed_data(data_dict, filepath, compress=True):
    """Save processed data to disk with optional compression"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.mat':
        mat_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, (datetime.datetime, np.datetime64)):
                mat_dict[key] = str(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], datetime.datetime):
                mat_dict[key] = [str(dt) for dt in value]
            else:
                mat_dict[key] = value
        
        savemat(filepath, mat_dict, do_compression=compress, oned_as='column')
        print(f"Saved to {filepath} (MATLAB format)")
        
    elif filepath.suffix == '.npz':
        if compress:
            np.savez_compressed(filepath, **data_dict)
        else:
            np.savez(filepath, **data_dict)
        print(f"Saved to {filepath} (NumPy format)")
    else:
        raise ValueError("Filepath must end with .mat or .npz")
    
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.2f} MB")

def load_processed_data(filepath):
    """Load processed data from disk"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.mat':
        loaded = loadmat(filepath, squeeze_me=True, struct_as_record=False)
        data_dict = {key: value for key, value in loaded.items() 
                     if not key.startswith('__')}
        print(f"Loaded from {filepath} (MATLAB format)")
        
    elif filepath.suffix == '.npz':
        loaded = np.load(filepath, allow_pickle=True)
        data_dict = {key: loaded[key] for key in loaded.files}
        print(f"Loaded from {filepath} (NumPy format)")
    else:
        raise ValueError("Filepath must end with .mat or .npz")
    
    print(f"Loaded data contains {len(data_dict)} fields:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    return data_dict

# =============================================================================
# LOADING FUNCTIONS WITH DATETIME RECONSTRUCTION
# =============================================================================

def load_full_position_trace(filepath):
    """
    Load full position trace and reconstruct all information including datetimes
    Handles both .npz and .mat formats
    
    Returns:
    Dictionary with all position trace data ready for analysis
    """
    data = load_processed_data(filepath)
    
    # Flatten arrays if needed (MATLAB saves 1D as columns)
    def flatten_if_needed(arr):
        if isinstance(arr, np.ndarray):
            return arr.flatten() if arr.ndim > 1 else arr
        return arr
    
    # Reconstruct angle_segments structure
    angle_segments = []
    for i in range(len(data['segment_start_times'])):
        angle_segments.append({
            'start_time': float(flatten_if_needed(data['segment_start_times'])[i]),
            'end_time': float(flatten_if_needed(data['segment_end_times'])[i]),
            'index': int(flatten_if_needed(data['segment_indices'])[i])
        })
    
    # Reconstruct metadata from flattened structure (for .mat files)
    metadata = {}
    for key, value in data.items():
        if key.startswith('meta_'):
            clean_key = key[5:]
            metadata[clean_key] = value
    
    # If no metadata found, try extracting as object array (for .npz files)
    if not metadata and 'metadata' in data:
        if isinstance(data['metadata'], np.ndarray):
            metadata = data['metadata'].item()
        else:
            metadata = data['metadata']
    
    # Reconstruct datetime information if present
    file_datetimes = None
    if 'file_datetimes_str' in data and 'file_paths' in data:
        file_datetimes = []
        dt_strings = data['file_datetimes_str']
        paths = data['file_paths']
        
        # Handle both single strings and arrays
        if isinstance(dt_strings, str):
            dt_strings = [dt_strings]
            paths = [paths]
        
        for path_str, dt_str in zip(paths, dt_strings):
            dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
            file_datetimes.append((path_str, dt))
    
    # Extract absolute time reference if available
    absolute_time_reference = None
    if file_datetimes and len(file_datetimes) > 0:
        # Use the first file's datetime as the absolute reference
        absolute_time_reference = file_datetimes[0][1]
    
    # Use ensure_1d_array to guarantee all arrays are 1D
    return {
        'time': ensure_1d_array(flatten_if_needed(data['time'])),
        'angles': ensure_1d_array(flatten_if_needed(data['angles'])),
        'position': ensure_1d_array(flatten_if_needed(data['position'])),
        'velocity': ensure_1d_array(flatten_if_needed(data['velocity'])),
        'phase_changes': flatten_if_needed(data['phase_changes']),
        'angle_segments': angle_segments,
        'metadata': metadata,
        'file_datetimes': file_datetimes,
        'absolute_time_reference': absolute_time_reference
    }

# =============================================================================
# SEQUENTIAL BINNING WITH AUTOMATIC STITCHING
# =============================================================================

class SequentialBinner:
    """
    Handles sequential loading and binning of position trace files
    with automatic stitching of leftovers between files
    """
    
    def __init__(self, bin_duration_minutes=10, overlap_window_seconds=60):
        """
        Parameters:
        bin_duration_minutes: Size of each bin in minutes
        overlap_window_seconds: Window for tapering/stitching between files
        """
        self.bin_duration = bin_duration_minutes * 60  # Convert to seconds
        self.overlap_window = overlap_window_seconds
        
        # Storage for leftover data from previous file
        self.leftover_time = None
        self.leftover_position = None
        self.leftover_velocity = None
        self.leftover_angles = None
        
        # Storage for binned results
        self.binned_time = []
        self.binned_position = []
        self.binned_velocity = []
        self.binned_position_std = []
        self.binned_velocity_std = []
        self.binned_counts = []
        
        # Absolute time tracking
        self.absolute_time_reference = None
        self.last_file_end_datetime = None
        
        print(f"Initialized SequentialBinner:")
        print(f"  Bin duration: {bin_duration_minutes} minutes")
        print(f"  Overlap window: {overlap_window_seconds} seconds")
        
    def stitch_with_previous_leftover(self, new_time, new_position, new_velocity, 
                              new_angles, file_datetime):
      """
      Stitch new data with leftover from previous file.
      - Offsets position and velocity to ensure continuity across files
      - Fills any time gaps with synthetic bridge data + light noise
      """

      # Ensure all inputs are 1D arrays
      new_time = ensure_1d_array(new_time)
      new_position = ensure_1d_array(new_position)
      new_velocity = ensure_1d_array(new_velocity)
      new_angles = ensure_1d_array(new_angles)

      # First file case
      if self.leftover_time is None:
          print("  First file - no stitching needed")

          if self.absolute_time_reference is None and file_datetime is not None:
              self.absolute_time_reference = file_datetime
              print(f"  Set absolute time reference: {file_datetime}")

          return new_time, new_position, new_velocity, new_angles

      print(f"  Stitching with {len(self.leftover_time)} leftover points")

      # Compute time offset
      time_gap = 0.0
      if file_datetime is not None and self.last_file_end_datetime is not None:
          time_gap = (file_datetime - self.last_file_end_datetime).total_seconds()
          print(f"  Time gap between files: {time_gap:.2f} seconds")

          if time_gap < 0:
              print("  WARNING: Negative time gap detected – overlapping files! Check datetimes.")
          elif time_gap > 3600:
              print(f"  WARNING: Large time gap ({time_gap/3600:.2f} hours)")
      
          time_offset = self.leftover_time[-1] + time_gap
      else:
          print("  Warning: No datetime info, assuming continuous sequence")
          time_offset = self.leftover_time[-1]

      # Offset the new time array so it continues seamlessly
      offset_new_time = new_time - new_time[0] + time_offset

      # NEW: Offset position and velocity to ensure continuity
      position_offset = self.leftover_position[-1] - new_position[0]
      offset_new_position = new_position + position_offset
      print(f"  Applied position offset: {position_offset*1e6:.3f} μm")

      velocity_offset = self.leftover_velocity[-1] - new_velocity[0]
      offset_new_velocity = new_velocity + velocity_offset
      print(f"  Applied velocity offset: {velocity_offset*1e6:.3f} μm/s")

      # If the new file starts before the old one ends, warn and truncate
      if offset_new_time[0] < self.leftover_time[-1]:
          print("  WARNING: Overlapping time detected, truncating new data to prevent duplication")
          valid_mask = offset_new_time > self.leftover_time[-1]
          offset_new_time = offset_new_time[valid_mask]
          offset_new_position = offset_new_position[valid_mask]
          offset_new_velocity = offset_new_velocity[valid_mask]
          new_angles = new_angles[valid_mask]

      # Handle positive time gaps (non-overlapping data)
      gap = offset_new_time[0] - self.leftover_time[-1]
      if gap > 0:
          print(f"  Filling time gap of {gap:.2f} seconds with synthetic bridge")

          # Generate synthetic bridge segment
          last_pos = self.leftover_position[-1]
          first_pos = offset_new_position[0]  # Use offset position here

          n_bridge = int(min(200, max(20, gap / 0.5)))  # approx 2 Hz, capped
          bridge_time = np.linspace(self.leftover_time[-1], offset_new_time[0], n_bridge)

          # Small Gaussian noise scaled to local variance
          local_std = np.std(self.leftover_position[-min(100, len(self.leftover_position)):])
          bridge_position = np.linspace(last_pos, first_pos, n_bridge) + np.random.normal(
              0, local_std * 0.05, n_bridge
          )
          bridge_velocity = np.linspace(self.leftover_velocity[-1], offset_new_velocity[0], n_bridge)
          bridge_angles = np.linspace(self.leftover_angles[-1], new_angles[0], n_bridge)

          # Stitch everything together
          stitched_time = np.concatenate([self.leftover_time, bridge_time, offset_new_time])
          stitched_position = np.concatenate([self.leftover_position, bridge_position, offset_new_position])
          stitched_velocity = np.concatenate([self.leftover_velocity, bridge_velocity, offset_new_velocity])
          stitched_angles = np.concatenate([self.leftover_angles, bridge_angles, new_angles])

          print(f"  Inserted {n_bridge} synthetic bridge points")

      else:
          # No meaningful gap – just concatenate (already continuous)
          stitched_time = np.concatenate([self.leftover_time, offset_new_time])
          stitched_position = np.concatenate([self.leftover_position, offset_new_position])
          stitched_velocity = np.concatenate([self.leftover_velocity, offset_new_velocity])
          stitched_angles = np.concatenate([self.leftover_angles, new_angles])
          print("  Continuous sequence – direct concatenation")

      print(f"  Stitched data: {len(stitched_time)} total points")
      return stitched_time, stitched_position, stitched_velocity, stitched_angles   
    

    
    def bin_data(self, time, position, velocity, angles, min_points_per_bin=10):
        """
        Bin the data into fixed-duration bins
        
        Returns:
        leftover_time, leftover_position, leftover_velocity, leftover_angles
        (data that didn't fill a complete bin)
        """
        # Ensure all inputs are 1D arrays
        time = ensure_1d_array(time)
        position = ensure_1d_array(position)
        velocity = ensure_1d_array(velocity)
        angles = ensure_1d_array(angles)
        
        start_time = time[0]
        end_time = time[-1]
        total_duration = end_time - start_time
        
        # Calculate number of complete bins
        n_complete_bins = int(total_duration / self.bin_duration)
        
        print(f"  Data duration: {total_duration/3600:.2f} hours")
        print(f"  Complete bins: {n_complete_bins}")
        
        if n_complete_bins == 0:
            print(f"  Insufficient data for a complete bin - storing as leftover")
            return time, position, velocity, angles
        
        # Process each complete bin
        bins_added = 0
        for i in range(n_complete_bins):
            bin_start = start_time + i * self.bin_duration
            bin_end = bin_start + self.bin_duration
            
            # Create mask for this bin
            bin_mask = (time >= bin_start) & (time < bin_end)
            bin_count = np.sum(bin_mask)
            
            if bin_count >= min_points_per_bin:
                # Extract data for this bin
                bin_time = time[bin_mask]
                bin_position = position[bin_mask]
                bin_velocity = velocity[bin_mask]
                
                # Calculate statistics
                self.binned_time.append(np.mean(bin_time))
                self.binned_position.append(np.mean(bin_position))
                self.binned_velocity.append(np.mean(bin_velocity))
                self.binned_position_std.append(np.std(bin_position))
                self.binned_velocity_std.append(np.std(bin_velocity))
                self.binned_counts.append(bin_count)
                
                bins_added += 1
        
        print(f"  Added {bins_added} bins to results")
        
        # Find leftover data (after last complete bin)
        leftover_start = start_time + n_complete_bins * self.bin_duration
        leftover_mask = time >= leftover_start
        
        if np.sum(leftover_mask) > 0:
            leftover_time = time[leftover_mask]
            leftover_position = position[leftover_mask]
            leftover_velocity = velocity[leftover_mask]
            leftover_angles = angles[leftover_mask]
            print(f"  Leftover data: {len(leftover_time)} points ({(leftover_time[-1] - leftover_time[0])/60:.2f} minutes)")
            return leftover_time, leftover_position, leftover_velocity, leftover_angles
        else:
            print(f"  No leftover data")
            return None, None, None, None
    
    def process_file(self, filepath):
        """
        Process a single position trace file
        
        Parameters:
        filepath: Path to .mat or .npz file
        
        Returns:
        Number of bins added
        """
        print(f"\n{'='*80}")
        print(f"Processing file: {filepath.name}")
        print(f"{'='*80}")
        
        # Load the data
        data = load_full_position_trace(filepath)
        
        # Extract datetime info
        file_datetime = None
        if data['file_datetimes'] and len(data['file_datetimes']) > 0:
            file_datetime = data['file_datetimes'][0][1]
            print(f"File datetime: {file_datetime}")
        
        # Stitch with previous leftover
        stitched_time, stitched_position, stitched_velocity, stitched_angles = \
            self.stitch_with_previous_leftover(
                data['time'], data['position'], data['velocity'], 
                data['angles'], file_datetime
            )
        
        # Bin the stitched data
        initial_bin_count = len(self.binned_time)
        self.leftover_time, self.leftover_position, self.leftover_velocity, self.leftover_angles = \
            self.bin_data(stitched_time, stitched_position, stitched_velocity, stitched_angles)
        
        bins_added = len(self.binned_time) - initial_bin_count
        
        # Update last file datetime
        if data['file_datetimes'] and len(data['file_datetimes']) > 0:
            # Use the last file's datetime
            self.last_file_end_datetime = data['file_datetimes'][-1][1]
            # Use the actual time array to determine file duration
            if data['file_datetimes'] and len(data['file_datetimes']) > 0:
              file_datetime = data['file_datetimes'][0][1]
              file_duration = data['time'][-1] - data['time'][0]
              self.last_file_end_datetime = file_datetime + timedelta(seconds=file_duration)

        
        return bins_added
    
    def process_directory(self, directory, pattern="*_full_position_trace.mat"):
        """
        Process all matching files in a directory sequentially
        Automatically sorts by datetime if present in filename
        """
        directory = Path(directory)
     
        # Find all matching files
        files = list(directory.glob(pattern))
    
        if len(files) == 0:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    
        # Try to sort by datetime in filename
        def extract_datetime_from_filename(filepath):
            """Extract start datetime from filename if present"""
            try:
                # Pattern: *_YYYYMMDD-HHMMSS_to_YYYYMMDD-HHMMSS_full_position_trace.mat
                parts = filepath.stem.split('_')
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 8:  # YYYYMMDD
                        if i+1 < len(parts) and '-' in parts[i+1]:  # HHMMSS
                            datetime_str = f"{part}{parts[i+1]}"
                            return datetime.datetime.strptime(datetime_str, '%Y%m%d-%H%M%S')
            except:
                pass
            return None
    
        # Try datetime sorting first
        files_with_dt = [(f, extract_datetime_from_filename(f)) for f in files]
    
        if all(dt is not None for _, dt in files_with_dt):
            # All files have datetime - sort by it
            files_with_dt.sort(key=lambda x: x[1])
            files = [f for f, _ in files_with_dt]
            print(f"Found {len(files)} files - sorted by datetime in filename")
        else:
            # Fall back to alphabetical sorting
            files = sorted(files)
            print(f"Found {len(files)} files - sorted alphabetically")
        
        print(f"\n{'='*80}")
        print(f"SEQUENTIAL BINNING OF MULTIPLE FILES")
        print(f"{'='*80}")
        print(f"Found {len(files)} files to process")
        print(f"Output will be {self.bin_duration/60:.0f}-minute bins")
        print(f"{'='*80}\n")
        
        # Process each file sequentially
        total_bins = 0
        for i, filepath in enumerate(files):
            print(f"\nFile {i+1}/{len(files)}")
            bins_added = self.process_file(filepath)
            total_bins += bins_added
            
            print(f"Running total: {total_bins} bins")
        
        # Handle any final leftover data
        if self.leftover_time is not None and len(self.leftover_time) > 0:
            print(f"\n{'='*80}")
            print(f"WARNING: {len(self.leftover_time)} points remaining after last file")
            print(f"         ({(self.leftover_time[-1] - self.leftover_time[0])/60:.2f} minutes)")
            print(f"         This data does not form a complete bin and is excluded")
            print(f"{'='*80}\n")
        
        # Convert lists to arrays
        results = {
            'time': np.array(self.binned_time),
            'position': np.array(self.binned_position),
            'velocity': np.array(self.binned_velocity),
            'position_std': np.array(self.binned_position_std),
            'velocity_std': np.array(self.binned_velocity_std),
            'counts': np.array(self.binned_counts),
            'metadata': {
                'bin_duration_minutes': self.bin_duration / 60,
                'overlap_window_seconds': self.overlap_window,
                'total_bins': len(self.binned_time),
                'files_processed': len(files),
                'absolute_time_reference': str(self.absolute_time_reference) if self.absolute_time_reference else None,
                'total_duration_hours': (self.binned_time[-1] - self.binned_time[0]) / 3600 if len(self.binned_time) > 0 else 0
            }
        }
        
        print(f"\n{'='*80}")
        print(f"BINNING COMPLETE")
        print(f"{'='*80}")
        print(f"Total bins created: {len(results['time'])}")
        print(f"Total duration: {results['metadata']['total_duration_hours']:.2f} hours")
        print(f"Average points per bin: {np.mean(results['counts']):.1f}")
        print(f"{'='*80}\n")
        
        return results

# =============================================================================
# SAVING AND PLOTTING FUNCTIONS (unchanged from the mega script I originally had)
# =============================================================================

def save_binned_data(binned_results, output_dir, dataset_name, bin_duration_minutes):
    """Save binned data to .mat file"""
    output_path = Path(output_dir) / f"{dataset_name}_binned_{bin_duration_minutes}min.mat"
    
    save_dict = {
        'time': binned_results['time'],
        'position': binned_results['position'],
        'velocity': binned_results['velocity'],
        'position_std': binned_results['position_std'],
        'velocity_std': binned_results['velocity_std'],
        'counts': binned_results['counts'],
    }
    
    for key, value in binned_results['metadata'].items():
        if isinstance(value, (int, float, str)) or value is None:
            save_dict[f'meta_{key}'] = value if value is not None else 'None'
    
    save_processed_data(save_dict, output_path, compress=True)
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(binned_results['metadata'], f, indent=2)
    print(f"Metadata also saved to {metadata_path}")
    
    return output_path

def plot_binned_results(binned_results, show_error_bars=True):
    """Create comprehensive plots of binned results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    time_hours = binned_results['time'] / 3600
    position_microns = binned_results['position'] * 1e6
    velocity_microns = binned_results['velocity'] * 1e6
    
    # Plot 1: Position vs time
    ax1 = axes[0, 0]
    ax1.plot(time_hours, position_microns, 'bo-', markersize=3, linewidth=1)
    if show_error_bars:
        ax1.errorbar(time_hours, position_microns, 
                    yerr=binned_results['position_std'] * 1e6,
                    fmt='b', alpha=0.3, capsize=2)
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Position (μm)')
    ax1.set_title(f'Binned Position Data\n({len(time_hours)} bins, {binned_results["metadata"]["bin_duration_minutes"]:.0f}-min bins)')
    ax1.grid(True, alpha=0.3)
    
    from matplotlib.ticker import MultipleLocator
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.grid(True, which='minor', alpha=0.2, linestyle='--')
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    
    # Plot 2: Velocity vs time
    ax2 = axes[0, 1]
    ax2.plot(time_hours, velocity_microns, 'ro-', markersize=3, linewidth=1)
    if show_error_bars:
        ax2.errorbar(time_hours, velocity_microns,
                    yerr=binned_results['velocity_std'] * 1e6,
                    fmt='r', alpha=0.3, capsize=2)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Velocity (μm/s)')
    ax2.set_title('Binned Velocity Data')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Points per bin
    ax3 = axes[1, 0]
    ax3.bar(range(len(binned_results['counts'])), binned_results['counts'], 
            alpha=0.7, color='green')
    ax3.set_xlabel('Bin Number')
    ax3.set_ylabel('Data Points per Bin')
    ax3.set_title(f'Data Density\n(avg: {np.mean(binned_results["counts"]):.1f} points/bin)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position std dev
    ax4 = axes[1, 1]
    ax4.plot(time_hours, binned_results['position_std'] * 1e9, 
            'go-', markersize=3, linewidth=1)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Position Std Dev (nm)')
    ax4.set_title('Position Variability per Bin')
    ax4.grid(True, alpha=0.3)
    
    # Add metadata text
    metadata = binned_results['metadata']
    stats_text = (f'Total bins: {metadata["total_bins"]}\n'
              f'Duration: {metadata["total_duration_hours"]:.2f} hours\n'
              f'Files processed: {metadata["files_processed"]}\n'
              f'Bin size: {metadata["bin_duration_minutes"]:.0f} min')

    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=9)

    plt.tight_layout()
    plt.show()

def create_clean_position_plot(binned_results):
    """Create a simple, clean plot of just the binned position data"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    time_hours = binned_results['time'] / 3600
    position_microns = binned_results['position'] * 1e6
    
    ax.plot(time_hours, position_microns, 'b-', linewidth=0.8, marker='o', markersize=3)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Position (μm)')
    ax.set_title(f'Binned Interferometric Position ({binned_results["metadata"]["bin_duration_minutes"]:.0f}-minute bins)')
    
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, which='minor', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    metadata = binned_results['metadata']
    total_displacement = (position_microns[-1] - position_microns[0])
    position_range = np.max(position_microns) - np.min(position_microns)
    
    stats_text = (f'Data points: {len(position_microns)}\n'
                  f'Duration: {metadata["total_duration_hours"]:.2f} hours\n'
                  f'Total displacement: {total_displacement:.3f} μm\n'
                  f'Position range: {position_range:.3f} μm\n'
                  f'Bin size: {metadata["bin_duration_minutes"]:.0f} min\n'
                  f'Files: {metadata["files_processed"]}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Configuration
    INPUT_DIR = "D:/USB Drive Backup/processed_data_mats/Chunk 1 Output/"
    OUTPUT_DIR = "D:/USB Drive Backup/processed_data_mats/Pipeline Chunk 2 Output"
    DATASET_NAME = "Processed_Data_Chunk"
    BIN_DURATION_MINUTES = 10
    OVERLAP_WINDOW_SECONDS = 30 #Not used anymore but I don't want it to break so I kept them around
    
    print("="*80)
    print("SEQUENTIAL BINNING OF POSITION TRACE FILES")
    print("="*80)
    
    # Create the sequential binner
    binner = SequentialBinner(
        bin_duration_minutes=BIN_DURATION_MINUTES,
        overlap_window_seconds=OVERLAP_WINDOW_SECONDS
    )
    
    # Process all files in the directory
    binned_results = binner.process_directory(
        directory=INPUT_DIR,
        pattern=f"{DATASET_NAME}_*_full_position_trace.mat"
    )
    
    # Save the binned results
    print("\nSaving binned results...")
    output_path = save_binned_data(
        binned_results, OUTPUT_DIR, DATASET_NAME, BIN_DURATION_MINUTES
    )
    
    # Create plots
    print("\nCreating plots...")
    plot_binned_results(binned_results, show_error_bars=True)
    create_clean_position_plot(binned_results)
    
    print("\n" + "="*80)
    print("SEQUENTIAL BINNING COMPLETE")
    print(f"Output saved to: {output_path}")
    print("="*80)