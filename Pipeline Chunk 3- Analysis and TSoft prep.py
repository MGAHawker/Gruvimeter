# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:40:05 2025

@author: migsh
"""

import numpy as np
import scipy as scipy
from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.io import savemat, loadmat
from datetime import timedelta, datetime
import json

# Colour scheme
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOURS)

# =============================================================================
# LOAD FUNCTIONS FROM CHUNK 2
# =============================================================================

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

def load_binned_data(filepath):
    """
    Load binned data from Chunk 2 output
    
    Returns:
    Dictionary with binned data and metadata
    """
    data = load_processed_data(filepath)
    
    # Flatten arrays if needed (MATLAB saves 1D as columns)
    def flatten_if_needed(arr):
        if isinstance(arr, np.ndarray):
            return arr.flatten() if arr.ndim > 1 else arr
        return arr
    
    # Extract metadata from flattened structure
    metadata = {}
    for key, value in data.items():
        if key.startswith('meta_'):
            clean_key = key[5:]
            metadata[clean_key] = value
    
    # Parse absolute time reference if present
    absolute_time_reference = None
    if 'absolute_time_reference' in metadata:
        try:
            ref_str = metadata['absolute_time_reference']
            if ref_str != 'None':
                absolute_time_reference = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S.%f')
        except:
            print("Warning: Could not parse absolute_time_reference")
    
    return {
        'time': flatten_if_needed(data['time']),
        'position': flatten_if_needed(data['position']),
        'velocity': flatten_if_needed(data['velocity']),
        'position_std': flatten_if_needed(data['position_std']),
        'velocity_std': flatten_if_needed(data['velocity_std']),
        'counts': flatten_if_needed(data['counts']),
        'metadata': metadata,
        'absolute_time_reference': absolute_time_reference
    }

# =============================================================================
# PROPER PHYSICS: POSITION TO ACCELERATION CONVERSION
# =============================================================================

def position_to_acceleration_harmonic_oscillator(position_array, oscillator_frequency_hz=20.0, 
                                                 mass_grams=1.198):
    """
    Convert position to acceleration using harmonic oscillator physics
    
    For a harmonic oscillator: F = kx = ma
    Therefore: a = (k/m)x = ω²x
    
    Where:
    - ω = 2πf (angular frequency)
    - f = resonant frequency of the oscillator
    - The mass cancels out in the equation
    
    Parameters:
    -----------
    position_array : array
        Position in meters
    oscillator_frequency_hz : float
        Resonant frequency of your levitated mass (Hz)
        Default: 20 Hz (update if that's not what it is!)
    mass_grams : float
        Mass of levitated object in grammes 
    
    Returns:
    --------
    acceleration_m_s2 : array
        Acceleration in m/s²
    
    Why no mass?:
    ------
    1. For harmonic oscillator: F = -kx (Hooke's law)
    2. Newton's second law: F = ma
    3. Combining: ma = -kx → a = -(k/m)x
    4. For SHM: k = mω² where ω = 2πf
    5. Therefore: a = -ω²x = -(2πf)²x
    
    Unit conversion:
    - Position in m
    - ω in rad/s
    - Acceleration in m/s²
    - Convert to μGal: 1 m/s² = 10⁸ μGal
    """
    
    print(f"\n{'='*80}")
    print(f"HARMONIC OSCILLATOR ACCELERATION CONVERSION")
    print(f"{'='*80}")
    print(f"Oscillator parameters:")
    print(f"  Frequency: {oscillator_frequency_hz} Hz")
    print(f"  Mass: {mass_grams} g (informational - cancels in calculation)")
    
    # Calculate angular frequency
    omega = 2 * np.pi * oscillator_frequency_hz  # rad/s
    
    print(f"  Angular frequency ω: {omega:.4f} rad/s")
    
    # Calculate spring constant k = mω² (for information)
    mass_kg = mass_grams / 1000
    k = mass_kg * omega**2
    print(f"  Spring constant k = mω²: {k:.6f} N/m")
    print(f"  Conversion factor k/m = ω²: {omega**2:.4f} s⁻²")
    
    # Calculate acceleration: a = ω²x
    # We use absolute value since direction is already captured in sign of position
    # The negative sign indicates restoring force
    acceleration_m_s2 = (omega**2) * position_array
    
    print(f"\nPosition to acceleration conversion:")
    print(f"  Position range: {np.min(position_array)*1e6:.4f} to {np.max(position_array)*1e6:.4f} μm")
    print(f"  Acceleration range: {np.min(acceleration_m_s2):.6e} to {np.max(acceleration_m_s2):.6e} m/s²")
    
    # Convert to μGal for display
    acceleration_ugal = acceleration_m_s2 * 1e8
    print(f"  Acceleration range: {np.min(acceleration_ugal):.4f} to {np.max(acceleration_ugal):.4f} μGal")
    print(f"  Acceleration std: {np.std(acceleration_ugal):.4f} μGal")
    
    return acceleration_m_s2
def detrend_position_data(position_m, time_s, detrend_method='linear'):
    """
    Remove drift/trend from position data before converting to acceleration
    
    For tidal analysis, we care about CHANGES in position, not absolute position.
    The drift is likely thermal expansion, pressure changes, etc.
    
    Parameters:
    -----------
    position_m : array
        Position in meters
    time_s : array  
        Time in seconds
    detrend_method : str
        'linear' - fit and remove linear trend
        'polynomial' - fit and remove polynomial (order 2-3)
        'highpass' - high-pass filter (keeps periods < cutoff)
    
    Returns:
    --------
    position_detrended : array
        Detrended position (oscillations around zero)
    trend : array
        The removed trend (for inspection)
    """
    
    print(f"\n{'='*80}")
    print(f"DETRENDING POSITION DATA")
    print(f"{'='*80}")
    print(f"Method: {detrend_method}")
    print(f"Original position range: {np.min(position_m)*1e6:.3f} to {np.max(position_m)*1e6:.3f} μm")
    print(f"Original position std: {np.std(position_m)*1e6:.3f} μm")
    
    if detrend_method == 'linear':
        # Fit linear trend: x(t) = a + b*t
        coeffs = np.polyfit(time_s, position_m, 1)
        trend = np.polyval(coeffs, time_s)
        
        drift_rate = coeffs[0] * 1e6 * 3600  # μm/hour
        print(f"Linear drift rate: {drift_rate:.4f} μm/hour")
        
    elif detrend_method == 'polynomial':
        # Fit polynomial trend (order 2 or 3)
        order = 2
        coeffs = np.polyfit(time_s, position_m, order)
        trend = np.polyval(coeffs, time_s)
        
        print(f"Polynomial fit (order {order})")
        
    elif detrend_method == 'highpass':
        # High-pass filter (keeps tidal frequencies, removes long-term drift)
        from scipy.signal import butter, filtfilt
        
        # Cutoff: keep periods shorter than 48 hours (removes long-term drift)
        sample_rate = 1.0 / np.median(np.diff(time_s))
        cutoff_period = 48 * 3600  # 48 hours
        cutoff_freq = 1.0 / cutoff_period  # Hz
        
        # Normalize frequency (Nyquist frequency = sample_rate/2)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        print(f"High-pass filter cutoff: {cutoff_period/3600:.1f} hours")
        print(f"Removes periods longer than {cutoff_period/3600:.1f} hours")
        
        # Design Butterworth high-pass filter
        b, a = butter(4, normalized_cutoff, btype='high')
        
        # Apply filter (zero-phase)
        position_filtered = filtfilt(b, a, position_m)
        
        # Trend is the low-frequency component (original - filtered)
        trend = position_m - position_filtered
        position_detrended = position_filtered
        
        print(f"Detrended position range: {np.min(position_detrended)*1e6:.3f} to {np.max(position_detrended)*1e6:.3f} μm")
        print(f"Detrended position std: {np.std(position_detrended)*1e6:.3f} μm")
        
        return position_detrended, trend
        
    else:
        raise ValueError(f"Unknown detrend method: {detrend_method}")
    
    # Remove trend
    position_detrended = position_m - trend
    
    print(f"Detrended position range: {np.min(position_detrended)*1e6:.3f} to {np.max(position_detrended)*1e6:.3f} μm")
    print(f"Detrended position std: {np.std(position_detrended)*1e6:.3f} μm")
    print(f"Trend range: {np.min(trend)*1e6:.3f} to {np.max(trend)*1e6:.3f} μm")
    
    return position_detrended, trend
# =============================================================================
# FFT ANALYSIS WITH COMPONENT EXTRACTION
# =============================================================================

def fft_analysis_with_component_extraction(signal, sampling_time, total_duration, 
                                         signal_name="Signal",
                                         reference_datetime=None, 
                                         extract_components=True, 
                                         target_periods_hours=[12.0, 12.42], 
                                         tolerance_hours=0.15):
    """
    Calculate FFT with component extraction
    
    Parameters:
    signal: time-domain signal
    sampling_time: time between samples (seconds)  
    total_duration: total duration of signal (seconds)
    signal_name: Name for the signal
    reference_datetime: Reference datetime for component plots
    extract_components: Whether to extract sinusoidal components
    target_periods_hours: Target periods for extraction [12.0, 12.42]
    tolerance_hours: Tolerance for component extraction (±hours)
    
    Returns:
    periods, magnitudes, period_errors, amplitude_errors, [component_results, component_signals]
    """
    print(f"\n=== FFT ANALYSIS: {signal_name.upper()} ===")
    
    N = len(signal)
    print(f"Signal length: {N} points")
    print(f"Sampling interval: {sampling_time:.1f} seconds")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    
    # Calculate FFT
    yf = fft(signal)
    xf = fftfreq(N, sampling_time)[:N//2]
    magnitudes = 2.0/N * np.abs(yf[:N//2])
    
    # Convert to periods, skip DC component
    with np.errstate(divide='ignore'):
        periods = 1.0 / xf[1:]
    
    xf = xf[1:]
    magnitudes = magnitudes[1:]
    
    # Frequency and period resolution
    df = 1.0 / total_duration
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_freq_error = df / xf
        period_errors = periods * relative_freq_error
    
    period_errors = np.nan_to_num(period_errors)
    
    # Minimum error based on bin spacing
    bin_spacings = np.diff(periods)
    min_period_errors = np.zeros_like(periods)
    min_period_errors[:-1] = bin_spacings / 2
    min_period_errors[-1] = min_period_errors[-2] if len(min_period_errors) > 1 else df
    
    period_errors = np.maximum(period_errors, min_period_errors)
    period_errors *= 0.5  # Scale factor
    
    # Amplitude errors
    noise_level = np.median(magnitudes[-int(N/10):]) if N > 20 else np.std(magnitudes)
    amplitude_errors = np.ones_like(magnitudes) * noise_level
    
    print(f"Frequency resolution: {df:.2e} Hz")
    print(f"Period resolution at lunar period: {((12.42*3600)**2 * df)/3600:.3f} hours")
    
    # Component extraction
    component_results = None
    component_signals = None
    
    if extract_components:
        print(f"\n=== SINUSOIDAL COMPONENT EXTRACTION ===")
        
        yf_full = fft(signal)
        xf_full = fftfreq(N, sampling_time)
        time_array = np.arange(N) * sampling_time
        
        component_results = {}
        component_signals = {}
        
        for target_period_h in target_periods_hours:
            print(f"\n--- Extracting {target_period_h}h component ---")
            
            target_freq = 1.0 / (target_period_h * 3600)
            
            period_min = target_period_h - tolerance_hours
            period_max = target_period_h + tolerance_hours
            freq_min = 1.0 / (period_max * 3600)
            freq_max = 1.0 / (period_min * 3600)
            
            print(f"  Target frequency: {target_freq:.2e} Hz")
            print(f"  Search range: {freq_min:.2e} to {freq_max:.2e} Hz")
            
            positive_freqs = xf_full[xf_full > 0]
            freq_mask_pos = (positive_freqs >= freq_min) & (positive_freqs <= freq_max)
            
            if not np.any(freq_mask_pos):
                print(f"  Warning: No frequencies found near {target_period_h}h period")
                continue
            
            pos_indices = np.where(xf_full > 0)[0]
            target_indices = pos_indices[freq_mask_pos]
            
            amplitudes_in_range = np.abs(yf_full[target_indices])
            max_amp_idx = np.argmax(amplitudes_in_range)
            best_freq_idx = target_indices[max_amp_idx]
            
            dominant_freq = xf_full[best_freq_idx]
            complex_amplitude = yf_full[best_freq_idx]
            amplitude = np.abs(complex_amplitude) * 2.0 / N
            phase_at_zero = np.angle(complex_amplitude)
            
            actual_period_h = 1.0 / dominant_freq / 3600
            
            print(f"  Found frequency: {dominant_freq:.2e} Hz")
            print(f"  Actual period: {actual_period_h:.3f} hours")
            print(f"  Amplitude: {amplitude:.4f}")
            print(f"  Phase at t=0: {phase_at_zero:.3f} rad ({phase_at_zero*180/np.pi:.1f}°)")
            
            # Reconstruct component
            component_signal = amplitude * np.sin(2 * np.pi * dominant_freq * time_array + phase_at_zero)
            
            component_results[f'{target_period_h}h'] = {
                'period_hours': actual_period_h,
                'frequency_hz': dominant_freq,
                'amplitude': amplitude,
                'phase_at_zero_rad': phase_at_zero,
                'target_period': target_period_h
            }
            
            component_signals[f'{target_period_h}h'] = {
                'time': time_array,
                'component': component_signal,
                'amplitude': amplitude,
                'period_hours': actual_period_h
            }
    
    if extract_components:
        return periods, magnitudes, period_errors, amplitude_errors, component_results, component_signals
    else:
        return periods, magnitudes, period_errors, amplitude_errors

# =============================================================================
# PSD ANALYSIS
# =============================================================================

def calculate_position_psd_improved(position_array, time_array, 
                                  oscillator_frequency_hz=20.0,
                                  window="hann", nperseg=None, 
                                  overlap_factor=0.5, detrend='linear'):
    """
    Calculate PSD with proper physics-based acceleration conversion
    
    Parameters:
    position_array: Position data in meters
    time_array: Time data in seconds
    oscillator_frequency_hz: Resonant frequency for acceleration conversion
    window: Window function for PSD
    nperseg: Segment length for Welch's method
    overlap_factor: Overlap between segments (0-1)
    detrend: Detrending method
    
    Returns:
    freqs, psd_displacement, psd_acceleration, sample_rate
    """
    
    dt = np.mean(np.diff(time_array))
    sample_rate = 1.0 / dt
    
    print(f"\n{'='*80}")
    print(f"POSITION PSD ANALYSIS")
    print(f"{'='*80}")
    print(f"  Data points: {len(position_array)}")
    print(f"  Duration: {(time_array[-1] - time_array[0])/3600:.2f} hours")
    print(f"  Sampling interval: {dt/60:.1f} minutes")
    
    if nperseg is None:
        nperseg = len(position_array) // 4
        nperseg = max(nperseg, 32)
    
    noverlap = int(nperseg * overlap_factor)
    
    print(f"  Welch parameters: nperseg={nperseg}, noverlap={noverlap}")
    
    # Calculate PSD for displacement
    freqs, psd_displacement = signal.welch(
        position_array, 
        sample_rate,
        window=window, 
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend
    )
    
    # Calculate acceleration using proper physics
    acceleration = position_to_acceleration_harmonic_oscillator(
        position_array, 
        oscillator_frequency_hz=oscillator_frequency_hz
    )
    
    # Calculate PSD for acceleration
    freqs_acc, psd_acceleration = signal.welch(
        acceleration,
        sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend
    )
    
    return freqs, psd_displacement, psd_acceleration, sample_rate

# =============================================================================
# TSOFT EXPORT
# =============================================================================

def export_acceleration_to_tsoft(binned_data, start_datetime,
                                oscillator_frequency_hz=20.0,
                                mass_grams=1.198,
                                station_name="Gravimeter Station",
                                latitude=50.9097, longitude=-1.4044, elevation=10.0,
                                author_email="user@example.com",
                                output_filename="gravimeter_data.tsf",
                                file_format="TSF",
                                detrend_method='highpass',
                                position_already_detrended=False,
                                regularize_time=True):
    """
    Export binned data to TSOFT format with proper harmonic oscillator physics
    
    Parameters:
    binned_data: Dictionary from Chunk 2 with 'time' and 'position'
    start_datetime: Absolute start time
    oscillator_frequency_hz: Resonant frequency of levitated mass
    mass_grams: Mass in grams (for documentation)
    station_name, latitude, longitude, elevation: Station parameters
    author_email: Contact email
    output_filename: Output .tsf filename
    file_format: "TSF" (only TSF supported for now)
    detrend_method: 'linear', 'polynomial', or 'highpass' (if position not already detrended)
    position_already_detrended: Set to True if you've already detrended the position
    regularize_time: If True, interpolate to perfectly regular time grid (REQUIRED for TSOFT)
    
    Returns:
    Dictionary with export results
    """
    
    print(f"\n{'='*80}")
    print(f"TSOFT EXPORT WITH HARMONIC OSCILLATOR PHYSICS")
    print(f"{'='*80}")
    
    # Extract data
    time_seconds = binned_data['time'].copy()
    position_m = binned_data['position'].copy()
    
    # Check time units
    time_span = time_seconds[-1] - time_seconds[0]
    if time_span < 1000:
        print(f"⚠ Time appears to be in hours, converting to seconds")
        time_seconds = time_seconds * 3600
    
    print(f"Data points: {len(time_seconds)}")
    print(f"Time range: {time_seconds[0]:.1f} to {time_seconds[-1]:.1f} seconds")
    print(f"Duration: {(time_seconds[-1] - time_seconds[0])/3600:.2f} hours")
    print(f"Position range: {np.min(position_m)*1e6:.3f} to {np.max(position_m)*1e6:.3f} μm")
    
    # Check time step consistency
    time_diffs = np.diff(time_seconds)
    sample_rate_median = np.median(time_diffs)
    time_variation = np.std(time_diffs)
    
    print(f"\nTime step analysis:")
    print(f"  Median interval: {sample_rate_median:.2f} seconds ({sample_rate_median/60:.2f} minutes)")
    print(f"  Std deviation: {time_variation:.2f} seconds")
    print(f"  Min interval: {np.min(time_diffs):.2f} seconds")
    print(f"  Max interval: {np.max(time_diffs):.2f} seconds")
    
    if time_variation > sample_rate_median * 0.01:  # More than 1% variation
        print(f"  ⚠ WARNING: Time steps are irregular!")
        print(f"             Variation: {time_variation/sample_rate_median*100:.2f}%")
        if regularize_time:
            print(f"             Will regularise time grid for TSOFT compatibility")
        else:
            print(f"             TSOFT may have issues with irregular time steps!")
    else:
        print(f"  ✓ Time steps are regular (variation < 1%)")
    
    # ==========================================================================
    # DETREND POSITION IF NOT ALREADY DONE
    # ==========================================================================
    if not position_already_detrended:
        print(f"\n{'='*80}")
        print(f"DETRENDING POSITION FOR TSOFT EXPORT")
        print(f"{'='*80}")
        print(f"Reason: TSOFT expects tidal oscillations, not absolute position with drift")
        
        position_detrended, position_trend = detrend_position_data(
            position_m, 
            time_seconds, 
            detrend_method=detrend_method
        )
        
        print(f"\nDetrending summary:")
        print(f"  Original position std: {np.std(position_m)*1e6:.3f} μm")
        print(f"  Detrended position std: {np.std(position_detrended)*1e6:.3f} μm")
        print(f"  Trend removed: {(position_m[-1] - position_m[0])*1e6:.3f} μm total drift")
        
    else:
        print(f"\n✓ Position already detrended - using as-is")
        position_detrended = position_m
        position_trend = None
    
    # ==========================================================================
    # REGULARISE TIME GRID (REQUIRED FOR TSOFT)
    # ==========================================================================
    if regularize_time:
        print(f"\n{'='*80}")
        print(f"REGULARISING TIME GRID FOR TSOFT")
        print(f"{'='*80}")
        print(f"TSOFT requires perfectly uniform time steps")
        
        # Calculate the exact interval needed
        time_start = time_seconds[0]
        time_end = time_seconds[-1]
        n_points_original = len(time_seconds)
        total_duration = time_end - time_start
        
        # Round to nearest integer second for INCREMENT
        sample_rate_exact = total_duration / (n_points_original - 1)
        sample_rate_int = int(np.round(sample_rate_exact))
        
        print(f"Original time span: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
        print(f"Original points: {n_points_original}")
        print(f"Calculated interval: {sample_rate_exact:.3f} seconds")
        print(f"Rounded interval: {sample_rate_int} seconds ({sample_rate_int/60:.2f} minutes)")
        
        # Create perfectly regular grid
        n_intervals = int(np.round(total_duration / sample_rate_int))
        time_regular = time_start + np.arange(n_intervals + 1) * sample_rate_int
        
        # Trim to match duration if needed
        time_regular = time_regular[time_regular <= time_end + sample_rate_int/2]
        
        print(f"Regular grid points: {len(time_regular)}")
        print(f"Regular grid span: {time_regular[0]:.2f} to {time_regular[-1]:.2f} seconds")
        
        # Verify regularity
        if len(time_regular) > 1:
            diffs = np.diff(time_regular)
            print(f"Time step verification:")
            print(f"  Min: {np.min(diffs):.6f} seconds")
            print(f"  Max: {np.max(diffs):.6f} seconds")
            print(f"  Std: {np.std(diffs):.9f} seconds")
            
            if np.std(diffs) < 1e-10:
                print(f"  ✓ Time grid is perfectly regular")
            else:
                print(f"  ⚠ Warning: Small irregularities detected")
        
        # Interpolate position to regular grid
        print(f"\nInterpolating position data to regular grid...")
        from scipy.interpolate import interp1d
        
        interp_func = interp1d(time_seconds, position_detrended, 
                              kind='linear', 
                              bounds_error=False, 
                              fill_value=np.nan)
        
        position_detrended_reg = interp_func(time_regular)
        
        # Check for NaN values (shouldn't be any with proper bounds)
        n_nan = np.sum(np.isnan(position_detrended_reg))
        if n_nan > 0:
            print(f"  ⚠ WARNING: {n_nan} NaN values after interpolation")
            print(f"             This suggests time grid extends beyond data range")
            # Remove NaN values
            valid_mask = ~np.isnan(position_detrended_reg)
            time_regular = time_regular[valid_mask]
            position_detrended_reg = position_detrended_reg[valid_mask]
            print(f"             Trimmed to {len(time_regular)} valid points")
        else:
            print(f"  ✓ Interpolation successful, no NaN values")
        
        # Update with regularised data
        time_seconds = time_regular
        position_detrended = position_detrended_reg
        sample_rate_seconds = sample_rate_int
        
        print(f"✓ Time grid regularized")
        print(f"  Final points: {len(time_seconds)}")
        print(f"  Final interval: {sample_rate_seconds} seconds")
        
    else:
        # Use original time grid
        sample_rate_seconds = int(np.round(sample_rate_median))
        print(f"\n⚠ WARNING: Time regularization disabled")
        print(f"           TSOFT may have issues with irregular time steps")
    
    # ==========================================================================
    # CONVERT DETRENDED POSITION TO ACCELERATION
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"CONVERTING DETRENDED POSITION TO ACCELERATION")
    print(f"{'='*80}")
    
    acceleration_m_s2 = position_to_acceleration_harmonic_oscillator(
        position_detrended,  # ← Use detrended (and possibly regularised) position!
        oscillator_frequency_hz=oscillator_frequency_hz,
        mass_grams=mass_grams
    )
    
    # Convert to μGal
    acceleration_ugal = acceleration_m_s2 * 1e8
    
    print(f"\nAcceleration statistics (from detrended position):")
    print(f"  Range: {np.min(acceleration_ugal):.2f} to {np.max(acceleration_ugal):.2f} μGal")
    print(f"  Peak-to-peak: {np.ptp(acceleration_ugal):.2f} μGal")
    print(f"  Std dev: {np.std(acceleration_ugal):.2f} μGal")
    print(f"  Mean: {np.mean(acceleration_ugal):.2f} μGal")
    
    # Sanity check
    if np.std(acceleration_ugal) > 10000:
        print(f"\n  ⚠ WARNING: Acceleration std is very large ({np.std(acceleration_ugal):.0f} μGal)")
        print(f"             Expected for tidal signals: ~100-500 μGal")
        print(f"             Consider stronger detrending or check your data")
    elif 50 < np.std(acceleration_ugal) < 2000:
        print(f"\n  ✓ Acceleration std looks reasonable for tidal gravimetry")
    else:
        print(f"\n  ⚠ Acceleration std seems small ({np.std(acceleration_ugal):.2f} μGal)")
        print(f"    Expected tidal variations: ~100-300 μGal")
    
    # Create datetime array
    datetime_array = np.array([start_datetime + timedelta(seconds=float(t)) 
                               for t in time_seconds])
    
    print(f"\nTime range: {datetime_array[0]} to {datetime_array[-1]}")
    
    # ==========================================================================
    # WRITE TSF FILE
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"WRITING TSF FILE")
    print(f"{'='*80}")
    
    # For TSF, we want acceleration oscillating around zero (mean removed)
    accel_mean = np.mean(acceleration_ugal)
    accel_centered = acceleration_ugal - accel_mean
    
    print(f"Mean acceleration: {accel_mean:.2f} μGal (will be removed for TSF)")
    print(f"Centered acceleration range: {np.min(accel_centered):.2f} to {np.max(accel_centered):.2f} μGal")
    
    with open(output_filename, 'w') as f:
        f.write("[TSF-file] v01.0\n")
        f.write("[COMMENT]\n")
        f.write(f"Gravimeter data from {station_name}\n")
        f.write(f"Acceleration from harmonic oscillator physics: a = -ω²x\n")
        f.write(f"IMPORTANT: Position was detrended before acceleration conversion\n")
        f.write(f"Detrending method: {detrend_method if not position_already_detrended else 'pre-detrended'}\n")
        f.write(f"Time regularization: {'Applied' if regularize_time else 'Not applied'}\n")
        f.write(f"Oscillator frequency: {oscillator_frequency_hz} Hz\n")
        f.write(f"Mass: {mass_grams} g\n")
        f.write(f"Original position range: {np.min(position_m)*1e6:.3f} to {np.max(position_m)*1e6:.3f} μm\n")
        if not position_already_detrended:
            f.write(f"Detrended position range: {np.min(position_detrended)*1e6:.3f} to {np.max(position_detrended)*1e6:.3f} μm\n")
        f.write(f"Conversion: ω² = {(2*np.pi*oscillator_frequency_hz)**2:.4f} s⁻²\n")
        f.write(f"Mean acceleration: {accel_mean:.6f} μGal (removed)\n")
        f.write(f"Sample rate: {sample_rate_seconds} seconds\n")
        f.write(f"Author: {author_email}\n")
        
        f.write("[TIMEFORMAT] DATETIME\n")
        f.write(f"[INCREMENT] {sample_rate_seconds}\n")
        
        f.write("[CHANNELS]\n")
        f.write(f"{station_name}:Michelson Interferometer:Gravity\n")
        f.write(f"{station_name}:Michelson Interferometer:Pressure\n")
        
        f.write("[UNITS]\n")
        f.write("uGal\n")
        f.write("hPa\n")
        
        f.write("[UNDETVAL] 9999.999\n")
        
        f.write("[DATA]\n")
        
        for i, dt in enumerate(datetime_array):
            date_str = dt.strftime("%Y %m %d %H %M %S")
            # Use centered acceleration (mean removed)
            accel_val = accel_centered[i] if not np.isnan(accel_centered[i]) else 9999.999
            pres_val = 0.0
            
            f.write(f"{date_str} {accel_val:15.10f} {pres_val:10.3f}\n")
    
    print(f"\n✓ Successfully exported to: {output_filename}")
    print(f"\nTSOFT Export Summary:")
    print(f"  - Data points: {len(acceleration_ugal)}")
    print(f"  - Time span: {(datetime_array[-1] - datetime_array[0]).total_seconds()/3600:.2f} hours")
    print(f"  - Sample rate: {sample_rate_seconds} seconds (perfectly regular)")
    print(f"  - Acceleration method: Harmonic oscillator (a = -ω²x)")
    print(f"  - Position detrending: {detrend_method if not position_already_detrended else 'pre-detrended'}")
    print(f"  - Time regularization: {'Applied' if regularize_time else 'Not applied'}")
    print(f"  - Acceleration range: {np.min(accel_centered):.2f} to {np.max(accel_centered):.2f} μGal")
    print(f"  - Acceleration std: {np.std(accel_centered):.2f} μGal")
    
    print(f"\nNext steps for TSOFT:")
    print(f"  1. Open TSOFT and load: {output_filename}")
    print(f"  2. Verify time axis is regular (should be no warnings)")
    print(f"  3. Calculate tidal model:")
    print(f"     - Latitude: {latitude:.4f}°")
    print(f"     - Longitude: {longitude:.4f}°")
    print(f"     - Elevation: {elevation:.1f} m")
    print(f"  4. Fit tidal parameters")
    print(f"  5. Export residuals for comparison")
    
    return {
        'acceleration_ugal': acceleration_ugal,
        'acceleration_centered': accel_centered,
        'acceleration_m_s2': acceleration_m_s2,
        'datetime_array': datetime_array,
        'position_detrended': position_detrended,
        'position_trend': position_trend,
        'time_seconds': time_seconds,
        'metadata': {
            'oscillator_frequency_hz': oscillator_frequency_hz,
            'mass_grams': mass_grams,
            'mean_acceleration_ugal': accel_mean,
            'sample_rate_seconds': sample_rate_seconds,
            'start_datetime': start_datetime,
            'detrend_method': detrend_method if not position_already_detrended else 'pre-detrended',
            'position_already_detrended': position_already_detrended,
            'time_regularized': regularize_time
        }
    }

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_fft_analysis(periods, magnitudes, period_errors, amplitude_errors,
                     signal_name="Position", max_hours=25, units="μm"):
    """Create FFT analysis plot"""
    periods_hours = periods / 3600
    period_errors_hours = period_errors / 3600
    
    mask = periods_hours <= max_hours
    periods_hours = periods_hours[mask]
    magnitudes = magnitudes[mask]
    period_errors_hours = period_errors_hours[mask]
    amplitude_errors = amplitude_errors[mask]
    
    plt.figure(figsize=(14, 6))
    plt.title(f"FFT Analysis: {signal_name} Data")
    
    plt.errorbar(periods_hours, magnitudes, 
                xerr=period_errors_hours, yerr=amplitude_errors, 
                fmt='o-', capsize=3, alpha=0.7, markersize=4)
    
    plt.axvline(x=12.42, color='r', linestyle='--', linewidth=2, label="Lunar Period (~12.42h)")
    plt.axvline(x=24.0, color='g', linestyle='-.', alpha=0.7, label="Solar Day (24h)")
    plt.axvline(x=12.0, color='orange', linestyle=':', alpha=0.7, label="Half-Day (12h)")
    
    plt.xlim(0, max_hours)
    plt.ylim(bottom=0)
    plt.xlabel("Period (hours)")
    plt.ylabel(f"Amplitude ({units})")
    
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(6))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION FOR CHUNK 3
# =============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("CHUNK 3: ANALYSIS PIPELINE")
    print("="*80)
    
    # Configuration
    #""
    INPUT_FILE = "D:/USB Drive Backup/processed_data_mats/Pipeline Chunk 2 Output/Processed_Data_Chunk_binned_10min.mat"
    OUTPUT_DIR = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results"
    
    # Physical parameters of your system - UPDATE THESE!
    OSCILLATOR_FREQUENCY_HZ = 0.809  # Resonant frequency of levitated mass
    MASS_GRAMMES = 1.198  # Mass of levitated object
    
    # Station parameters
    STATION_NAME = "Gruvimeter"
    LONGITUDE = -1.3994
    LATITUDE = 50.9353
    ELEVATION = 42.0
    AUTHOR_EMAIL = "mh3g21@soton.ac.uk"
    
    # Load binned data from Chunk 2
    print("\nLoading binned data from Chunk 2...")
    binned_data = load_binned_data(INPUT_FILE)
    
    # Extract data
    time_seconds = binned_data['time']
    position_m = binned_data['position']
    reference_datetime = binned_data['absolute_time_reference']
    
    print(f"\nLoaded data:")
    print(f"  Points: {len(time_seconds)}")
    print(f"  Duration: {(time_seconds[-1] - time_seconds[0])/3600:.2f} hours")
    print(f"  Reference time: {reference_datetime}")
    
    # Load binned data
    binned_data = load_binned_data(INPUT_FILE)
    time_seconds = binned_data['time']
    position_m = binned_data['position']
    
    # ==========================================================================
    # DETREND POSITION ONCE
    # ==========================================================================
    #position_detrended, position_trend = detrend_position_data(
    #   position_m, 
    #   time_seconds, 
    #   detrend_method='linear'  #  'linear', 'polynomial' or 'highpass'
    # )
   
    # Create detrending diagnostic plot
    #fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    #time_hours = time_seconds / 3600
   
    #axes[0].plot(time_hours, position_m * 1e6, 'b-', linewidth=1)
    #axes[0].set_ylabel('Position (μm)')
    #axes[0].set_title('Original Position (with drift)')
    #axes[0].grid(True, alpha=0.3)
   
    #axes[1].plot(time_hours, position_trend * 1e6, 'r-', linewidth=1)
    #axes[1].set_ylabel('Trend (μm)')
    #axes[1].set_title('Removed Drift Component')
    #axes[1].grid(True, alpha=0.3)
   
    #axes[2].plot(time_hours, position_detrended * 1e6, 'g-', linewidth=1)
    #axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    #axes[2].set_xlabel('Time (hours)')
    #axes[2].set_ylabel('Position (μm)')
    #axes[2].set_title('Detrended Position (tidal oscillations)')
    #axes[2].grid(True, alpha=0.3)
   
    #plt.tight_layout()
    #plt.savefig(Path(OUTPUT_DIR) / "position_detrending.png", dpi=300)
    #plt.show()
    position_detrended = position_m
    # ==========================================================================
    # CONVERT DETRENDED POSITION TO ACCELERATION
    # ==========================================================================
    acceleration_m_s2 = position_to_acceleration_harmonic_oscillator(
       position_detrended,
       oscillator_frequency_hz=OSCILLATOR_FREQUENCY_HZ,
       mass_grams=MASS_GRAMMES
      )
   
    acceleration_ugal = acceleration_m_s2 * 1e8
    position_um = position_detrended * 1e6  # Use detrended for all analysis
   
    # ==========================================================================
    # FFT ANALYSIS (on detrended data)
    # ==========================================================================
    print("\n" + "="*80)
    print("FFT ANALYSIS - DETRENDED POSITION")
    print("="*80)
   
    sampling_time = np.median(np.diff(time_seconds))
    total_duration = time_seconds[-1] - time_seconds[0]
   
    periods_pos, mags_pos, perr_pos, aerr_pos, comp_results_pos, comp_sigs_pos = \
        fft_analysis_with_component_extraction(
            position_um, sampling_time, total_duration, "Detrended Position",
            reference_datetime=reference_datetime, extract_components=True
        )
   
    plot_fft_analysis(periods_pos, mags_pos, perr_pos, aerr_pos, "Detrended Position", 25, "μm")
   
    # FFT Analysis on acceleration
    print("\n" + "="*80)
    print("FFT ANALYSIS - ACCELERATION FROM DETRENDED POSITION")
    print("="*80)
   
    periods_acc, mags_acc, perr_acc, aerr_acc, comp_results_acc, comp_sigs_acc = \
        fft_analysis_with_component_extraction(
            acceleration_ugal, sampling_time, total_duration, "Acceleration",
            reference_datetime=reference_datetime, extract_components=True
        )
   
    plot_fft_analysis(periods_acc, mags_acc, perr_acc, aerr_acc, "Acceleration", 25, "μGal")
   
    # ==========================================================================
    #  PSD ANALYSIS (pass detrended position)
    # ==========================================================================
    print("\n" + "="*80)
    print("POWER SPECTRAL DENSITY ANALYSIS")
    print("="*80)
   
    # Need to create a binned_data dict with detrended position for PSD function
    binned_data_detrended = binned_data.copy()
    binned_data_detrended['position'] = position_detrended
   
    freqs, psd_disp, psd_acc, sample_rate = calculate_position_psd_improved(
        position_detrended,  # Use detrended!
        time_seconds,
        oscillator_frequency_hz=OSCILLATOR_FREQUENCY_HZ,
        window="hann", nperseg=None, overlap_factor=0.5, detrend='constant'  # Already detrended
    )
   
    # ==========================================================================
    # EXPORT TO TSOFT (with detrended position)
    # ==========================================================================
    print("\n" + "="*80)
    print("EXPORTING TO TSOFT FORMAT")
    print("="*80)
   
    if reference_datetime is None:
        print("Warning: No reference datetime found")
        reference_datetime = datetime(2025, 7, 24, 6, 4, 10)  # UPDATE!
        print(f"Using: {reference_datetime}")
   
    # Create binned_data with detrended position
    binned_data_for_export = binned_data.copy()
    binned_data_for_export['position'] = position_detrended
   
    tsoft_results = export_acceleration_to_tsoft(
     binned_data=binned_data_detrended,
     start_datetime=reference_datetime,
     oscillator_frequency_hz=OSCILLATOR_FREQUENCY_HZ,
     mass_grams=MASS_GRAMMES,
     station_name=STATION_NAME,
     latitude=LATITUDE,
     longitude=LONGITUDE,
     elevation=ELEVATION,
     author_email=AUTHOR_EMAIL,
     output_filename=Path(OUTPUT_DIR) / "gravimeter_detrended_regular.tsf",
     file_format="TSF",
     detrend_method='highpass',
     position_already_detrended=True,
     regularize_time=True  # ← TSOFT requires this!
     )
    
    print("\n" + "="*80)
    print("CHUNK 3 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs created:")
    print(f"  - FFT plots (position and acceleration)")
    print(f"  - PSD analysis")
    print(f"  - TSOFT file: gravimeter_harmonic_oscillator.tsf")
    print(f"\nKey results:")
    print(f"  - Oscillator frequency: {OSCILLATOR_FREQUENCY_HZ} Hz")
    print(f"  - ω² conversion factor: {(2*np.pi*OSCILLATOR_FREQUENCY_HZ)**2:.4f} s⁻²")
    print(f"  - Acceleration range: {np.min(acceleration_ugal):.2f} to {np.max(acceleration_ugal):.2f} μGal")
    print(f"  - This uses PROPER PHYSICS (a = -ω²x), NOT double differentiation")
    
    # Plot comparison: Position vs Acceleration
    print("\nCreating comparison plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Convert time to hours for plotting
    time_hours = time_seconds / 3600
    
    # Plot position
    ax1.plot(time_hours, position_um, 'b-', linewidth=1, marker='o', markersize=2)
    ax1.set_ylabel('Position (μm)', fontsize=12)
    ax1.set_title('Position and Acceleration Time Series\n(Harmonic Oscillator Physics: a = -ω²x)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    
    # Plot acceleration
    ax2.plot(time_hours, acceleration_ugal, 'r-', linewidth=1, marker='o', markersize=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Acceleration (μGal)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    

    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "position_acceleration_comparison.png", dpi=300)
    plt.show()
    
    # Create detailed PSD plots
    print("\nCreating PSD plots...")
    
    asd_displacement = np.sqrt(psd_disp)
    asd_acceleration = np.sqrt(psd_acc)
    asd_acceleration_ugal = asd_acceleration * 1e8  # Convert to μGal/√Hz
    
    # Filter for plotting range
    max_period_hours = 50
    min_frequency = 1.0 / (max_period_hours * 3600)
    mask = (freqs >= min_frequency) & (freqs > 0)
    freqs_plot = freqs[mask]
    asd_disp_plot = asd_displacement[mask]
    asd_acc_plot = asd_acceleration_ugal[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Displacement spectrum
    ax1.loglog(freqs_plot, asd_disp_plot, 'b-', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Displacement PSD (m/√Hz)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Power Spectral Density Analysis\n(Harmonic Oscillator Conversion)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.4)
    
    # Add reference lines
    lunar_freq = 1.0 / (12.42 * 3600)
    solar_freq = 1.0 / (24.0 * 3600)
    
    if min_frequency <= lunar_freq <= max(freqs_plot):
        ax1.axvline(lunar_freq, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Lunar period (12.42h)')
    if min_frequency <= solar_freq <= max(freqs_plot):
        ax1.axvline(solar_freq, color='green', linestyle='-.', linewidth=2, alpha=0.8, 
                   label=f'Solar day (24h)')
    
    ax1.legend(loc='best', fontsize=11)
    
    # Plot 2: Acceleration spectrum
    ax2.loglog(freqs_plot, asd_acc_plot, 'r-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Acceleration PSD (μGal/√Hz)', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.4)
    
    # Add reference lines
    if min_frequency <= lunar_freq <= max(freqs_plot):
        ax2.axvline(lunar_freq, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'Lunar period (12.42h)')
    if min_frequency <= solar_freq <= max(freqs_plot):
        ax2.axvline(solar_freq, color='green', linestyle='-.', linewidth=2, alpha=0.8, 
                   label=f'Solar day (24h)')
    
    ax2.legend(loc='best', fontsize=11)
    
    # Add secondary x-axis with periods
    for ax in [ax1, ax2]:
        ax_top = ax.twiny()
        ax_top.set_xscale('log')
        ax_top.set_xlim(ax.get_xlim())
        
        period_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 24, 48]
        freq_ticks = [1.0/(p*3600) for p in period_ticks]
        
        valid_ticks = [(f, p) for f, p in zip(freq_ticks, period_ticks) 
                       if min(freqs_plot) <= f <= max(freqs_plot)]
        
        if valid_ticks:
            freq_ticks_valid, period_labels_valid = zip(*valid_ticks)
            ax_top.set_xticks(freq_ticks_valid)
            ax_top.set_xticklabels([f'{p}h' for p in period_labels_valid])
            ax_top.set_xlabel('Period', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "psd_analysis.png", dpi=300)
    plt.show()
    
    # Print spectral analysis summary
    print(f"\n{'='*80}")
    print(f"SPECTRAL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Frequency range: {freqs_plot[0]:.2e} to {freqs_plot[-1]:.2e} Hz")
    
    periods_hours = 1.0 / (freqs_plot * 3600)
    print(f"Period range: {periods_hours[-1]:.1f} to {periods_hours[0]:.1f} hours")
    
    print(f"\nDisplacement ASD range: {np.min(asd_disp_plot):.2e} to {np.max(asd_disp_plot):.2e} m/√Hz")
    print(f"Acceleration ASD range: {np.min(asd_acc_plot):.2e} to {np.max(asd_acc_plot):.2e} μGal/√Hz")
    
    # Find values at lunar frequency
    if min_frequency <= lunar_freq <= max(freqs_plot):
        lunar_idx = np.argmin(np.abs(freqs_plot - lunar_freq))
        lunar_period_actual = 1.0 / (freqs_plot[lunar_idx] * 3600)
        
        print(f"\nAt lunar frequency (~12.42h period):")
        print(f"  Actual period: {lunar_period_actual:.3f} h")
        print(f"  Displacement ASD: {asd_disp_plot[lunar_idx]:.2e} m/√Hz")
        print(f"  Acceleration ASD: {asd_acc_plot[lunar_idx]:.2e} μGal/√Hz")
    
    # Summary of physics used
    print(f"\n{'='*80}")
    print(f"PHYSICS SUMMARY")
    print(f"{'='*80}")
    print(f"Conversion method: Harmonic oscillator")
    print(f"  Formula: a = -ω²x")
    print(f"  ω = 2πf = 2π × {OSCILLATOR_FREQUENCY_HZ} Hz = {2*np.pi*OSCILLATOR_FREQUENCY_HZ:.4f} rad/s")
    print(f"  ω² = {(2*np.pi*OSCILLATOR_FREQUENCY_HZ)**2:.4f} s⁻²")
    print(f"  Mass: {MASS_GRAMMES} g (cancels in equation)")
    print(f"\nNOTE: This is DIFFERENT from double differentiation!")
    print(f"  - Double differentiation: a = d²x/dt²")
    print(f"  - Harmonic oscillator: a = -ω²x")
    print(f"  - For SHM, these are equivalent, but harmonic oscillator is direct")
    print(f"  - Harmonic oscillator avoids numerical differentiation noise")
    
    # Create summary file
    summary_file = Path(OUTPUT_DIR) / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHUNK 3: ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Data points: {len(time_seconds)}\n")
        f.write(f"Duration: {(time_seconds[-1] - time_seconds[0])/3600:.2f} hours\n")
        f.write(f"Sample rate: {np.median(np.diff(time_seconds))/60:.1f} minutes\n\n")
        
        f.write("Physical Parameters:\n")
        f.write(f"  Oscillator frequency: {OSCILLATOR_FREQUENCY_HZ} Hz\n")
        f.write(f"  Mass: {MASS_GRAMMES} g\n")
        f.write(f"  ω² conversion factor: {(2*np.pi*OSCILLATOR_FREQUENCY_HZ)**2:.4f} s⁻²\n\n")
        
        f.write("Position Statistics:\n")
        f.write(f"  Range: {np.min(position_um):.3f} to {np.max(position_um):.3f} μm\n")
        f.write(f"  Mean: {np.mean(position_um):.3f} μm\n")
        f.write(f"  Std: {np.std(position_um):.3f} μm\n\n")
        
        f.write("Acceleration Statistics (from a = -ω²x):\n")
        f.write(f"  Range: {np.min(acceleration_ugal):.2f} to {np.max(acceleration_ugal):.2f} μGal\n")
        f.write(f"  Mean: {np.mean(acceleration_ugal):.2f} μGal\n")
        f.write(f"  Std: {np.std(acceleration_ugal):.2f} μGal\n\n")
        
        f.write("Outputs Generated:\n")
        f.write(f"  - gravimeter_harmonic_oscillator.tsf (TSOFT format)\n")
        f.write(f"  - position_acceleration_comparison.png\n")
        f.write(f"  - psd_analysis.png\n")
        f.write(f"  - FFT plots (displayed)\n\n")
        
        f.write("Next Steps:\n")
        f.write(f"  1. Open TSOFT and load: gravimeter_harmonic_oscillator.tsf\n")
        f.write(f"  2. Calculate tidal model using station coordinates:\n")
        f.write(f"     Latitude: {LATITUDE}°\n")
        f.write(f"     Longitude: {LONGITUDE}°\n")
        f.write(f"     Elevation: {ELEVATION} m\n")
        f.write(f"  3. Fit tidal parameters (LSQ adjustment)\n")
        f.write(f"  4. Export and analyze residuals\n")
    
    print(f"\nSummary written to: {summary_file}")
    
    print("\n" + "="*80)
    print("ALL ANALYSIS COMPLETE!")
    print("="*80)