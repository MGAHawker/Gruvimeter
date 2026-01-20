# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:53:51 2025

@author: migsh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks, correlate
from scipy import signal
from pathlib import Path

# -------- PARAMETERS --------
COMPONENT_DIR = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tidal Model component dats/"
obs_file = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tsoft dat exports/Acceleration Data 10 min bins.dat"
model_file = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tsoft dat exports/240725.dat"
start_time = pd.Timestamp("2025-07-23 12:00:00")
dt_input = 600.0  # seconds
flip_model = True

# WHITE/PINK/BROWN NOISE GENERATION PARAMETERS
USE_SYNTHETIC_NOISE = False # SET TO TRUE FOR NULL HYPOTHESIS TEST
NOISE_TYPE = 'brown'  #'white', 'pink', or 'brown'
NOISE_AMPLITUDE_MULTIPLIER = 1.0  # Scale factor for noise amplitude relative to real data

# FITTED NOISE FROM LONG DATA
USE_FITTED_NOISE = True  # SET TO TRUE TO USE FITTED NOISE FROM LONG DATA
LONG_DATA_FILE = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tsoft dat exports/Jul-Oct data set.dat"

# MANUAL SCALING
MANUAL_SCALE_FACTOR = None

# Resonant frequency
ASSUMED_RESONANT_FREQ_HZ = 0.809

# HIGH-PASS FILTER
HIGHPASS_CUTOFF_HOURS = 50
ENABLE_HIGHPASS = False

# TIDAL CONSTITUENT IDENTIFICATION
PEAK_THRESHOLD = 0.3  
PEAK_WIDTH_HOURS = 0.5
TOLERANCE_HOURS = 0.4

# Force include these constituents
FORCE_INCLUDE_CONSTITUENTS = ['M2', 'S2', 'K1+P1', 'O1']

# Filter type selection
SPECTRAL_PEAK_BANDWIDTH_HOURS = 0.2

# Drift removal
REMOVE_DRIFT_ORDER = None

# Known tidal constituents
TIDAL_CONSTITUENTS = {
    'M2': 12.4206,
    'S2': 12.0000,
    'N2': 12.6583,
    'K1+P1': 23.9997,
    'O1': 25.8193,
    'Q1': 26.8684,
}

INDIVIDUAL_CONSTITUENTS = {
    'K1': 23.9345,
    'P1': 24.0659,
}

# --------------------------------

def load_with_time(fname, t0, dt):
    data = np.loadtxt(fname, comments="#")
    n = len(data)
    t = t0 + pd.to_timedelta(np.arange(n)*dt, unit='s')
    y = data[:,1]
    return t, y

def generate_white_noise(n_points, std_dev, seed=42):
    """Generate white noise with specified standard deviation"""
    np.random.seed(seed)
    noise = np.random.normal(0, std_dev, n_points)
    print(f"\n{'='*80}")
    print(f" WHITE NOISE GENERATION (NULL HYPOTHESIS TEST)")
    print(f"{'='*80}")
    print(f"  Generated {n_points} points of white noise")
    print(f"  Standard deviation: {std_dev:.2f} µGal")
    print(f"  Mean: {np.mean(noise):.2f} µGal")
    print(f"  This noise contains NO tidal signal")
    print(f"  Any apparent tidal detection would be a FALSE POSITIVE")
    return noise

def generate_pink_noise(n_points, std_dev, seed=42):
    """
    Generate 1/f (pink) noise with specified standard deviation
    Uses the Voss-McCartney algorithm via spectral shaping
    """
    np.random.seed(seed)
    
    # Generate white noise in frequency domain
    white = np.random.randn(n_points)
    white_fft = fft(white)
    
    # Create frequency array
    freqs = fftfreq(n_points)
    
    # Create 1/f filter (avoid division by zero at DC)
    # Power spectrum goes as 1/f, so amplitude goes as 1/sqrt(f)
    pink_filter = np.zeros(n_points)
    pink_filter[0] = 1.0  # DC component
    pink_filter[1:] = 1.0 / np.sqrt(np.abs(freqs[1:]))
    
    # Apply filter in frequency domain
    pink_fft = white_fft * pink_filter
    
    # Convert back to time domain
    pink = np.real(ifft(pink_fft))
    
    # Normalise to desired standard deviation
    pink = pink * (std_dev / np.std(pink))
    
    print(f"\n{'='*80}")
    print(f" PINK NOISE (1/f) GENERATION (NULL HYPOTHESIS TEST)")
    print(f"{'='*80}")
    print(f"  Generated {n_points} points of 1/f (pink) noise")
    print(f"  Standard deviation: {std_dev:.2f} µGal")
    print(f"  Mean: {np.mean(pink):.2f} µGal")
    print(f"  Power spectrum: P(f) ∝ 1/f")
    print(f"  This noise contains NO tidal signal")
    print(f"  Any apparent tidal detection would be a FALSE POSITIVE")
    
    # Verify 1/f characteristics
    pink_fft_check = fft(pink - np.mean(pink))
    freqs_check = fftfreq(n_points)
    power_spectrum = np.abs(pink_fft_check[1:n_points//2])**2
    freqs_positive = np.abs(freqs_check[1:n_points//2])
    
    # Log-log slope should be approximately -1
    log_freqs = np.log10(freqs_positive[freqs_positive > 0])
    log_power = np.log10(power_spectrum[freqs_positive > 0])
    slope = np.polyfit(log_freqs, log_power, 1)[0]
    print(f"  Verification: log-log slope = {slope:.2f} (ideal = -1.0 for 1/f power)")
    
    return pink

def generate_brown_noise(n_points, std_dev, seed=42):
    """
    Generate 1/f² (brown/Brownian/red) noise with specified standard deviation
    Brown noise corresponds to random walk - integral of white noise
    Power spectrum: P(f) ∝ 1/f²
    """
    np.random.seed(seed)
    
    # Generate white noise in frequency domain
    white = np.random.randn(n_points)
    white_fft = fft(white)
    
    # Create frequency array
    freqs = fftfreq(n_points)
    
    # Create 1/f² filter (avoid division by zero at DC)
    # Power spectrum goes as 1/f², so amplitude goes as 1/f
    brown_filter = np.zeros(n_points)
    brown_filter[0] = 1.0  # DC component
    brown_filter[1:] = 1.0 / np.abs(freqs[1:])
    
    # Apply filter in frequency domain
    brown_fft = white_fft * brown_filter
    
    # Convert back to time domain
    brown = np.real(ifft(brown_fft))
    
    # Normalise to desired standard deviation
    brown = brown * (std_dev / np.std(brown))
    
    print(f"\n{'='*80}")
    print(f" BROWN NOISE (1/f²) GENERATION (NULL HYPOTHESIS TEST)")
    print(f"{'='*80}")
    print(f"  Generated {n_points} points of 1/f² (Brownian/red) noise")
    print(f"  Standard deviation: {std_dev:.2f} µGal")
    print(f"  Mean: {np.mean(brown):.2f} µGal")
    print(f"  Power spectrum: P(f) ∝ 1/f² (random walk)")
    print(f"  This noise contains NO tidal signal")
    print(f"  Any apparent tidal detection would be a FALSE POSITIVE")
    
    # Verify 1/f² characteristics
    brown_fft_check = fft(brown - np.mean(brown))
    freqs_check = fftfreq(n_points)
    power_spectrum = np.abs(brown_fft_check[1:n_points//2])**2
    freqs_positive = np.abs(freqs_check[1:n_points//2])
    
    # Log-log slope should be approximately -2
    log_freqs = np.log10(freqs_positive[freqs_positive > 0])
    log_power = np.log10(power_spectrum[freqs_positive > 0])
    slope = np.polyfit(log_freqs, log_power, 1)[0]
    print(f"  Verification: log-log slope = {slope:.2f} (ideal = -2.0 for 1/f² power)")
    
    return brown

def generate_fitted_noise(n_points, long_data_file, t0, dt, seed=42):
    """
    Generate noise based on the spectral characteristics of a long data trace.
    Fits the power spectrum and generates synthetic noise matching that spectrum.
    """
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f" FITTED NOISE FROM LONG DATA (NULL HYPOTHESIS TEST)")
    print(f"{'='*80}")
    
    # Load long data
    print(f"Loading long data file: {long_data_file}")
    t_long, y_long = load_with_time(long_data_file, t0, dt)
    print(f"  Loaded {len(y_long)} points ({(t_long[-1]-t_long[0]).total_seconds()/3600:.1f} hours)")
    
    # Remove mean and detrend
    y_long_detrended, _ = remove_polynomial_drift(y_long, order=3)
    
    # Compute power spectrum
    y_long_fft = fft(y_long_detrended)
    freqs_long = fftfreq(len(y_long_detrended), dt)
    power_spectrum_long = np.abs(y_long_fft)**2
    
    # Only use positive frequencies for fitting
    pos_mask = freqs_long > 0
    freqs_pos = freqs_long[pos_mask]
    power_pos = power_spectrum_long[pos_mask]
    
    # Fit power law to the spectrum: P(f) = A * f^(-beta)
    # Use log-log fit, excluding very low frequencies (< 1/50 hours)
    min_freq = 1.0 / (50 * 3600)  # Exclude periods > 50 hours
    fit_mask = freqs_pos > min_freq
    
    log_freqs = np.log10(freqs_pos[fit_mask])
    log_power = np.log10(power_pos[fit_mask])
    
    # Robust fit (exclude outliers)
    fit_coeffs = np.polyfit(log_freqs, log_power, 1)
    beta = -fit_coeffs[0]  # Power law exponent
    log_A = fit_coeffs[1]
    A = 10**log_A
    
    print(f"\n  Fitted power spectrum: P(f) = {A:.2e} × f^({-beta:.2f})")
    print(f"  Beta = {beta:.2f} (1/f noise has beta=1, white noise has beta=0)")
    
    # Generate noise with fitted spectrum
    white = np.random.randn(n_points)
    white_fft = fft(white)
    freqs_new = fftfreq(n_points, dt)
    
    # Create filter matching the fitted power spectrum
    # Power goes as f^(-beta), so amplitude goes as f^(-beta/2)
    fitted_filter = np.zeros(n_points, dtype=complex)
    fitted_filter[0] = 1.0  # DC component
    
    for i in range(1, n_points):
        freq = abs(freqs_new[i])
        if freq > 0:
            fitted_filter[i] = (freq / freqs_pos[0])**(-beta/2)
    
    # Apply filter
    fitted_noise_fft = white_fft * fitted_filter
    fitted_noise = np.real(ifft(fitted_noise_fft))
    
    # Scale to match original data standard deviation
    original_std = np.std(y_long_detrended)
    fitted_noise = fitted_noise * (original_std / np.std(fitted_noise))
    
    print(f"  Generated {n_points} points of fitted noise")
    print(f"  Standard deviation: {np.std(fitted_noise):.2f} µGal (original: {original_std:.2f} µGal)")
    print(f"  Mean: {np.mean(fitted_noise):.2f} µGal")
    print(f"  This noise contains NO tidal signal")
    print(f"  Any apparent tidal detection would be a FALSE POSITIVE")
    
    # Verify fitted characteristics
    fitted_check_fft = fft(fitted_noise - np.mean(fitted_noise))
    freqs_check = fftfreq(n_points, dt)
    power_check = np.abs(fitted_check_fft[1:n_points//2])**2
    freqs_check_pos = np.abs(freqs_check[1:n_points//2])
    
    check_mask = freqs_check_pos > min_freq
    log_freqs_check = np.log10(freqs_check_pos[check_mask])
    log_power_check = np.log10(power_check[check_mask])
    slope_check = np.polyfit(log_freqs_check, log_power_check, 1)[0]
    print(f"  Verification: log-log slope = {slope_check:.2f} (target = {-beta:.2f})")
    
    return fitted_noise

def remove_polynomial_drift(data, order=3):
    """Remove polynomial trend"""
    if order is None:
        print(f"\n⚠️  DETRENDING DISABLED - Using raw data (mean removed only)")
        mean_val = np.mean(data)
        return data - mean_val, np.full_like(data, mean_val)
    
    n = len(data)
    x = np.arange(n)
    coeffs = np.polyfit(x, data, order)
    trend = np.polyval(coeffs, x)
    print(f"\nPolynomial detrending (order {order}):")
    print(f"  Trend range: {np.ptp(trend):.2f} µGal")
    print(f"  Data std before: {np.std(data):.2f} µGal")
    print(f"  Data std after: {np.std(data - trend):.2f} µGal")
    return data - trend, trend

def highpass_filter(signal, dt, cutoff_period_hours=100):
    """Remove drift by high-pass filtering"""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    signal_fft = fft(signal_centered)
    freqs = fftfreq(n, dt)
    
    cutoff_freq = 1.0 / (cutoff_period_hours * 3600)
    highpass_mask = np.abs(freqs) >= cutoff_freq
    
    filtered_fft = signal_fft * highpass_mask
    filtered_signal = np.real(ifft(filtered_fft))
    
    removed_component = signal_centered - filtered_signal
    
    print(f"\n{'='*80}")
    print(f"HIGH-PASS DRIFT REMOVAL")
    print(f"{'='*80}")
    print(f"Cutoff period: {cutoff_period_hours:.1f} hours")
    print(f"Drift std: {np.std(removed_component):.2f} µGal")
    print(f"Signal std after removal: {np.std(filtered_signal):.2f} µGal")
    
    return filtered_signal, removed_component, highpass_mask

def identify_tidal_peaks(signal, dt, threshold=0.2, min_separation_hours=2.0):
    """Identify tidal constituent peaks in FFT spectrum"""
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    signal_fft = fft(signal_centered)
    freqs = fftfreq(n, dt)
    
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    fft_magnitude = np.abs(signal_fft[positive_mask])
    
    periods = 1.0 / (freqs_pos * 3600)
    
    tidal_mask = (periods >= 6) & (periods <= 50)
    periods_tidal = periods[tidal_mask]
    magnitude_tidal = fft_magnitude[tidal_mask]
    
    peak_threshold = threshold * np.max(magnitude_tidal)
    min_distance = int(min_separation_hours / (periods_tidal[1] - periods_tidal[0]))
    
    peak_indices, properties = find_peaks(
        magnitude_tidal, 
        height=peak_threshold,
        distance=max(1, min_distance)
    )
    
    peak_periods = periods_tidal[peak_indices]
    peak_amplitudes = magnitude_tidal[peak_indices]
    peak_freqs = freqs_pos[tidal_mask][peak_indices]
    
    peak_labels = []
    for period in peak_periods:
        label = None
        min_diff = float('inf')
        for const_name, const_period in TIDAL_CONSTITUENTS.items():
            diff = abs(period - const_period)
            if diff < TOLERANCE_HOURS and diff < min_diff:
                label = const_name
                min_diff = diff
        peak_labels.append(label if label else f"~{period:.1f}h")
    
    sort_idx = np.argsort(peak_amplitudes)[::-1]
    
    results = {
        'periods': peak_periods[sort_idx],
        'amplitudes': peak_amplitudes[sort_idx],
        'frequencies': peak_freqs[sort_idx],
        'labels': [peak_labels[i] for i in sort_idx],
        'full_spectrum': {
            'periods': periods_tidal,
            'magnitude': magnitude_tidal,
            'freqs': freqs_pos[tidal_mask]
        }
    }
    
    return results

def force_include_missing_constituents(peaks, signal, dt, force_list):
    """Add any missing constituents from force_list"""
    detected = set([label for label in peaks['labels'] if label in TIDAL_CONSTITUENTS])
    missing = [c for c in force_list if c in TIDAL_CONSTITUENTS and c not in detected]
    
    if not missing:
        return peaks
    
    print(f"\n{'='*80}")
    print(f"⚠️  FORCE-INCLUDING MISSING CONSTITUENTS")
    print(f"{'='*80}")
    print(f"Extracting: {missing}")
    
    n = len(signal)
    signal_centered = signal - np.mean(signal)
    signal_fft = fft(signal_centered)
    freqs = fftfreq(n, dt)
    
    for const_name in missing:
        period = TIDAL_CONSTITUENTS[const_name]
        
        if const_name == 'K1+P1':
            extraction_bandwidth_hours = 0.25
            print(f"\n  ✓ {const_name}: Combined diurnal constituent (K1 + P1)")
            print(f"      K1 period: {INDIVIDUAL_CONSTITUENTS['K1']:.3f}h")
            print(f"      P1 period: {INDIVIDUAL_CONSTITUENTS['P1']:.3f}h")
            print(f"      Center period: {period:.3f}h")
        else:
            extraction_bandwidth_hours = 0.15
        
        period_low = period - extraction_bandwidth_hours
        period_high = period + extraction_bandwidth_hours
        
        const_freq = 1.0 / (period * 3600)
        freq_high = 1.0 / (period_low * 3600)
        freq_low = 1.0 / (period_high * 3600)
        
        pos_mask = (freqs >= freq_low) & (freqs <= freq_high)
        neg_mask = (freqs >= -freq_high) & (freqs <= -freq_low)
        extraction_mask = pos_mask | neg_mask
        
        extracted_fft = signal_fft[extraction_mask]
        
        if len(extracted_fft) > 0:
            amplitude = np.max(np.abs(extracted_fft))
        else:
            amplitude = 0.0
        
        if not isinstance(peaks['labels'], list):
            peaks['labels'] = list(peaks['labels'])
        peaks['labels'].append(const_name)
            
        peaks['periods'] = np.append(peaks['periods'], period)
        peaks['frequencies'] = np.append(peaks['frequencies'], const_freq)
        peaks['amplitudes'] = np.append(peaks['amplitudes'], amplitude)
        
        n_bins = np.sum(pos_mask)
        print(f"      Extraction window: {period_low:.3f}h to {period_high:.3f}h")
        print(f"      Frequency bins captured: {n_bins}")
        print(f"      Extracted amplitude: {amplitude:.2e}")
    
    sort_idx = np.argsort(peaks['amplitudes'])[::-1]
    peaks['periods'] = peaks['periods'][sort_idx]
    peaks['amplitudes'] = peaks['amplitudes'][sort_idx]
    peaks['frequencies'] = peaks['frequencies'][sort_idx]
    peaks['labels'] = [peaks['labels'][i] for i in sort_idx]
    
    print(f"\nFinal constituent list after force-inclusion:")
    print(f"  {[label for label in peaks['labels'] if label in TIDAL_CONSTITUENTS]}")
    
    return peaks

def create_narrowband_constituent_filter(peaks, n_points, dt, bandwidth_hours=0.25):
    """Create narrow brick-wall bandpass filter around identified constituents"""
    freqs = fftfreq(n_points, dt)
    filter_mask = np.zeros(len(freqs), dtype=float)
    
    print(f"\n{'='*80}")
    print(f"NARROW-BAND CONSTITUENT FILTER")
    print(f"{'='*80}")
    print(f"Creating brick-wall bandpass around identified tidal constituents:")
    
    constituents_used = []
    n_bins_per_constituent = []
    
    for i, (peak_freq, label, period) in enumerate(zip(
        peaks['frequencies'], 
        peaks['labels'],
        peaks['periods']
    )):
        if label in TIDAL_CONSTITUENTS:
            if label == 'K1+P1':
                bw = 0.25
                print(f"  ✓ {label:<6}: Period={period:.3f}h (combined K1+P1 peak)")
            else:
                bw = bandwidth_hours
                print(f"  ✓ {label:<6}: Period={period:.3f}h, Freq={peak_freq:.6e} Hz")
            
            period_low = period - bw
            period_high = period + bw
            
            if period_low > 0:
                freq_high = 1.0 / (period_low * 3600)
                freq_low = 1.0 / (period_high * 3600)
            else:
                freq_low = 0
                freq_high = 1.0 / (period_high * 3600)
            
            pos_mask = (freqs >= freq_low) & (freqs <= freq_high)
            neg_mask = (freqs >= -freq_high) & (freqs <= -freq_low)
            
            constituent_mask = pos_mask | neg_mask
            filter_mask[constituent_mask] = 1.0
            
            n_bins = np.sum(pos_mask)
            n_bins_per_constituent.append(n_bins)
            constituents_used.append(label)
            
            print(f"      Window: {period_low:.3f}h to {period_high:.3f}h")
            print(f"      Bins passed: {n_bins}")
        else:
            print(f"  ✗ {label:<6}: SKIPPED (not a named constituent)")
    
    total_bins = len(freqs)
    positive_bins = np.sum(freqs > 0)
    passed_positive = int(np.sum(filter_mask[freqs > 0]))
    
    print(f"\n{'='*60}")
    print(f"Filter Summary:")
    print(f"  Constituents used: {len(constituents_used)}")
    print(f"  Bins passed by filter: {passed_positive} / {positive_bins} ({100*passed_positive/positive_bins:.2f}%)")
    
    return filter_mask, freqs

def apply_filter(data_signal, filter_mask):
    """Apply filter in frequency domain"""
    data_centered = data_signal - np.mean(data_signal)
    data_fft = fft(data_centered)
    filtered_fft = data_fft * filter_mask
    filtered_signal = np.real(ifft(filtered_fft))
    
    data_std = np.std(data_centered)
    filtered_std = np.std(filtered_signal)
    
    print(f"\nFiltering results:")
    print(f"  Original std: {data_std:.2f} µGal")
    print(f"  Filtered std: {filtered_std:.2f} µGal")
    print(f"  Power retained: {(filtered_std/data_std)**2 * 100:.2f}%")
    
    return filtered_signal, data_fft, filtered_fft

def calibrate_amplitude(filtered_obs, model_signal, manual_scale=None):
    """Calibrate amplitude AFTER filtering"""
    filtered_obs_centered = filtered_obs - np.mean(filtered_obs)
    model_centered = model_signal - np.mean(model_signal)
    
    obs_rms = np.std(filtered_obs_centered)
    model_rms = np.std(model_centered)
    
    calculated_scale = model_rms / obs_rms
    
    correlation_before = np.corrcoef(filtered_obs_centered, model_centered)[0, 1]
    
    if manual_scale is not None:
        scale_factor = manual_scale
        print(f"\n{'='*80}")
        print(f"MANUAL SCALE FACTOR APPLIED")
        print(f"{'='*80}")
        print(f"Using manual scale: {scale_factor:.6f}")
    else:
        scale_factor = calculated_scale
        print(f"\n{'='*80}")
        print(f"AUTOMATIC AMPLITUDE CALIBRATION")
        print(f"{'='*80}")
        print(f"Calculated scale factor: {scale_factor:.6f}")
    
    filtered_obs_scaled = filtered_obs_centered * scale_factor
    
    correlation_after = np.corrcoef(filtered_obs_scaled, model_centered)[0, 1]
    rms_diff = np.sqrt(np.mean((filtered_obs_scaled - model_centered)**2))
    residual = filtered_obs_scaled - model_centered
    
    print(f"\nBefore scaling:")
    print(f"  Filtered obs RMS: {obs_rms:.2f} µGal")
    print(f"  Model RMS: {model_rms:.2f} µGal")
    print(f"  Correlation: {correlation_before:.4f}")
    
    print(f"\nAfter scaling:")
    print(f"  Scaled obs RMS: {np.std(filtered_obs_scaled):.2f} µGal")
    print(f"  Model RMS: {model_rms:.2f} µGal")
    print(f"  Correlation: {correlation_after:.4f}")
    print(f"  RMS difference: {rms_diff:.2f} µGal")
    
    return scale_factor, filtered_obs_scaled, correlation_after, rms_diff, residual

def calculate_amplitude_ratios(peaks, verbose=True):
    """Calculate ratios between major constituents"""
    ratios = {}
    
    constituent_amps = {}
    for i, label in enumerate(peaks['labels']):
        if label in TIDAL_CONSTITUENTS:
            constituent_amps[label] = peaks['amplitudes'][i]
    
    if verbose:
        print(f"\n  Detected constituents: {list(constituent_amps.keys())}")
    
    ratio_definitions = [
        ('M2', 'S2', 'M2/S2'),
        ('M2', 'N2', 'M2/N2'),
        ('S2', 'N2', 'S2/N2'),
        ('K1+P1', 'O1', '(K1+P1)/O1'),
        ('M2', 'K1+P1', 'M2/(K1+P1)'),
        ('M2', 'O1', 'M2/O1'),
        ('S2', 'K1+P1', 'S2/(K1+P1)'),
        ('S2', 'O1', 'S2/O1'),
    ]
    
    for num_const, den_const, ratio_name in ratio_definitions:
        if num_const in constituent_amps and den_const in constituent_amps:
            ratios[ratio_name] = constituent_amps[num_const] / constituent_amps[den_const]
            if verbose:
                print(f"  ✓ {ratio_name}: {ratios[ratio_name]:.3f}")
        elif verbose:
            missing = []
            if num_const not in constituent_amps:
                missing.append(num_const)
            if den_const not in constituent_amps:
                missing.append(den_const)
            print(f"  ✗ {ratio_name}: Missing {', '.join(missing)}")
    
    return ratios, constituent_amps

def compare_ratios_with_tiered_thresholds(model_ratios, data_ratios):
    """Compare ratios with multiple threshold levels"""
    print(f"\n{'='*80}")
    print(f"DETAILED RATIO COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Ratio':<18}{'Model':<12}{'Data':<12}{'Abs Diff':<12}{'% Diff':<12}{'Quality'}")
    print(f"{'-'*80}")
    
    quality_counts = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0, 'Failed': 0}
    all_pct_diffs = []
    
    for ratio_name in sorted(model_ratios.keys()):
        if ratio_name in data_ratios:
            model_val = model_ratios[ratio_name]
            data_val = data_ratios[ratio_name]
            abs_diff = abs(model_val - data_val)
            pct_diff = 100 * abs_diff / model_val
            all_pct_diffs.append(pct_diff)
            
            if pct_diff < 5:
                quality = "✓✓ Excellent"
                quality_counts['Excellent'] += 1
            elif pct_diff < 10:
                quality = "✓ Good"
                quality_counts['Good'] += 1
            elif pct_diff < 20:
                quality = "~ Fair"
                quality_counts['Fair'] += 1
            elif pct_diff < 30:
                quality = "⚠ Poor"
                quality_counts['Poor'] += 1
            else:
                quality = "✗ Failed"
                quality_counts['Failed'] += 1
            
            print(f"{ratio_name:<18}{model_val:<12.3f}{data_val:<12.3f}"
                  f"{abs_diff:<12.3f}{pct_diff:<12.1f}{quality}")
    
    if len(all_pct_diffs) > 0:
        mean_diff = np.mean(all_pct_diffs)
        
        print(f"\n{'='*80}")
        print(f"Statistics over {len(all_pct_diffs)} ratio(s):")
        print(f"  Mean difference: {mean_diff:.2f}%")
        
        excellent_or_good = quality_counts['Excellent'] + quality_counts['Good']
        pct_excellent_or_good = 100 * excellent_or_good / len(all_pct_diffs)
        
        print(f"\n{'='*80}")
        if pct_excellent_or_good >= 80 and mean_diff < 10:
            print(f"✓✓✓ VERY HIGH CONFIDENCE: Strong tidal detection!")
            confidence = "VERY_HIGH"
        elif pct_excellent_or_good >= 60 and mean_diff < 15:
            print(f"✓✓ HIGH CONFIDENCE: Clear tidal detection")
            confidence = "HIGH"
        elif pct_excellent_or_good >= 40 and mean_diff < 20:
            print(f"✓ MODERATE CONFIDENCE: Likely tidal detection")
            confidence= 'MEDIUM'
        else:
            print(f"⚠ LOW CONFIDENCE: Detection uncertain")
            confidence = "LOW"
        
        return confidence, mean_diff, quality_counts
    
    return "NO_DATA", None, quality_counts

# ================================
# MAIN EXECUTION
# ================================

print(f"\n{'='*80}")
if USE_SYNTHETIC_NOISE:
    if NOISE_TYPE == 'pink':
        print(f" PINK NOISE (1/f) NULL HYPOTHESIS TEST")
    elif NOISE_TYPE == 'brown':
        print(f"BROWN NOISE NULL HYPOTHESIS TEST")
    else:
        print(f" WHITE NOISE NULL HYPOTHESIS TEST")
    print(f"{'='*80}")
else:
    print(f"TIDAL DETECTION WITH COMBINED K1+P1 CONSTITUENT")
    print(f"{'='*80}")

print(f"  K1 (23.93h) and P1 (24.07h) are separated by only 0.14 hours")
print(f"   At this data length, they cannot be resolved separately")
print(f"   Treating as combined constituent: K1+P1 at {TIDAL_CONSTITUENTS['K1+P1']:.3f}h")
print(f"\nForce-include: {', '.join(FORCE_INCLUDE_CONSTITUENTS)}")

# Load real data to get statistics
t_obs_real, y_obs_real = load_with_time(obs_file, start_time, dt_input)
t_model, y_model = load_with_time(model_file, start_time, dt_input)

if flip_model:
    y_model = -y_model

y_model_ugal = y_model / 10.0

min_len = min(len(y_obs_real), len(y_model_ugal))

if USE_SYNTHETIC_NOISE:
    # Calculate statistics from real data
    real_std = np.std(y_obs_real[:min_len])
    noise_std = real_std * NOISE_AMPLITUDE_MULTIPLIER
    
    # Generate selected noise type
    if NOISE_TYPE == 'pink':
        y_obs = generate_pink_noise(min_len, noise_std)
    else:
        y_obs = generate_white_noise(min_len, noise_std)
    
    t_obs = t_obs_real[:min_len]
    
    print(f"\n  CRITICAL: This is {NOISE_TYPE.upper()} NOISE with NO tidal signal!")
    print(f"   Real data std: {real_std:.2f} µGal")
    print(f"   Noise std: {noise_std:.2f} µGal (×{NOISE_AMPLITUDE_MULTIPLIER:.2f})")
elif USE_FITTED_NOISE:
    long_data_file = LONG_DATA_FILE
    y_obs = generate_fitted_noise(min_len, long_data_file, start_time, dt_input)
    t_obs = t_obs_real[:min_len]
else:
    y_obs = y_obs_real[:min_len]
    t_obs = t_obs_real[:min_len]
y_model_ugal = y_model_ugal[:min_len]

print(f"\nData: {len(y_obs)} points, {(t_obs[-1]-t_obs[0]).total_seconds()/3600:.2f} hours")

# Calculate frequency resolution
duration_hours = (t_obs[-1]-t_obs[0]).total_seconds()/3600
freq_resolution = 1.0 / duration_hours
period_resolution_at_24h = (24.0**2) * freq_resolution
print(f"Frequency resolution: {freq_resolution:.6f} hr⁻¹")
print(f"Period resolution at ~24h: {period_resolution_at_24h:.3f} hours")

# STEP 1: Detrend
y_obs_detrended, trend = remove_polynomial_drift(y_obs, REMOVE_DRIFT_ORDER)

# STEP 2: HIGH-PASS FILTER
if ENABLE_HIGHPASS:
    y_obs_highpassed, drift_component, highpass_mask = highpass_filter(
        y_obs_detrended, dt_input, HIGHPASS_CUTOFF_HOURS)
else:
    print(f"\n  HIGH-PASS FILTER DISABLED")
    y_obs_highpassed = y_obs_detrended
    drift_component = np.zeros_like(y_obs_detrended)
    highpass_mask = np.ones(len(y_obs_detrended), dtype=bool)

# STEP 3: IDENTIFY TIDAL PEAKS IN MODEL
print(f"\n{'='*80}")
print(f"ANALYSING TIDAL MODEL SPECTRUM")
print(f"{'='*80}")

model_peaks = identify_tidal_peaks(y_model_ugal, dt_input, PEAK_THRESHOLD, PEAK_WIDTH_HOURS)
model_peaks = force_include_missing_constituents(
    model_peaks, y_model_ugal, dt_input, FORCE_INCLUDE_CONSTITUENTS
)

print(f"\nIdentified {len(model_peaks['periods'])} tidal constituents in MODEL:")
print(f"\n{'Rank':<6}{'Label':<10}{'Period (h)':<12}{'Amplitude':<15}{'% of Max':<10}")
print(f"{'-'*65}")
max_amp = model_peaks['amplitudes'][0] if len(model_peaks['amplitudes']) > 0 else 1
for i, (label, period, amp) in enumerate(zip(
    model_peaks['labels'], 
    model_peaks['periods'], 
    model_peaks['amplitudes']
)):
    pct = 100 * amp / max_amp
    print(f"{i+1:<6}{label:<10}{period:<12.2f}{amp:<15.2e}{pct:<10.1f}")

print(f"\n{'='*80}")
if USE_SYNTHETIC_NOISE:
    print(f"WHITE NOISE CONSTITUENT RATIOS (EXPECT RANDOM VALUES)")
else:
    print(f"DATA CONSTITUENT RATIOS")
print(f"{'='*80}")
model_ratios, model_amps = calculate_amplitude_ratios(model_peaks, verbose=True)
if 'K1+P1' in model_amps:
    print(f"\n  Adjusting K1+P1: Assuming single-component detection")
    model_amps['K1+P1'] = model_amps['K1+P1'] / 2.0
    
    # Recalculate ratios involving K1+P1
    if 'O1' in model_amps:
        model_ratios['(K1+P1)/O1'] = model_amps['K1+P1'] / model_amps['O1']
    if 'M2' in model_amps:
        model_ratios['M2/(K1+P1)'] = model_amps['M2'] / model_amps['K1+P1']
    if 'S2' in model_amps:
        model_ratios['S2/(K1+P1)'] = model_amps['S2'] / model_amps['K1+P1']
# STEP 4: CREATE FILTER
filter_mask, freqs = create_narrowband_constituent_filter(
    model_peaks, len(y_obs_highpassed), dt_input, 
    bandwidth_hours=SPECTRAL_PEAK_BANDWIDTH_HOURS
)
model_centered = y_model_ugal - np.mean(y_model_ugal)
model_fft = fft(model_centered)

# STEP 5: Apply filter to data
filtered_obs, data_fft, filtered_fft = apply_filter(y_obs_highpassed, filter_mask)

# STEP 6: ANALYSE DATA SPECTRUM
print(f"\n{'='*80}")
print(f"ANALYZING OBSERVED DATA SPECTRUM")
print(f"{'='*80}")

data_peaks = identify_tidal_peaks(filtered_obs, dt_input, PEAK_THRESHOLD, PEAK_WIDTH_HOURS)

# FORCE INCLUDE MISSING CONSTITUENTS IN DATA
data_peaks = force_include_missing_constituents(
    data_peaks, filtered_obs, dt_input, FORCE_INCLUDE_CONSTITUENTS)

print(f"\nIdentified {len(data_peaks['periods'])} peaks in FILTERED DATA:")
print(f"\n{'Rank':<6}{'Label':<10}{'Period (h)':<12}{'Amplitude':<15}{'% of Max':<10}")
print(f"{'-'*65}")
max_amp = data_peaks['amplitudes'][0] if len(data_peaks['amplitudes']) > 0 else 1
for i, (label, period, amp) in enumerate(zip(
    data_peaks['labels'], 
    data_peaks['periods'], 
    data_peaks['amplitudes']
)):
    pct = 100 * amp / max_amp
    print(f"{i+1:<6}{label:<10}{period:<12.2f}{amp:<15.2e}{pct:<10.1f}")

print(f"\n{'='*80}")
print(f"DATA CONSTITUENT RATIOS")
print(f"{'='*80}")
data_ratios, data_amps = calculate_amplitude_ratios(data_peaks, verbose=True)

# STEP 8: COMPARE RATIOS
confidence, mean_error, quality_dist = compare_ratios_with_tiered_thresholds(
    model_ratios, data_ratios)

# STEP 9: Calibrate amplitude
scale_factor, filtered_obs_scaled, correlation, rms_diff, residual = \
    calibrate_amplitude(filtered_obs, y_model_ugal, MANUAL_SCALE_FACTOR)
# ================================
# PLOTTING
# ================================

fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True)

# Panel 1: Raw with trend
axes[0].plot(t_obs, y_obs, linewidth=0.8, alpha=0.7, label='Raw', color='blue')
if REMOVE_DRIFT_ORDER is not None:
    axes[0].plot(t_obs, trend, linewidth=2, label=f'Polynomial Trend (order {REMOVE_DRIFT_ORDER})', color='red', alpha=0.8)
    title_1 = f'Raw Data and Polynomial Trend (order {REMOVE_DRIFT_ORDER})'
else:
    axes[0].plot(t_obs, trend, linewidth=2, label='Mean', color='red', alpha=0.8, linestyle='--')
    title_1 = 'Raw Data (No Polynomial Detrending)'
axes[0].set_ylabel('Acceleration (µGal)')
axes[0].set_title(title_1)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)


# Panel 3: HIGH-PASS FILTERED
if ENABLE_HIGHPASS:
    axes[1].plot(t_obs, y_obs_highpassed, linewidth=0.8, color='purple', label='High-passed')
    axes[1].plot(t_obs, drift_component, linewidth=1.5, color='red', alpha=0.7, linestyle='--', label='Removed drift')
    axes[1].set_ylabel('Acceleration (µGal)')
    axes[1].set_title(f'After High-Pass Filter (>{HIGHPASS_CUTOFF_HOURS}h removed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].plot(t_obs, y_obs_detrended, linewidth=0.8, color='gray')
    axes[1].set_ylabel('Acceleration (µGal)')
    axes[1].set_title('High-Pass Filter DISABLED')
    axes[1].grid(True, alpha=0.3)

# Panel 4: Filtered (before scaling)
axes[2].plot(t_obs, filtered_obs, linewidth=1, label='Filtered (unscaled)', color='green', alpha=0.8)
axes[2].plot(t_obs, y_model_ugal, linewidth=1, label='TSoft Model', 
             color='red', linestyle='--', alpha=0.7)
axes[2].set_ylabel('Acceleration (µGal)')
axes[2].set_title(f'Extracted Tidal Component (Combined K1+P1)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Panel 5: After scaling
axes[3].plot(t_obs, filtered_obs_scaled, linewidth=1, label='Filtered', color='green')
axes[3].plot(t_obs, y_model_ugal, linewidth=1, label='TSoft Model', 
             color='red', linestyle='--', alpha=0.7)
axes[3].set_ylabel('Acceleration (µGal)')
axes[3].set_title(f'Filtered data TSoft model comparison')
axes[3].legend()
axes[3].grid(True, alpha=0.3)
axes[3].text(0.02, 0.98, 
             f'Correlation: {correlation:.4f}\nRMS diff: {rms_diff:.2f} µGal',
             transform=axes[4].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 6: Residual
axes[4].plot(t_obs, residual, linewidth=0.8, color='purple')
axes[4].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[4].set_xlabel('Time (UTC)')
axes[4].set_ylabel('Residual (µGal)')
axes[4].set_title('Residual: Scaled - TSoft Model')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()

# Frequency domain plots
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))

positive_mask = freqs > 0
freqs_pos = freqs[positive_mask]
periods_hours = 1.0 / (freqs_pos * 3600)
period_mask = (periods_hours >= 6) & (periods_hours <= 50)

# Calculate PSDs
N = len(y_obs_highpassed)
sample_rate = 1.0 / dt_input

data_fft_full = fft(y_obs_highpassed - np.mean(y_obs_highpassed))
data_psd = (np.abs(data_fft_full[positive_mask])**2) / (N * sample_rate)
model_psd = (np.abs(model_fft[positive_mask])**2) / (N * sample_rate)
filtered_psd = (np.abs(filtered_fft[positive_mask])**2) / (N * sample_rate)

data_psd_plot = data_psd[period_mask]
model_psd_plot = model_psd[period_mask]
filtered_psd_plot = filtered_psd[period_mask]
periods_plot = periods_hours[period_mask]

# Get filter response for overlay
filter_response_plot = filter_mask[positive_mask][period_mask]

# Data PSD (after high-pass) WITH FILTER OVERLAY
axes2[0, 0].semilogy(periods_plot, data_psd_plot, linewidth=1.5, color='purple', alpha=0.8, label='Data PSD')
# Add filter overlay as shaded regions
ax_twin = axes2[0, 0].twinx()
ax_twin.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                      step='mid', label='Filter Response')
ax_twin.set_ylim(0, 1.0)
ax_twin.set_ylabel('Filter Response', fontsize=10, color='orange')
ax_twin.tick_params(axis='y', labelcolor='orange')
axes2[0, 0].set_xlabel('Period (hours)', fontsize=11)
axes2[0, 0].set_ylabel('PSD (µGal²/Hz)', fontsize=11)
axes2[0, 0].set_title('PSD: After High-Pass Filter', fontsize=12, fontweight='bold')
axes2[0, 0].axvline(12.42, color='red', linestyle=':', linewidth=2, alpha=0.6, label='M2')
axes2[0, 0].axvline(24.00, color='green', linestyle=':', linewidth=2, alpha=0.6, label='K1+P1')
axes2[0, 0].set_xlim(6, 50)
axes2[0, 0].legend(loc='upper left', fontsize=10)
ax_twin.legend(loc='upper right', fontsize=10)
axes2[0, 0].grid(True, alpha=0.3)

# Model PSD with peak labels AND FILTER OVERLAY
axes2[0, 1].semilogy(periods_plot, model_psd_plot, linewidth=1.5, color='red', alpha=0.8, label='Model PSD')
# Add filter overlay
ax_twin2 = axes2[0, 1].twinx()
ax_twin2.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                       step='mid', label='Filter Response')
ax_twin2.set_ylim(0, 1.0)
ax_twin2.set_ylabel('Filter Response', fontsize=10, color='orange')
ax_twin2.tick_params(axis='y', labelcolor='orange')

# Add peak markers
for label, period, amp in zip(model_peaks['labels'][:5], 
                               model_peaks['periods'][:5], 
                               model_peaks['amplitudes'][:5]):
    if 6 <= period <= 50:
        period_idx = np.argmin(np.abs(periods_plot - period))
        psd_val = model_psd_plot[period_idx]
        axes2[0, 1].scatter([period], [psd_val], s=100, c='darkred', 
                           zorder=5, marker='o', edgecolors='black', linewidth=1.5)
        if label in TIDAL_CONSTITUENTS:
            axes2[0, 1].annotate(label, xy=(period, psd_val), xytext=(0, 10), 
                               textcoords='offset points', ha='center',
                               fontsize=9, fontweight='bold', color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

axes2[0, 1].set_xlabel('Period (hours)', fontsize=11)
axes2[0, 1].set_ylabel('PSD (µGal²/Hz)', fontsize=11)
axes2[0, 1].set_title('PSD: TSoft Model with Identified Peaks', fontsize=12, fontweight='bold')
axes2[0, 1].set_xlim(6, 50)
axes2[0, 1].legend(loc='upper left', fontsize=10)
ax_twin2.legend(loc='upper right', fontsize=10)
axes2[0, 1].grid(True, alpha=0.3)

# Filtered PSD with peak labels AND FILTER OVERLAY
axes2[0, 2].semilogy(periods_plot, filtered_psd_plot, linewidth=1.5, color='green', alpha=0.8, label='Filtered PSD')
# Add filter overlay
ax_twin3 = axes2[0, 2].twinx()
ax_twin3.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                       step='mid', label='Filter Response')
ax_twin3.set_ylim(0, 1.0)
ax_twin3.set_ylabel('Filter Response', fontsize=10, color='orange')
ax_twin3.tick_params(axis='y', labelcolor='orange')

# Add peak markers
for label, period, amp in zip(data_peaks['labels'][:5], 
                               data_peaks['periods'][:5], 
                               data_peaks['amplitudes'][:5]):
    if 6 <= period <= 50:
        period_idx = np.argmin(np.abs(periods_plot - period))
        psd_val = filtered_psd_plot[period_idx]
        axes2[0, 2].scatter([period], [psd_val], s=100, c='darkgreen', 
                           zorder=5, marker='o', edgecolors='black', linewidth=1.5)
        if label in TIDAL_CONSTITUENTS:
            axes2[0, 2].annotate(label, xy=(period, psd_val), xytext=(0, 10), 
                               textcoords='offset points', ha='center',
                               fontsize=9, fontweight='bold', color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

axes2[0, 2].set_xlabel('Period (hours)', fontsize=11)
axes2[0, 2].set_ylabel('PSD (µGal²/Hz)', fontsize=11)
axes2[0, 2].set_title('PSD: Extracted Tidal Signal with Identified Peaks', fontsize=12, fontweight='bold')
axes2[0, 2].set_xlim(6, 50)
axes2[0, 2].legend(loc='upper left', fontsize=10)
ax_twin3.legend(loc='upper right', fontsize=10)
axes2[0, 2].grid(True, alpha=0.3)

# High-pass filter mask
if ENABLE_HIGHPASS:
    highpass_display = highpass_mask[positive_mask][period_mask].astype(float)
    axes2[1, 0].fill_between(periods_plot, 0, highpass_display, alpha=0.7, color='purple', edgecolor='darkviolet', linewidth=1.5)
    axes2[1, 0].axvline(HIGHPASS_CUTOFF_HOURS, color='red', linestyle='--', linewidth=2, label=f'Cutoff: {HIGHPASS_CUTOFF_HOURS}h')
    axes2[1, 0].legend()
else:
    axes2[1, 0].text(0.5, 0.5, 'High-Pass\nDISABLED', ha='center', va='center',
                    transform=axes2[1, 0].transAxes, fontsize=14, fontweight='bold')

axes2[1, 0].set_xlabel('Period (hours)', fontsize=11)
axes2[1, 0].set_ylabel('Filter Response', fontsize=11)
axes2[1, 0].set_title(f'High-Pass Filter (cutoff={HIGHPASS_CUTOFF_HOURS}h)', fontsize=12, fontweight='bold')
axes2[1, 0].set_ylim(-0.05, 1.15)
axes2[1, 0].set_xlim(periods_plot[0], periods_plot[-1])
axes2[1, 0].grid(True, alpha=0.3)

# Tidal filter mask
filter_display = filter_mask[positive_mask][period_mask].astype(float)
axes2[1, 1].step(periods_plot, filter_display, where='mid', color='orange', linewidth=2)
axes2[1, 1].fill_between(periods_plot, 0, filter_display, alpha=0.7, color='orange', 
                         step='mid', edgecolor='darkorange', linewidth=1.5)
axes2[1, 1].set_xlabel('Period (hours)', fontsize=11)
axes2[1, 1].set_ylabel('Filter Response', fontsize=11)
axes2[1, 1].set_title(f'Tidal Filter (Combined K1+P1)', fontsize=12, fontweight='bold')
axes2[1, 1].set_ylim(-0.05, 1.15)
axes2[1, 1].set_xlim(periods_plot[0], periods_plot[-1])
axes2[1, 1].invert_xaxis() 
axes2[1, 1].grid(True, alpha=0.3)
# Mark constituent locations
for name, period in [('M2', 12.42), ('S2', 12.00), ('K1+P1', 24.00), ('O1', 25.82)]:
    if periods_plot[0] <= period <= periods_plot[-1]:
        axes2[1, 1].axvline(period, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        axes2[1, 1].text(period, 1.08, name, ha='center', fontsize=9, color='gray')

# Amplitude ratio comparison
axes2[1, 2].remove()
axes2[1, 2] = fig2.add_subplot(2, 3, 6)

# Prepare data for bar chart
common_ratios = [r for r in model_ratios.keys() if r in data_ratios]
if len(common_ratios) > 0:
    x_pos = np.arange(len(common_ratios))
    model_vals = [model_ratios[r] for r in common_ratios]
    data_vals = [data_ratios[r] for r in common_ratios]
    
    width = 0.35
    axes2[1, 2].bar(x_pos - width/2, model_vals, width, label='Model', color='red', alpha=0.7)
    axes2[1, 2].bar(x_pos + width/2, data_vals, width, label='Data', color='blue', alpha=0.7)
    
    axes2[1, 2].set_ylabel('Ratio Value', fontsize=11)
    axes2[1, 2].set_title('Constituent Amplitude Ratios', fontsize=12, fontweight='bold')
    axes2[1, 2].set_xticks(x_pos)
    axes2[1, 2].set_xticklabels(common_ratios, rotation=45, ha='right', fontsize=9)
    axes2[1, 2].legend()
    axes2[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Add agreement indicators
    for i, ratio_name in enumerate(common_ratios):
        model_val = model_ratios[ratio_name]
        data_val = data_ratios[ratio_name]
        pct_diff = 100 * abs(model_val - data_val) / model_val
        
        color = 'green' if pct_diff < 10 else 'orange' if pct_diff < 20 else 'red'
        marker = '✓' if pct_diff < 10 else '~' if pct_diff < 20 else '✗'
        
        y_max = max(model_val, data_val)
        axes2[1, 2].text(i, y_max * 1.1, marker, ha='center', fontsize=14, 
                        color=color, fontweight='bold')
else:
    axes2[1, 2].text(0.5, 0.5, 'Not enough\ncommon ratios', ha='center', va='center',
                    transform=axes2[1, 2].transAxes, fontsize=12)

plt.tight_layout()

plt.show()

fig_ratios_standalone, ax_standalone = plt.subplots(1, 1, figsize=(12, 8))

common_ratios = [r for r in model_ratios.keys() if r in data_ratios]
if len(common_ratios) > 0:
    x_pos = np.arange(len(common_ratios))
    model_vals = [model_ratios[r] for r in common_ratios]
    data_vals = [data_ratios[r] for r in common_ratios]
    
    width = 0.35
    ax_standalone.bar(x_pos - width/2, model_vals, width, label='Model', color='red', alpha=0.7, edgecolor='darkred', linewidth=1.5)
    ax_standalone.bar(x_pos + width/2, data_vals, width, label='Data', color='blue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
    
    ax_standalone.set_ylabel('Ratio Value', fontsize=13, fontweight='bold')
    ax_standalone.set_xlabel('Constituent Ratio', fontsize=13, fontweight='bold')
    ax_standalone.set_title('Constituent Amplitude Ratios: Model vs Data', fontsize=15, fontweight='bold', pad=20)
    ax_standalone.set_xticks(x_pos)
    ax_standalone.set_xticklabels(common_ratios, rotation=45, ha='right', fontsize=11)
    ax_standalone.legend(fontsize=12, loc='upper left')
    ax_standalone.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Calculate y-axis limit with MORE headroom for markers and percentages (35% extra space)
    max_val = max(max(model_vals), max(data_vals))
    ax_standalone.set_ylim(0, max_val * 1.35)
    
    # Add agreement indicators - MATCHING THE ORIGINAL PLOT
    for i, ratio_name in enumerate(common_ratios):
        model_val = model_ratios[ratio_name]
        data_val = data_ratios[ratio_name]
        pct_diff = 100 * abs(model_val - data_val) / model_val
        
        color = 'green' if pct_diff < 10 else 'orange' if pct_diff < 20 else 'red'
        marker = '✓' if pct_diff < 10 else '~' if pct_diff < 20 else '✗'
        
        y_max = max(model_val, data_val)
        
        # Add percentage difference FIRST (below marker)
        ax_standalone.text(i, y_max + 0.08, f'{pct_diff:.1f}%', ha='center', fontsize=9,
                          color=color, fontweight='bold', style='italic')
        
        # Add marker ABOVE percentage
        ax_standalone.text(i, y_max + 0.2, marker, ha='center', fontsize=16, 
                          color=color, fontweight='bold')
    
    # Add legend for quality markers in TOP RIGHT (replacing stats box)
    legend_text = '✓ Good (<10%)\n~ Fair (<20%)\n✗ Failed (≥20%)'
    ax_standalone.text(0.98, 0.97, legend_text, transform=ax_standalone.transAxes,
                      fontsize=11, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9, 
                              edgecolor='black', linewidth=1.5))

else:
    ax_standalone.text(0.5, 0.5, 'Not enough\ncommon ratios', ha='center', va='center',
                      transform=ax_standalone.transAxes, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"\n  Note: K1 and P1 treated as combined constituent due to insufficient")
print(f"   frequency resolution ({period_resolution_at_24h:.3f}h vs 0.14h separation)")