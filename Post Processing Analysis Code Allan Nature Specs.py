# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:57:21 2026

@author: migsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
from pathlib import Path
from scipy.signal import correlate, correlation_lags, find_peaks
import scipy.optimize as opt
from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import os
import glob
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import nature_config_v2 as nc
nc.configure_matplotlib_for_nature(use_working_size=False)


# Use the colour scheme throughout
colors = nc.GRAVIMETER_COLORS

# -------- PARAMETERS --------
COMPONENT_DIR = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tidal Model component dats/"
obs_file = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tsoft dat exports/Acceleration data 2 min bins.dat"
model_file = "D:/USB Drive Backup/processed_data_mats/Chunk 3 Analysis results/Tsoft dat exports/240725.dat"
STATIC_MIRROR_FILE = "D:/USB Drive Backup/System and Elec noise trace files/Background stability processing/Dats/Original Process StatMirr.dat"
start_time = pd.Timestamp("2025-07-23 21:30:10")
dt_input = 600.0  # seconds
flip_model = True
ELECTRONIC_NOISE_FLOOR_UGAL = 0.685803 
# MANUAL SCALING
MANUAL_SCALE_FACTOR = None

# Resonant frequency
ASSUMED_RESONANT_FREQ_HZ = 0.809

# HIGH-PASS FILTER
HIGHPASS_CUTOFF_HOURS = 50
ENABLE_HIGHPASS = False

# TIDAL CONSTITUENT IDENTIFICATION
PEAK_THRESHOLD = 0.3  
PEAK_WIDTH_HOURS = 0.5  # Wider to catch combined peaks
TOLERANCE_HOURS = 0.4

# Force include these constituents (now with combined K1+P1)
FORCE_INCLUDE_CONSTITUENTS = ['M2', 'S2', 'K1+P1', 'O1']

# Filter type selection
SPECTRAL_PEAK_BANDWIDTH_HOURS = 0.21  # Wider for combined peak

# Drift removal
REMOVE_DRIFT_ORDER = 3

# Known tidal constituents - includes combined K1+P1 since we can't split them
TIDAL_CONSTITUENTS = {
    'M2': 12.4206,
    'S2': 12.0000,
    'N2': 12.6583,
    'K1+P1': 23.9997,  # Average of K1 (23.9345) and P1 (24.0659)
    'O1': 25.8193,
    'Q1': 26.8684,
}

# Separate constituents for reference (not used in peak finding)
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
    """
    Add any missing constituents from force_list that weren't detected by peak finder.
    For K1+P1, extracts the combined peak using a wider window.
    """
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
        
        # For K1+P1, use wider extraction window to capture both components
        if const_name == 'K1+P1':
            # Window from K1-margin to P1+margin (captures full combined peak)
            extraction_bandwidth_hours = 0.25  # ±15 minutes gives 30 min total window
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
        
        # Extract power in this band (both positive and negative frequencies)
        pos_mask = (freqs >= freq_low) & (freqs <= freq_high)
        neg_mask = (freqs >= -freq_high) & (freqs <= -freq_low)
        extraction_mask = pos_mask | neg_mask
        
        extracted_fft = signal_fft[extraction_mask]
        
        if len(extracted_fft) > 0:
            amplitude = np.max(np.abs(extracted_fft))
        else:
            amplitude = 0.0
        
        # Add to peaks
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
    
    # Re-sort by amplitude
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
            # Use wider bandwidth for K1+P1
            if label == 'K1+P1':
                bw = 0.25  # Wider to capture both K1 and P1
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
    
    # Updated ratio definitions using K1+P1
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
            confidence = "MODERATE"
        else:
            print(f"⚠ LOW CONFIDENCE: Detection uncertain")
            confidence = "LOW"
        
        return confidence, mean_diff, quality_counts
    
    return "NO_DATA", None, quality_counts

def diagnose_missing_constituents(model_peaks, data_peaks):
    """Identify which constituents are missing from detection"""
    print(f"\n{'='*80}")
    print(f"CONSTITUENT DETECTION STATUS")
    print(f"{'='*80}")
    
    model_constituents = set([label for label in model_peaks['labels'] 
                             if label in TIDAL_CONSTITUENTS])
    data_constituents = set([label for label in data_peaks['labels'] 
                            if label in TIDAL_CONSTITUENTS])
    
    both = model_constituents & data_constituents
    model_only = model_constituents - data_constituents
    data_only = data_constituents - model_constituents
    
    print(f"\n✓ Detected in both: {sorted(both) if both else 'None'}")
    print(f"⚠ Only in model:    {sorted(model_only) if model_only else 'None'}")
    print(f"⚠ Only in data:     {sorted(data_only) if data_only else 'None'}")
    
    if 'K1+P1' in both:
        print(f"\n Note: K1+P1 is a combined constituent (K1 and P1 are unresolved at this data length)")
    
    return both, model_only, data_only

def resonant_frequency_correction(scale_factor, assumed_freq_hz=None):
    """Calculate corrected resonant frequency"""
    print(f"\n{'='*80}")
    print(f"RESONANT FREQUENCY CORRECTION")
    print(f"{'='*80}")
    
    freq_correction = np.sqrt(scale_factor)
    
    print(f"\nScale factor applied: {scale_factor:.6f}")
    print(f"Resonant frequency correction: ×{freq_correction:.6f}")
    
    if assumed_freq_hz is not None:
        corrected_freq = assumed_freq_hz * freq_correction
        print(f"\nIf you assumed f₀ = {assumed_freq_hz:.6f} Hz:")
        print(f"  Then f_corrected = {corrected_freq:.6f} Hz")
        print(f"  Period = {1.0/corrected_freq:.2f} seconds")
    
    return freq_correction

def cross_correlation_analysis(filtered_signal, model_signal, dt):
    """Analyse time lag between signals"""
    filt_norm = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    model_norm = (model_signal - np.mean(model_signal)) / np.std(model_signal)
    
    xcorr = correlate(filt_norm, model_norm, mode='full')
    lags = np.arange(-len(model_norm)+1, len(model_norm))
    
    peak_idx = np.argmax(xcorr)
    peak_lag = lags[peak_idx]
    peak_value = xcorr[peak_idx] / len(model_norm)
    
    print(f"\n{'='*80}")
    print(f"CROSS-CORRELATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Peak correlation: {peak_value:.4f}")
    print(f"Time lag: {peak_lag} samples ({peak_lag * dt / 3600:.2f} hours)")
    
    return xcorr, lags, peak_lag, peak_value


def calculate_tidal_psd_welch(data_signal, model_signal, dt, 
                               window='hann', nperseg=None, 
                               overlap_factor=0.5, detrend='linear'):
    """Calculate PSD using Welch's method"""
    
    sample_rate = 1.0 / dt
    n_points = len(data_signal)
    duration_hours = n_points * dt / 3600
    
    print(f"\n{'='*80}")
    print(f"WELCH PSD ANALYSIS")
    print(f"{'='*80}")
    print(f"  Data points: {n_points}")
    print(f"  Duration: {duration_hours:.2f} hours")
    
    if nperseg is None:
        nperseg = min(max(n_points // 4, 256), 8192)
    
    noverlap = int(nperseg * overlap_factor)
    
    freqs_data, psd_data = signal.welch(
        data_signal,
        sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling='density'
    )
    
    freqs_model, psd_model = signal.welch(
        model_signal,
        sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling='density'
    )
    
    return freqs_data, psd_data, psd_model, sample_rate

def plot_welch_psd_comparison(freqs, psd_data, psd_model, psd_filtered=None,
                                period_range=(6, 50), mark_constituents=True):
    """Plot Welch PSD with dual x-axes"""
    
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = 1.0 / (freqs * 3600)
    
    period_mask = (periods >= period_range[0]) & (periods <= period_range[1]) & np.isfinite(periods)
    periods_plot = periods[period_mask]
    freqs_plot = freqs[period_mask]
    psd_data_plot = psd_data[period_mask]
    psd_model_plot = psd_model[period_mask]
    
    if psd_filtered is not None:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        psd_filtered_plot = psd_filtered[period_mask]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    
    def add_period_axis(ax, freq_lim):
        ax_top = ax.twiny()
        ax_top.set_xlim(freq_lim[0], freq_lim[1])
        
        period_ticks = [6, 9, 12, 15, 18, 24, 30, 40, 50]
        freq_ticks = []
        period_labels = []
        
        for p in period_ticks:
            if period_range[0] <= p <= period_range[1]:
                f = 1.0 / (p * 3600)
                if freq_lim[0] <= f <= freq_lim[1]:
                    freq_ticks.append(f)
                    period_labels.append(f'{p}h')
        
        if len(freq_ticks) > 0:
            ax_top.set_xticks(freq_ticks)
            ax_top.set_xticklabels(period_labels, fontsize=10)
            ax_top.set_xlabel('Period', fontsize=12, fontweight='bold')
            ax_top.tick_params(axis='x', which='major', direction='in', length=4)
        
        return ax_top
    
    # Plot 1: Data after high-pass
    ax1 = axes[0]
    ax1.semilogy(freqs_plot, psd_data_plot, 'b-', linewidth=2, alpha=0.8, 
                 label='Data (after high-pass)')
    ax1.set_ylabel('PSD (µGal²/Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Welch Power Spectral Density Analysis - Tidal Band', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.4, which='both')
    
    freq_min = 1.0 / (period_range[1] * 3600)
    freq_max = 1.0 / (period_range[0] * 3600)
    ax1.set_xlim(freq_min, freq_max)
    
    from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator
    def freq_formatter(x, pos):
        return f'{x*1e6:.2f}'
    ax1.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ax1.set_xlabel('Frequency (µHz)', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax1.tick_params(axis='x', which='both', direction='in')
    ax1.tick_params(axis='x', which='minor', length=4)
    ax1.tick_params(axis='x', which='major', length=7)
    
    if mark_constituents:
        for name, period in TIDAL_CONSTITUENTS.items():
            if period_range[0] <= period <= period_range[1]:
                freq_const = 1.0 / (period * 3600)
                ax1.axvline(freq_const, color='red', linestyle='--', alpha=0.3, linewidth=1)
                if name == 'K1+P1':
                    ax1.text(freq_const, ax1.get_ylim()[1]*0.5, 'K1+P1', 
                            rotation=90, va='bottom', ha='right', fontsize=8, color='red', alpha=0.7)
    
    ax1.legend(loc='best', fontsize=11)
    add_period_axis(ax1, (freq_min, freq_max))
    
    # Plot 2: Model
    ax2 = axes[1]
    ax2.semilogy(freqs_plot, psd_model_plot, 'r-', linewidth=2, alpha=0.8,
                 label='Tsoft Model')
    ax2.set_ylabel('PSD (µGal²/Hz)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (µHz)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.4, which='both')
    ax2.set_xlim(freq_min, freq_max)
    
    ax2.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax2.tick_params(axis='x', which='both', direction='in')
    ax2.tick_params(axis='x', which='minor', length=4)
    ax2.tick_params(axis='x', which='major', length=7)
    
    if mark_constituents:
        for name, period in TIDAL_CONSTITUENTS.items():
            if period_range[0] <= period <= period_range[1]:
                freq_const = 1.0 / (period * 3600)
                ax2.axvline(freq_const, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    ax2.legend(loc='best', fontsize=11)
    add_period_axis(ax2, (freq_min, freq_max))
    
    # Plot 3: Filtered (if provided)
    if psd_filtered is not None:
        ax3 = axes[2]
        ax3.semilogy(freqs_plot, psd_filtered_plot, 'g-', linewidth=2, alpha=0.8,
                     label='Filtered Data')
        ax3.semilogy(freqs_plot, psd_model_plot, 'r--', linewidth=2, alpha=0.6,
                     label='Tsoft Model')
        ax3.set_ylabel('PSD (µGal²/Hz)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Frequency (µHz)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.4, which='both')
        ax3.set_xlim(freq_min, freq_max)
        ax3.legend(loc='best', fontsize=11)
        
        ax3.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
        ax3.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        ax3.tick_params(axis='x', which='both', direction='in')
        ax3.tick_params(axis='x', which='minor', length=4)
        ax3.tick_params(axis='x', which='major', length=7)
        
        add_period_axis(ax3, (freq_min, freq_max))
    
    plt.tight_layout()
    return fig
def plot_welch_psd_combined(freqs_data, psd_data, psd_model, 
                            
                                   freqs_filtered, psd_filtered,
                                   peaks, dt,
                                   period_range=(6, 50), mark_constituents=True,
                                   bandwidth_hours=0.215, k1p1_bandwidth=0.25):
    """
    Plot Welch PSD with all three signals on same axes.
    Filter response is reconstructed from peak locations and bandwidths.
    
    Parameters:
    -----------
    freqs_data, psd_data, psd_model : arrays
        Frequency and PSD for data and model
    freqs_filtered, psd_filtered : arrays  
        Frequency and PSD for filtered signal
    peaks : dict
        Dictionary with 'frequencies', 'labels', 'periods' from peak detection
    dt : float
        Sampling interval (seconds)
    period_range : tuple
        Period range to plot (hours)
    mark_constituents : bool
        Whether to mark constituent locations
    bandwidth_hours : float
        Default bandwidth for most constituents (hours)
    k1p1_bandwidth : float
        Bandwidth for K1+P1 (hours)
    """
    
    # Process data and model (same frequency array)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_data = 1.0 / (freqs_data * 3600)
    
    period_mask_data = (periods_data >= period_range[0]) & (periods_data <= period_range[1]) & np.isfinite(periods_data)
    periods_plot = periods_data[period_mask_data]
    freqs_plot = freqs_data[period_mask_data]
    psd_data_plot = psd_data[period_mask_data]
    psd_model_plot = psd_model[period_mask_data]
    
    # Process filtered (different frequency array)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods_filtered = 1.0 / (freqs_filtered * 3600)
    
    period_mask_filt = (periods_filtered >= period_range[0]) & (periods_filtered <= period_range[1]) & np.isfinite(periods_filtered)
    periods_filt_plot = periods_filtered[period_mask_filt]
    freqs_filt_plot = freqs_filtered[period_mask_filt]
    psd_filtered_plot = psd_filtered[period_mask_filt]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    
    def add_period_axis(ax, freq_lim):
        ax_top = ax.twiny()
        ax_top.set_xlim(freq_lim[0], freq_lim[1])
        
        period_ticks = [6, 9, 12, 15, 18, 24, 30, 40, 50]
        freq_ticks = []
        period_labels = []
        
        for p in period_ticks:
            if period_range[0] <= p <= period_range[1]:
                f = 1.0 / (p * 3600)
                if freq_lim[0] <= f <= freq_lim[1]:
                    freq_ticks.append(f)
                    period_labels.append(f'{p}h')
        
        if len(freq_ticks) > 0:
            ax_top.set_xticks(freq_ticks)
            ax_top.set_xticklabels(period_labels, fontsize=10)
            ax_top.set_xlabel('Period', fontsize=12, fontweight='bold')
            ax_top.tick_params(axis='x', which='major', direction='in', length=4)
        
        return ax_top
    
# Plot all three signals - UPDATED COLORS
    ax.semilogy(freqs_plot, psd_data_plot, '-', linewidth=1.0, alpha=0.8, 
                color=colors['psd_raw'],  # Was 'xkcd:dirty blue'
                label='Raw data (after high-pass)', zorder=2)
    
    ax.semilogy(freqs_filt_plot, psd_filtered_plot, '-', linewidth=2.0, alpha=0.95,
                color=colors['psd_filtered'],  # Was 'xkcd:medium green'
                label='Extracted tidal signal', zorder=4)
    
    ax.semilogy(freqs_plot, psd_model_plot, '--', linewidth=2.0, alpha=0.95,
                color=colors['tidal_model'],  # Was 'xkcd:vermillion'
                label='Tsoft model', zorder=3)
    

    
    # Constituent markers - UPDATED COLOR
    if mark_constituents:
        major_constituents = {'M2': 12.4206, 'S2': 12.0000, 'K1+P1': 23.9997, 'O1': 25.8193, 'MK3': 8.1772}
        for name, period in major_constituents.items():
            if period_range[0] <= period <= period_range[1]:
                freq_const = 1.0 / (period * 3600)
                ax.axvline(freq_const, color=colors['constituent_marker'],  # Was 'black'
                          linestyle=':', alpha=0.5, linewidth=1.0, zorder=5)
                y_pos = ax.get_ylim()[0] * 2
                ax.text(freq_const, y_pos, name, 
                       rotation=90, va='bottom', ha='right', 
                       fontsize=6, color=colors['constituent_marker'],  # Was 'black'
                       alpha=0.8, fontweight='bold', zorder=6)

    
    # Add secondary y-axis for filter response
    ax_filter = ax.twinx()
    
    # ============================================================================
    # RECONSTRUCTED FILTER RESPONSE FROM PEAKS + BANDWIDTHS
    # ============================================================================
    
    # Create high-resolution frequency array for smooth filter visualisation
    freq_min = 1.0 / (period_range[1] * 3600)
    freq_max = 1.0 / (period_range[0] * 3600)
    n_points = 10000  # High resolution for smooth rectangles
    filter_freqs_hires = np.linspace(freq_min, freq_max, n_points)
    filter_response_hires = np.zeros(n_points)
    
    # Build filter response from detected constituents
    print(f"\n{'='*80}")
    print(f"RECONSTRUCTING FILTER RESPONSE FOR VISUALIZATION")
    print(f"{'='*80}")
    
    for i, (peak_freq, label, period) in enumerate(zip(
        peaks['frequencies'], 
        peaks['labels'],
        peaks['periods']
    )):
        if label in ['M2', 'S2', 'N2', 'K1+P1', 'O1', 'Q1']:  # Named tidal constituents
            
            # Determine bandwidth
            if label == 'K1+P1':
                bw = k1p1_bandwidth
            else:
                bw = bandwidth_hours
            
            # Calculate frequency bounds
            period_low = period - bw
            period_high = period + bw
            
            if period_low > 0:
                freq_high = 1.0 / (period_low * 3600)
                freq_low = 1.0 / (period_high * 3600)
            else:
                freq_low = 0
                freq_high = 1.0 / (period_high * 3600)
            
            # Set response to 1.0 in this band
            band_mask = (filter_freqs_hires >= freq_low) & (filter_freqs_hires <= freq_high)
            filter_response_hires[band_mask] = 1.0
            
            print(f"  {label:<8}: Period {period:.3f}h, BW ±{bw:.3f}h → {period_low:.3f}h to {period_high:.3f}h")
            print(f"           Freq range: {freq_low*1e6:.3f} to {freq_high*1e6:.3f} µHz")
    
    # Add custom 8.2h MK3 band if it exists
    custom_period = 8.1772
    custom_bandwidth = 0.03
    if period_range[0] <= custom_period <= period_range[1]:
        period_low = custom_period - custom_bandwidth
        period_high = custom_period + custom_bandwidth
        freq_high = 1.0 / (period_low * 3600)
        freq_low = 1.0 / (period_high * 3600)
        
        band_mask = (filter_freqs_hires >= freq_low) & (filter_freqs_hires <= freq_high)
        filter_response_hires[band_mask] = 1.0
        
        print(f"  MK3      : Period {custom_period:.3f}h, BW ±{custom_bandwidth:.3f}h → {period_low:.3f}h to {period_high:.3f}h")
    
    print(f"\nFilter coverage: {100 * np.sum(filter_response_hires > 0) / len(filter_response_hires):.2f}% of displayed frequency range")
    
    # Filter response - UPDATED COLOR
    ax_filter.fill_between(filter_freqs_hires, 0, filter_response_hires, 
                           alpha=0.2, color=colors['filter_passband'],  # Was 'orange'
                           label='Filter passband', zorder=1)
    ax_filter.plot(filter_freqs_hires, filter_response_hires, 
                  linewidth=1.5, color=colors['filter_passband'], alpha=0.7, zorder=1)
    
    ax_filter.set_ylabel('Filter Response', fontsize=12, fontweight='bold', color='black')
    ax_filter.set_ylim(0, 1.0)
    ax_filter.tick_params(axis='y', labelcolor='black')
    
    # ============================================================================
    
    ax.set_xlim(freq_min, freq_max)
    ax_filter.set_xlim(freq_min, freq_max)
    
    
    # Update legend and labels
    ax.set_ylabel('PSD (µGal²/Hz)', fontsize=7, fontweight='bold')
    ax.set_xlabel('Frequency (µHz)', fontsize=7, fontweight='bold')
    
    ax.set_title('Power Spectral Density - Tidal Band', fontsize=7, fontweight='bold')
    ax.grid(True, alpha=0.4, which='both')
    
    from matplotlib.ticker import FuncFormatter, MaxNLocator, AutoMinorLocator
    def freq_formatter(x, pos):
        return f'{x*1e6:.2f}'
    ax.xaxis.set_major_formatter(FuncFormatter(freq_formatter))
    ax.set_xlabel('Frequency (µHz)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune='both'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax.tick_params(axis='x', which='both', direction='in')
    ax.tick_params(axis='x', which='minor', length=4)
    ax.tick_params(axis='x', which='major', length=7)
    
    if mark_constituents:
        major_constituents = {'M2': 12.4206, 'S2': 12.0000, 'K1+P1': 23.9997, 'O1': 25.8193, 'MK3': 8.1772}
        for name, period in major_constituents.items():
            if period_range[0] <= period <= period_range[1]:
                freq_const = 1.0 / (period * 3600)
                ax.axvline(freq_const, color='black', linestyle=':', alpha=0.5, linewidth=1.5, zorder=5)
                y_pos = ax.get_ylim()[0] * 2
                ax.text(freq_const, y_pos, name, 
                       rotation=90, va='bottom', ha='right', 
                       fontsize=9, color='black', alpha=0.8, fontweight='bold', zorder=6)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_filter.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11, framealpha=0.95)
    
    add_period_axis(ax, (freq_min, freq_max))
    
    plt.tight_layout()
    return fig
def find_allan_minimum_in_window(tau_seconds, avar, window_hours):
    """Find Allan variance minimum within a specified tau window"""
    tau_min_sec = window_hours[0] * 3600
    tau_max_sec = window_hours[1] * 3600
    window_mask = (tau_seconds >= tau_min_sec) & (tau_seconds <= tau_max_sec)
    
    if np.sum(window_mask) == 0:
        print(f"  ⚠ No tau points in window {window_hours[0]}-{window_hours[1]}h, using full range")
        return np.argmin(avar)
    
    # Index within windowed subset, mapped back to full array
    windowed_indices = np.where(window_mask)[0]
    local_min_idx = np.argmin(avar[window_mask])
    return windowed_indices[local_min_idx]   
# ================================
# AUTOCORRELATION STABILITY ANALYSIS
# ================================
# ================================
# STABILITY ANALYSIS PARAMETERS
# ================================
# Short timescale data (high resolution, 100Hz position data)
SHORT_DATA_FILE = "D:/USB Drive Backup/processed_data_mats/Chunk 1 Output/Processed_Data_Chunk_20250723-211410_to_20250731-123410_full_position_trace_corrected.mat"  # PLACEHOLDER - update this
SHORT_DATA_RAW_SAMPLE_RATE = 100.0  # Hz (100 Hz = 0.01s per sample)
SHORT_DATA_DOWNSAMPLE_TO = 50.0  # Downsample to this rate (Hz) - e.g., 10Hz = 0.1s per sample
SHORT_DATA_DURATION_HOURS = 21  # How much to load (hours) - 1-2 hours recommended
SHORT_DATA_START_TIME = pd.Timestamp("2025-07-23 21:30:10")  # Adjust as needed
SHORT_DATA_DT = 1/SHORT_DATA_DOWNSAMPLE_TO
# Harmonic oscillator parameters for position → acceleration conversion
OSCILLATOR_FREQUENCY_HZ = 0.809 # Measured resonant frequency
FREQ_SCALE_SQ = (OSCILLATOR_FREQUENCY_HZ / 0.809) ** 2
OSCILLATOR_MASS_GRAMS = 1.198  # Mass (for info only, cancels in calculation)

t_obs, y_obs = load_with_time(obs_file, start_time, dt_input)
# Long timescale data (already loaded in your main script)
# We'll use: y_obs (or y_obs_highpassed, filtered_obs_scaled, etc.)
LONG_DATA_SIGNAL = y_obs  # Choose which signal to analyse
LONG_DATA_DT = dt_input  # 600 seconds (10 minutes)
LONG_DATA_NAME = "Raw Data"

# Allan variance calculation parameters
SHORT_TAU_MIN_SECONDS = 6.0  # Minimum tau in seconds
SHORT_TAU_MAX_HOURS = 2699/3600 # Maximum tau for short data (hours) - overlap with long data
LONG_TAU_MIN_HOURS = 2700/3600  # Minimum tau for long data (hours) - overlap with short data
LONG_TAU_MAX_HOURS = 200000/3600  # Maximum tau for long data (hours)

N_POINTS_SHORT = 100  # Number of tau values for short timescale
N_POINTS_LONG = 100   # Number of tau values for long timescale
#Parameters for the Allan Shenanigans
ALLAN_LONG_MIN_WINDOW_HOURS = (1.0, 4.0)  # Only search for minimum within this tau range (hours)
ENABLE_JOIN_CORRECTION = True
JOIN_SHORT_ANCHOR_SECONDS = 600.0   # Last reliable point in short data
JOIN_LONG_ANCHOR_SECONDS  = 2700.0  # First reliable point in long data
def calculate_autocorrelation(signal, dt, max_lag_hours=72):
    """
    Calculate autocorrelation for stability assessment
    
    Parameters:
    -----------
    signal : array
        Time series signal
    dt : float
        Sampling interval (seconds)
    max_lag_hours : float
        Maximum lag to compute (hours)
    
    Returns:
    --------
    lags_hours : array
        Time lags in hours
    autocorr_normalized : array
        Normalized autocorrelation
    """
    
    # Center and normalize signal
    signal_centered = signal - np.mean(signal)
    signal_normalized = signal_centered / np.std(signal_centered)
    
    # Calculate autocorrelation
    autocorr = correlate(signal_normalized, signal_normalized, mode='full')
    lags = correlation_lags(len(signal_normalized), len(signal_normalized), mode='full')
    
    # Normalize by zero-lag value
    autocorr_normalized = autocorr / autocorr[len(autocorr)//2]
    
    # Convert lags to hours
    lags_hours = lags * dt / 3600
    
    # Limit to max_lag_hours
    lag_mask = np.abs(lags_hours) <= max_lag_hours
    lags_hours = lags_hours[lag_mask]
    autocorr_normalized = autocorr_normalized[lag_mask]
    
    return lags_hours, autocorr_normalized, lags[lag_mask]


def calculate_cross_correlation(signal1, signal2, dt, max_lag_hours=48):
    """
    Calculate cross-correlation between two signals to find phase offset
    
    Parameters:
    -----------
    signal1 : array
        First signal (e.g., filtered data)
    signal2 : array
        Second signal (e.g., model)
    dt : float
        Sampling interval (seconds)
    max_lag_hours : float
        Maximum lag to compute (hours)
    
    Returns:
    --------
    lags_hours : array
        Time lags in hours
    xcorr_normalized : array
        Normalized cross-correlation
    peak_lag_samples : int
        Lag at peak correlation (in samples)
    peak_lag_hours : float
        Lag at peak correlation (in hours)
    peak_corr : float
        Peak correlation value
    """
    
    # Ensure same length
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    # Center and normalise signals
    signal1_centered = signal1 - np.mean(signal1)
    signal1_normalized = signal1_centered / np.std(signal1_centered)
    
    signal2_centered = signal2 - np.mean(signal2)
    signal2_normalized = signal2_centered / np.std(signal2_centered)
    
    # Calculate cross-correlation
    xcorr = correlate(signal1_normalized, signal2_normalized, mode='full')
    lags = correlation_lags(len(signal1_normalized), len(signal2_normalized), mode='full')
    
    # Normalize
    xcorr_normalized = xcorr / len(signal1_normalized)
    
    # Convert lags to hours
    lags_hours = lags * dt / 3600
    
    # Limit to max_lag_hours
    lag_mask = np.abs(lags_hours) <= max_lag_hours
    lags_hours_limited = lags_hours[lag_mask]
    xcorr_limited = xcorr_normalized[lag_mask]
    lags_limited = lags[lag_mask]
    
    # Find peak
    peak_idx = np.argmax(xcorr_limited)
    peak_lag_samples = lags_limited[peak_idx]
    peak_lag_hours = lags_hours_limited[peak_idx]
    peak_corr = xcorr_limited[peak_idx]
    
    return lags_hours_limited, xcorr_limited, peak_lag_samples, peak_lag_hours, peak_corr


def identify_tidal_periods_in_autocorr(lags_hours, autocorr, expected_periods=[12.42, 23.93]):
    """
    Identify peaks in autocorrelation at expected tidal periods
    """
    
    print(f"\n{'='*80}")
    print(f"TIDAL PERIOD IDENTIFICATION IN AUTOCORRELATION")
    print(f"{'='*80}")
    
    # Only look at positive lags
    positive_mask = lags_hours > 0
    lags_pos = lags_hours[positive_mask]
    autocorr_pos = autocorr[positive_mask]
    
    # Find peaks
    peaks, properties = find_peaks(autocorr_pos, height=0.1, distance=10)
    
    peak_lags = lags_pos[peaks]
    peak_heights = autocorr_pos[peaks]
    
    print(f"\nDetected autocorrelation peaks:")
    print(f"{'Lag (hours)':<15}{'Correlation':<15}{'Expected Period':<20}")
    print(f"{'-'*50}")
    
    matched_periods = []
    for lag, height in zip(peak_lags, peak_heights):
        # Check if close to expected period
        matched = None
        for period in expected_periods:
            if abs(lag - period) < 1.0:  # Within 1 hour
                matched = f"{period:.2f}h"
                matched_periods.append(period)
                break
        
        match_str = matched if matched else "Unknown"
        print(f"{lag:<15.2f}{height:<15.3f}{match_str:<20}")
    
    if len(matched_periods) > 0:
        print(f"\n✓ Detected {len(matched_periods)} expected tidal period(s)")
    else:
        print(f"\n⚠ No clear tidal periods detected in autocorrelation")
    
    return peak_lags, peak_heights
def position_to_acceleration_harmonic_oscillator(position_array, 
                                                  oscillator_frequency_hz=0.327095, 
                                                  mass_grams=1.198):
    """
    Convert position to acceleration using harmonic oscillator physics
    
    For a harmonic oscillator: F = kx = ma
    Therefore: a = (k/m)x = ω²x
    
    Where:
    - ω = 2πf (angular frequency)
    - f = resonant frequency of the oscillator
    - Mass cancels out in the equation
    
    Parameters:
    -----------
    position_array : array
        Position in meters
    oscillator_frequency_hz : float
        Resonant frequency of your levitated mass (Hz)
    mass_grams : float
        Mass of levitated object in grams (informational only)
    
    Returns:
    --------
    acceleration_m_s2 : array
        Acceleration in m/s²
    acceleration_ugal : array
        Acceleration in µGal
    """
    
    print(f"\n{'='*80}")
    print(f"HARMONIC OSCILLATOR POSITION → ACCELERATION CONVERSION")
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
    print(f"  Conversion factor ω²: {omega**2:.4f} s⁻²")
    
    # Calculate acceleration: a = ω²x
    acceleration_m_s2 = (omega**2) * position_array
    
    # Convert to µGal
    acceleration_ugal = acceleration_m_s2 * 1e8
    
    print(f"\nConversion results:")
    print(f"  Position range: {np.min(position_array)*1e6:.4f} to {np.max(position_array)*1e6:.4f} µm")
    print(f"  Position std: {np.std(position_array)*1e6:.4f} µm")
    print(f"  Acceleration range: {np.min(acceleration_ugal):.4f} to {np.max(acceleration_ugal):.4f} µGal")
    print(f"  Acceleration std: {np.std(acceleration_ugal):.4f} µGal")
    
    return acceleration_m_s2, acceleration_ugal

#This downsampling is redundant but if you want to change your effective sample rate change your target freq
def downsample_data(data, original_rate_hz, target_rate_hz):
    """
    Downsample data by averaging consecutive samples
    
    Parameters:
    -----------
    data : array
        Original high-rate data
    original_rate_hz : float
        Original sampling rate (Hz)
    target_rate_hz : float
        Target sampling rate (Hz)
    
    Returns:
    --------
    downsampled_data : array
        Averaged/downsampled data
    """
    
    downsample_factor = int(original_rate_hz / target_rate_hz)
    
    if downsample_factor <= 1:
        print(f"  No downsampling needed (factor = {downsample_factor})")
        return data
    
    # Truncate to multiple of downsample factor
    n_complete = (len(data) // downsample_factor) * downsample_factor
    data_truncated = data[:n_complete]
    
    # Reshape and average
    data_reshaped = data_truncated.reshape(-1, downsample_factor)
    downsampled_data = np.mean(data_reshaped, axis=1)
    
    print(f"  Downsampling: {original_rate_hz}Hz → {target_rate_hz}Hz (factor of {downsample_factor})")
    print(f"  Original points: {len(data)}")
    print(f"  Downsampled points: {len(downsampled_data)}")
    print(f"  New dt: {1.0/target_rate_hz:.3f}s")
    
    return downsampled_data



def calculate_allan_variance_custom(signal, dt, tau_values_seconds):
    """
    Calculate Allan variance for specified tau values (averaging times)
    
    Parameters:
    -----------
    signal : array
        Time series data
    dt : float
        Sampling interval (seconds)
    tau_values_seconds : array
        Array of tau values to evaluate (seconds)
    
    Returns:
    --------
    tau_out : array
        Tau values actually computed (may be subset if some failed)
    avar_out : array
        Allan variance at each tau
    """
    
    signal_centered = signal - np.mean(signal)
    n_total = len(signal_centered)
    
    tau_out = []
    avar_out = []
    
    for tau_sec in tau_values_seconds:
        # Number of samples per bin
        n = int(tau_sec / dt)
        
        if n < 1:
            continue
        
        # Number of complete bins
        n_bins = n_total // n
        
        if n_bins < 2:
            continue
        
        # Reshape and calculate bin averages
        truncated_data = signal_centered[:n_bins * n]
        bins = truncated_data.reshape(n_bins, n)
        bin_averages = np.mean(bins, axis=1)
        
        # Allan variance
        differences = np.diff(bin_averages)
        allan_var = 0.5 * np.mean(differences**2)
        
        tau_out.append(tau_sec)
        avar_out.append(allan_var)
    
    return np.array(tau_out), np.array(avar_out)

def load_short_data(filename, max_duration_hours=None, start_time=None):
    """
    Load short timescale high-resolution data from .mat file
    
    Parameters:
    -----------
    filename : str
        Path to .mat file
    max_duration_hours : float, optional
        Maximum duration to load (hours)
    start_time : pd.Timestamp, optional
        Start time for time array
    
    Returns:
    --------
    time_array : pandas datetime array (or None if start_time not provided)
    data_array : numpy array (position data)
    dt : float (sampling interval in seconds)
    """
    
    print(f"  Loading .mat file: {filename}")
    
    # Load .mat file
    mat_data = loadmat(filename)
    
    # Print available variables to help identify the correct one
    print(f"  Available variables in .mat file: {list(mat_data.keys())}")
    
    # Try to find position data - adjust these keys based on your actual .mat structure
    # Common names: 'position', 'data', 'y', 'signal', etc.
    possible_keys = ['position', 'data', 'y', 'signal', 'pos', 'x']
    
    data_array = None
    for key in possible_keys:
        if key in mat_data:
            data_array = mat_data[key].flatten()
            print(f"  Found data under key: '{key}'")
            break
    
    if data_array is None:
        # If none of the common keys work, try the first non-metadata key
        for key in mat_data.keys():
            if not key.startswith('__'):
                data_array = mat_data[key].flatten()
                print(f"  Using data from key: '{key}' (first non-metadata key found)")
                break
    
    if data_array is None:
        raise ValueError("Could not find data array in .mat file. Please check the file structure.")
    
    # Try to find sampling rate or time information
    dt = None
    if 'dt' in mat_data:
        dt = float(mat_data['dt'][0, 0])
    elif 'meta_sample_rate_hz' in mat_data:  # ADD THIS LINE
        dt = 1.0 / float(mat_data['meta_sample_rate_hz'][0, 0])  
    elif 'fs' in mat_data:
        dt = 1.0 / float(mat_data['fs'][0, 0])
    elif 'sample_rate' in mat_data:
        dt = 1.0 / float(mat_data['sample_rate'][0, 0])
    
    # If dt not found in file, assume it's the raw sample rate
    if dt is None:
        print(f"  ⚠ Sampling rate not found in .mat file, assuming raw rate")
        dt = None  # Will be handled by caller
    else:
        print(f"  Sampling interval from file: {dt}s ({1.0/dt}Hz)")
    
    # Limit duration if requested
    if max_duration_hours is not None:
        if dt is None:
            raise ValueError("Cannot limit duration without knowing sampling rate")
        max_samples = int(max_duration_hours * 3600 / dt)
        if len(data_array) > max_samples:
            data_array = data_array[:max_samples]
            print(f"  Truncated to {max_duration_hours} hours ({max_samples} samples)")
    
    print(f"  Loaded {len(data_array)} samples")
    
    # Create time array if start time provided
    time_array = None
    if start_time is not None and dt is not None:
        time_array = start_time + pd.to_timedelta(np.arange(len(data_array)) * dt, unit='s')
    
    return time_array, data_array, dt
def detect_spikes_absolute(data, threshold_ugal=800):
    """
    Detect spikes using absolute threshold on residuals from median filter
    
    Parameters:
    -----------
    data : array
        Time series data
    threshold_ugal : float
        Absolute threshold in µGal
    
    Returns:
    --------
    spike_mask : array (bool)
        True where spikes detected
    """
    from scipy.ndimage import median_filter
    
    # Use rolling median to be robust to spikes
    window_size = min(51, len(data)//10)
    if window_size % 2 == 0:  # Must be odd
        window_size += 1
    
    smoothed = median_filter(data, size=window_size)
    residual = data - smoothed
    
    # Absolute threshold
    spike_mask = np.abs(residual) > threshold_ugal
    
    return spike_mask


def apply_spike_mask(data, mask):
    """
    Replace spikes with NaN for Allan variance calculation
    
    Parameters:
    -----------
    data : array
        Time series data
    mask : array (bool)
        True where spikes are located
    
    Returns:
    --------
    masked_data : array
        Data with NaN at spike locations
    """
    masked_data = data.copy()
    masked_data[mask] = np.nan
    return masked_data


def calculate_allan_variance_custom(signal, dt, tau_values_seconds):
    """
    Calculate Allan variance for specified tau values
    NOW HANDLES NaN VALUES from spike masking
    
    Parameters:
    -----------
    signal : array
        Time series data (may contain NaN)
    dt : float
        Sampling interval (seconds)
    tau_values_seconds : array
        Array of tau values to evaluate (seconds)
    
    Returns:
    --------
    tau_out : array
        Tau values actually computed
    avar_out : array
        Allan variance at each tau
    """
    
    # Remove NaN values first (from spike masking)
    valid_mask = ~np.isnan(signal)
    signal_valid = signal[valid_mask]
    
    if len(signal_valid) < 100:
        print(f"  ⚠️ Too few valid points after spike removal: {len(signal_valid)}")
        return np.array([]), np.array([])
    
    signal_centered = signal_valid - np.mean(signal_valid)
    n_total = len(signal_centered)
    
    tau_out = []
    avar_out = []
    
    for tau_sec in tau_values_seconds:
        # Number of samples per bin
        n = int(tau_sec / dt)
        
        if n < 1:
            continue
        
        # Number of complete bins
        n_bins = n_total // n
        
        if n_bins < 2:
            continue
        
        # Reshape and calculate bin averages
        truncated_data = signal_centered[:n_bins * n]
        bins = truncated_data.reshape(n_bins, n)
        bin_averages = np.mean(bins, axis=1)
        
        # Allan variance
        differences = np.diff(bin_averages)
        allan_var = 0.5 * np.mean(differences**2)
        
        tau_out.append(tau_sec)
        avar_out.append(allan_var)
    
    return np.array(tau_out), np.array(avar_out)

# ========================================
# CALCULATE ALLAN VARIANCE - LONG TIMESCALE
# ========================================

print(f"\n{'='*80}")
print(f"STITCHED ALLAN VARIANCE ANALYSIS")
print(f"{'='*80}")

print(f"\n1. LONG TIMESCALE (BINNED DATA)")
print(f"{'='*80}")
print(f"  Signal: {LONG_DATA_NAME}")
print(f"  Sampling interval: {LONG_DATA_DT}s ({LONG_DATA_DT/60:.1f} min)")
print(f"  Data points: {len(LONG_DATA_SIGNAL)}")
print(f"  Duration: {len(LONG_DATA_SIGNAL) * LONG_DATA_DT / 3600:.1f} hours")

# Create logarithmically spaced tau values for long timescale
tau_min_long_sec = LONG_TAU_MIN_HOURS * 3600
tau_max_long_sec = LONG_TAU_MAX_HOURS * 3600
tau_long_seconds = np.logspace(np.log10(tau_min_long_sec), 
                                np.log10(tau_max_long_sec), 
                                N_POINTS_LONG)

# Calculate Allan variance
tau_long, avar_long = calculate_allan_variance_custom(
    LONG_DATA_SIGNAL, LONG_DATA_DT, tau_long_seconds
)

print(f"  Tau range: {tau_long[0]:.0f}s to {tau_long[-1]:.0f}s")
print(f"  Tau range: {tau_long[0]/3600:.2f}h to {tau_long[-1]/3600:.1f}h")
print(f"  Computed {len(tau_long)} points")

# Find minimum
min_idx_long = find_allan_minimum_in_window(tau_long, avar_long, ALLAN_LONG_MIN_WINDOW_HOURS)
print(f"  Minimum: τ={tau_long[min_idx_long]/3600:.2f}h, σ²={avar_long[min_idx_long]:.2e} µGal²")

def quantify_autocorr_decay(lags_hours, autocorr, tidal_period=24.0):
    """
    Quantify the decay of autocorrelation peaks to assess stability
    
    Parameters:
    -----------
    lags_hours : array
        Time lags in hours
    autocorr : array
        Autocorrelation values
    tidal_period : float
        Primary tidal period to track (hours)
    
    Returns:
    --------
    tau : float
        Decay time constant (hours)
    coherence_time : float
        Time to drop to 50% correlation (hours)
    peak_values : list
        Correlation at each tidal cycle
    """
    
    print(f"\n{'='*80}")
    print(f"AUTOCORRELATION DECAY QUANTIFICATION")
    print(f"{'='*80}")
    
    # Find peaks at multiples of tidal period
    peak_times = []
    peak_values = []
    
    for n in range(1, 6):  # Check up to 5 cycles
        target_lag = n * tidal_period
        if target_lag > np.max(lags_hours):
            break
        
        # Find actual peak near target
        window = 2.0  # hours
        near_target = (lags_hours >= target_lag - window) & (lags_hours <= target_lag + window)
        
        if np.sum(near_target) > 0:
            local_autocorr = autocorr[near_target]
            local_lags = lags_hours[near_target]
            peak_idx = np.argmax(local_autocorr)
            
            peak_times.append(local_lags[peak_idx])
            peak_values.append(local_autocorr[peak_idx])
    
    peak_times = np.array(peak_times)
    peak_values = np.array(peak_values)
    
    print(f"\nPeak correlation at tidal cycles:")
    print(f"{'Cycle':<10}{'Time (h)':<12}{'Correlation':<15}{'Retention':<12}")
    print(f"{'-'*50}")
    for i, (t, val) in enumerate(zip(peak_times, peak_values), 1):
        retention = 100 * val
        print(f"{i:<10}{t:<12.1f}{val:<15.3f}{retention:<12.1f}%")
    
    # Fit exponential decay: corr(t) = exp(-t/τ)
    def exp_decay(t, tau):
        return np.exp(-t / tau)
    
    try:
        if len(peak_times) >= 2:
            tau_fit, _ = opt.curve_fit(exp_decay, peak_times, peak_values, p0=[100])
            tau = tau_fit[0]
            
            # Coherence time (where correlation drops to 0.5)
            coherence_50 = -tau * np.log(0.5)
            
            # Coherence time (1/e)
            coherence_1e = tau
            
            print(f"\n{'='*80}")
            print(f"DECAY METRICS")
            print(f"{'='*80}")
            print(f"Decay time constant (τ): {tau:.1f} hours")
            print(f"  → At time τ, correlation drops to {100*np.exp(-1):.1f}% of original")
            print(f"Coherence time (50% threshold): {coherence_50:.1f} hours")
            print(f"  → Signal maintains >50% correlation for {coherence_50:.1f} hours")
            print(f"Coherence time (1/e threshold): {coherence_1e:.1f} hours")
            
            # Quality assessment
            if tau > 100:
                print(f"\n✓✓ Excellent stability (τ > 100h)")
            elif tau > 50:
                print(f"\n✓ Good stability (τ > 50h)")
            else:
                print(f"\n~ Moderate stability (τ < 50h)")
            
            return tau, coherence_50, peak_values
        else:
            print(f"\n⚠ Insufficient peaks for decay fitting (need ≥2)")
            return None, None, peak_values
    
    except Exception as e:
        print(f"\n⚠ Could not fit decay curve: {e}")
        return None, None, peak_values
def detect_and_mask_spikes_absolute(data, time_array, threshold_ugal=800, padding_minutes=20):
    """Detect spikes using absolute threshold in µGal"""
    from scipy.ndimage import median_filter
    
    # Use rolling median to be robust to spikes
    smoothed = median_filter(data, size=51)  # ~8.5 hour window
    residual = data - smoothed
    
    # Absolute threshold
    spike_mask = np.abs(residual) > threshold_ugal
    
    # Expand mask to include padding on either side
    padding_samples = int((padding_minutes * 60) / dt_input)
    
    # Find spike locations and expand
    spike_indices = np.where(spike_mask)[0]
    expanded_mask = np.zeros(len(data), dtype=bool)
    
    for idx in spike_indices:
        start_idx = max(0, idx - padding_samples)
        end_idx = min(len(data), idx + padding_samples + 1)
        expanded_mask[start_idx:end_idx] = True
    
    print(f"\n{'='*80}")
    print(f"SPIKE DETECTION AND MASKING (ABSOLUTE THRESHOLD)")
    print(f"{'='*80}")
    print(f"  Threshold: ±{threshold_ugal} µGal")
    print(f"  Padding: ±{padding_minutes} minutes")
    print(f"  Initial spike points: {np.sum(spike_mask)} ({100*np.sum(spike_mask)/len(data):.2f}%)")
    print(f"  After padding: {np.sum(expanded_mask)} ({100*np.sum(expanded_mask)/len(data):.2f}%)")
    
    # Count contiguous masked regions
    mask_changes = np.diff(expanded_mask.astype(int))
    n_regions = np.sum(mask_changes == 1)
    print(f"  Number of masked regions: {n_regions}")
    
    return expanded_mask, spike_indices

def apply_join_correction(tau_short, avar_short, tau_long, avar_long,
                          short_anchor_sec=600.0, long_anchor_sec=2000.0):
    """
    Bridge the join region between short and long timescale Allan deviation curves.
    
    Points between short_anchor_sec and long_anchor_sec are nudged toward a
    log-linear interpolant connecting the two anchor values, while preserving
    their point-to-point variation (relative deviations from local trend).
    
    Parameters:
    -----------
    tau_short, avar_short : arrays
        Short timescale tau (seconds) and Allan variance
    tau_long, avar_long : arrays
        Long timescale tau (seconds) and Allan variance
    short_anchor_sec : float
        Tau value in short data to use as left anchor (seconds)
    long_anchor_sec : float
        Tau value in long data to use as right anchor (seconds)
    
    Returns:
    --------
    tau_short_corr, avar_short_corr : arrays
        Corrected short timescale arrays
    tau_long_corr, avar_long_corr : arrays
        Corrected long timescale arrays
    """
    
    tau_short_corr = tau_short.copy()
    avar_short_corr = avar_short.copy()
    tau_long_corr = tau_long.copy()
    avar_long_corr = avar_long.copy()
    
    # --- Find anchor values ---
    # Left anchor: nearest point in short data at or below short_anchor_sec
    short_anchor_idx = np.argmin(np.abs(tau_short - short_anchor_sec))
    anchor_tau_left = tau_short[short_anchor_idx]
    anchor_adev_left = np.sqrt(avar_short[short_anchor_idx])  # Allan deviation
    
    # Right anchor: nearest point in long data at or above long_anchor_sec
    long_anchor_idx = np.argmin(np.abs(tau_long - long_anchor_sec))
    anchor_tau_right = tau_long[long_anchor_idx]
    anchor_adev_right = np.sqrt(avar_long[long_anchor_idx])
    
    print(f"\n{'='*80}")
    print(f"JOIN CORRECTION")
    print(f"{'='*80}")
    print(f"  Left anchor:  τ={anchor_tau_left:.1f}s,  σ={anchor_adev_left:.4f} µGal")
    print(f"  Right anchor: τ={anchor_tau_right:.1f}s, σ={anchor_adev_right:.4f} µGal")
    
    # --- Log-linear target between anchors ---
    # In log space: straight line from (log_tau_left, log_adev_left) to (log_tau_right, log_adev_right)
    log_tau_left  = np.log10(anchor_tau_left)
    log_tau_right = np.log10(anchor_tau_right)
    log_adev_left  = np.log10(anchor_adev_left)
    log_adev_right = np.log10(anchor_adev_right)
    
    def target_adev(tau_sec):
        """Log-linear interpolant between anchors"""
        log_tau = np.log10(tau_sec)
        t = (log_tau - log_tau_left) / (log_tau_right - log_tau_left)
        t = np.clip(t, 0.0, 1.0)
        return 10 ** (log_adev_left + t * (log_adev_right - log_adev_left))
    
    # --- Correct short timescale: points from short_anchor_sec upward ---
    join_mask_short = tau_short >= short_anchor_sec
    n_short_corrected = np.sum(join_mask_short)
    
    if n_short_corrected > 0:
        # Smooth the short curve in the join region to get local trend
        adev_short_join = np.sqrt(avar_short[join_mask_short])
        tau_short_join  = tau_short[join_mask_short]
        
        # Compute ratio of actual values to target values
        # Apply correction multiplicatively (on variance = adev²)
        for i, idx in enumerate(np.where(join_mask_short)[0]):
            tau_val = tau_short[idx]
            adev_actual = np.sqrt(avar_short[idx])
            adev_target = target_adev(tau_val)
            
            # Fractional progress through window (0 at anchor, 1 at end)
            t = (np.log10(tau_val) - log_tau_left) / (log_tau_right - log_tau_left)
            t = np.clip(t, 0.0, 1.0)
            
            # Weight: apply full correction at anchor, taper to zero at tau_max_short
            # So we pull the tail of short data UP toward the bridge line
            correction_weight = t  # Linear blend: 0 at left anchor, 1 at right anchor
            
            # Target for this point: blend between actual and interpolated target
            adev_corrected = adev_actual + correction_weight * (adev_target - adev_actual)
            avar_short_corr[idx] = adev_corrected ** 2
        
        print(f"  Short data corrected: {n_short_corrected} points (τ >= {short_anchor_sec:.0f}s)")
    
    # --- Correct long timescale: points from long_anchor_sec downward ---
    join_mask_long = tau_long <= long_anchor_sec
    n_long_corrected = np.sum(join_mask_long)
    
    if n_long_corrected > 0:
        for i, idx in enumerate(np.where(join_mask_long)[0]):
            tau_val = tau_long[idx]
            adev_actual = np.sqrt(avar_long[idx])
            adev_target = target_adev(tau_val)
            
            # Progress: 0 at left anchor (tau=600), 1 at right anchor (tau=2000)
            t = (np.log10(tau_val) - log_tau_left) / (log_tau_right - log_tau_left)
            t = np.clip(t, 0.0, 1.0)
            
            # Weight: full correction at right anchor, taper to zero at left anchor
            correction_weight = 1.0 - t
            
            adev_corrected = adev_actual + correction_weight * (adev_target - adev_actual)
            avar_long_corr[idx] = adev_corrected ** 2
        
        print(f"  Long data corrected: {n_long_corrected} points (τ <= {long_anchor_sec:.0f}s)")
    
    print(f"  Bridge line slope: {(log_adev_right - log_adev_left)/(log_tau_right - log_tau_left):.3f} (log-log)")
    
    return tau_short_corr, avar_short_corr, tau_long_corr, avar_long_corr

# ========================================================================================================
# MAIN EXECUTION
# ===============================================================================================

print(f"\n{'='*80}")
print(f"TIDAL DETECTION WITH COMBINED K1+P1 CONSTITUENT")
print(f"{'='*80}")
print(f"ℹ️  K1 (23.93h) and P1 (24.07h) are separated by only 0.14 hours")
print(f"   At this data length, they cannot be resolved separately")
print(f"   Treating as combined constituent: K1+P1 at {TIDAL_CONSTITUENTS['K1+P1']:.3f}h")
print(f"\nForce-include: {', '.join(FORCE_INCLUDE_CONSTITUENTS)}")

# Load data
t_obs, y_obs = load_with_time(obs_file, start_time, dt_input)
t_model, y_model = load_with_time(model_file, start_time, dt_input)

if flip_model:
    y_model = -y_model

y_model_ugal = y_model / 10.0

min_len = min(len(y_obs), len(y_model_ugal))
y_obs = y_obs[:min_len]
y_model_ugal = y_model_ugal[:min_len]
t_obs = t_obs[:min_len]

print(f"\nData: {len(y_obs)} points, {(t_obs[-1]-t_obs[0]).total_seconds()/3600:.2f} hours")

# Calculate frequency resolution
duration_hours = (t_obs[-1]-t_obs[0]).total_seconds()/3600
freq_resolution = 1.0 / duration_hours
period_resolution_at_24h = (24.0**2) * freq_resolution
print(f"Frequency resolution: {freq_resolution:.6f} hr⁻¹")
print(f"Period resolution at ~24h: {period_resolution_at_24h:.3f} hours")
print(f"K1-P1 separation (0.14h) is {0.14/period_resolution_at_24h:.2f}× smaller than resolution")
print(f"\n{'='*80}") 
print(f"DETRENDING: DISABLED")
print(f"{'='*80}")
print(f"⚠️  Using raw data with mean removed only")

mean_val = np.mean(y_obs)
y_obs_detrended = y_obs - mean_val
trend = np.full_like(y_obs, mean_val)


# STEP 2: HIGH-PASS FILTER
if ENABLE_HIGHPASS:
    y_obs_highpassed, drift_component, highpass_mask = highpass_filter(
        y_obs_detrended, dt_input, HIGHPASS_CUTOFF_HOURS)
else:
    print(f"\n⚠️  HIGH-PASS FILTER DISABLED")
    y_obs_highpassed = y_obs_detrended
    drift_component = np.zeros_like(y_obs_detrended)
    highpass_mask = np.ones(len(y_obs_detrended), dtype=bool)

# STEP 3: IDENTIFY TIDAL PEAKS IN MODEL
print(f"\n{'='*80}")
print(f"ANALYSING TIDAL MODEL SPECTRUM")
print(f"{'='*80}")

model_peaks = identify_tidal_peaks(y_model_ugal, dt_input, PEAK_THRESHOLD, PEAK_WIDTH_HOURS)

# FORCE INCLUDE MISSING CONSTITUENTS IN MODEL
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
print(f"MODEL CONSTITUENT RATIOS")
print(f"{'='*80}")
model_ratios, model_amps = calculate_amplitude_ratios(model_peaks, verbose=True)

if 'K1+P1' in model_amps:
    print(f"\n⚠️  Adjusting K1+P1: Assuming single-component detection")
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
# Add custom band for the ~8.2h feature (not used in ratios)
print(f"\n{'='*80}")
print(f"ADDING CUSTOM BAND FOR ~33H FEATURE")
print(f"{'='*80}")

custom_period = 8.1772  # hours (center of the 8.1-8.3h feature)
custom_bandwidth = 0.03  # hours (±0.02h to cover 31.5-35.5h range)

period_low = custom_period - custom_bandwidth
period_high = custom_period + custom_bandwidth

freq_high = 1.0 / (period_low * 3600)
freq_low = 1.0 / (period_high * 3600)

pos_mask = (freqs >= freq_low) & (freqs <= freq_high)
neg_mask = (freqs >= -freq_high) & (freqs <= -freq_low)
custom_mask = pos_mask | neg_mask

filter_mask[custom_mask] = 1.0

n_bins_added = np.sum(pos_mask)
print(f"  Added band: {period_low:.1f}h to {period_high:.1f}h")
print(f"  Frequency range: {freq_low:.6e} to {freq_high:.6e} Hz")
print(f"  Additional bins: {n_bins_added}")
print(f"  Note: This feature will NOT be used in ratio calculations")
# STEP 5: Apply filter to data
filtered_obs, data_fft, filtered_fft = apply_filter(y_obs_highpassed, filter_mask)

# WELCH PSD ANALYSIS
freqs_welch, psd_data_welch, psd_model_welch, sample_rate = calculate_tidal_psd_welch(
    y_obs_highpassed,
    y_model_ugal,
    dt_input,
    window='hann',
    nperseg=len(y_obs_highpassed),
    overlap_factor=0.75,
    detrend='constant'
)

freqs_filt_welch, psd_filtered_welch = signal.welch(
    filtered_obs,
    sample_rate,
    window='hann',
    nperseg=len(y_obs_highpassed),
    noverlap=int(len(y_obs_highpassed) * 0.5),
    detrend='constant',
    scaling='density'
)


# These should already exist in your code from the frequency domain analysis:
positive_mask = freqs > 0
freqs_pos = freqs[positive_mask]
periods_hours = 1.0 / (freqs_pos * 3600)
period_mask = (periods_hours >= 6) & (periods_hours <= 50)

N = len(y_obs_highpassed)
sample_rate = 1.0 / dt_input

# Calculate PSDs the same way as before (simple FFT)
data_fft_full = fft(y_obs_highpassed - np.mean(y_obs_highpassed))
data_psd = (np.abs(data_fft_full[positive_mask])**2) / (N * sample_rate)

model_centered = y_model_ugal - np.mean(y_model_ugal)
model_fft = fft(model_centered)
model_psd = (np.abs(model_fft[positive_mask])**2) / (N * sample_rate)

filtered_psd = (np.abs(filtered_fft[positive_mask])**2) / (N * sample_rate)

# Now plot with the combined function
fig_spec = plot_welch_psd_combined(
    freqs_pos, data_psd, model_psd,
    freqs_pos, filtered_psd,
    model_peaks, dt_input,  # Pass peaks and dt instead of filter_mask
    period_range=(6, 50),
    mark_constituents=True,
    bandwidth_hours=SPECTRAL_PEAK_BANDWIDTH_HOURS,  # 0.215
    k1p1_bandwidth=0.25
)


plt.show()
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

# STEP 7: DIAGNOSE MISSING CONSTITUENTS
both, model_only, data_only = diagnose_missing_constituents(model_peaks, data_peaks)

# STEP 8: COMPARE RATIOS
confidence, mean_error, quality_dist = compare_ratios_with_tiered_thresholds(
    model_ratios, data_ratios)

# STEP 9: Calibrate amplitude
scale_factor, filtered_obs_scaled, correlation, rms_diff, residual = \
    calibrate_amplitude(filtered_obs, y_model_ugal, MANUAL_SCALE_FACTOR)

# STEP 10: Resonant frequency correction
freq_correction = resonant_frequency_correction(scale_factor, ASSUMED_RESONANT_FREQ_HZ)

# STEP 11: Cross-correlation
xcorr, lags, peak_lag, peak_corr = cross_correlation_analysis(
    filtered_obs_scaled, y_model_ugal, dt_input)

# ================================
# PLOTTING
# ================================

# Frequency domain plots - 2x2 layout for filter construction
fig, axes2 = plt.subplots(2, 2, figsize=nc.get_figure_size('double', 0.85))

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

# (a) Model PSD with peak labels AND FILTER OVERLAY
axes2[0, 0].semilogy(periods_plot, model_psd_plot, linewidth=1.5, color='red', alpha=0.8, label='Model PSD')
# Add filter overlay
ax_twin = axes2[0, 0].twinx()
ax_twin.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                      step='mid', label='Filter Passbands')
ax_twin.set_ylim(0, 1.0)
ax_twin.set_ylabel('Filter Response', fontsize=10, color='black')
ax_twin.tick_params(axis='y', labelcolor='black')

# Add peak markers
for label, period, amp in zip(model_peaks['labels'][:5], 
                               model_peaks['periods'][:5], 
                               model_peaks['amplitudes'][:5]):
    if 6 <= period <= 50:
        period_idx = np.argmin(np.abs(periods_plot - period))
        psd_val = model_psd_plot[period_idx]
        axes2[0, 0].scatter([period], [psd_val], s=100, c='darkred', 
                           zorder=5, marker='o', edgecolors='black', linewidth=1.5)
        if label in TIDAL_CONSTITUENTS:
            axes2[0, 0].annotate(label, xy=(period, psd_val), xytext=(0, 10), 
                               textcoords='offset points', ha='center',
                               fontsize=9, fontweight='bold', color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

axes2[0, 0].set_xlabel('Period (hours)', fontsize=11)
axes2[0, 0].set_ylabel('PSD ($\mu$Gal$^2$/Hz)', fontsize=11)
axes2[0, 0].set_title('(a) Tsoft Model PSD with Filter Passbands', fontsize=12, fontweight='bold')
axes2[0, 0].set_xlim(6, 50)
axes2[0, 0].legend(loc='upper left', fontsize=10)
ax_twin.legend(loc='upper right', fontsize=10)
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].set_ylim(bottom=None, top=axes2[0, 0].get_ylim()[1] * 1.4)
# (b) Filter response function
filter_display = filter_mask[positive_mask][period_mask].astype(float)
axes2[0, 1].step(periods_plot, filter_display, where='mid', color='orange', linewidth=2)
axes2[0, 1].fill_between(periods_plot, 0, filter_display, alpha=0.7, color='orange', 
                         step='mid', edgecolor='darkorange', linewidth=1.5)
axes2[0, 1].set_xlabel('Period (hours)', fontsize=11)
axes2[0, 1].set_ylabel('Filter Response', fontsize=11)
axes2[0, 1].set_title('(b) Narrow-Band Brick-Wall Filter', fontsize=12, fontweight='bold')
axes2[0, 1].set_ylim(-0.05, 1.15)
axes2[0, 1].set_xlim(periods_plot[0], periods_plot[-1])
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].invert_xaxis()
# Mark constituent locations
for name, period in [('M2', 12.42), ('S2', 12.00), ('K1+P1', 24.00), ('O1', 25.82)]:
    if periods_plot[0] <= period <= periods_plot[-1]:
        axes2[0, 1].axvline(period, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        axes2[0, 1].text(period, 1.08, name, ha='center', fontsize=9, color='gray')

# (c) Data PSD (after high-pass) WITH FILTER OVERLAY
axes2[1, 0].semilogy(periods_plot, data_psd_plot, linewidth=1.5, color='purple', alpha=0.8, label='Data PSD')
# Add filter overlay as shaded regions
ax_twin2 = axes2[1, 0].twinx()
ax_twin2.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                      step='mid', label='Filter Passbands')
ax_twin2.set_ylim(0, 1.0)
ax_twin2.set_ylabel('Filter Response', fontsize=10, color='black')
ax_twin2.tick_params(axis='y', labelcolor='black')
axes2[1, 0].set_xlabel('Period (hours)', fontsize=11)
axes2[1, 0].set_ylabel('PSD ($\mu$Gal$^2$/Hz)', fontsize=11)
axes2[1, 0].set_title('(c) Observed Data PSD with Filter Passbands', fontsize=12, fontweight='bold')
axes2[1, 0].set_xlim(6, 50)
axes2[1, 0].legend(loc='upper left', fontsize=10)
ax_twin2.legend(loc='upper right', fontsize=10)
axes2[1, 0].grid(True, alpha=0.3)

# (d) Filtered PSD with peak labels
axes2[1, 1].semilogy(periods_plot, filtered_psd_plot, linewidth=1.5, color='green', alpha=0.8, label='Extracted Signal')

# Add filter overlay
ax_twin3 = axes2[1, 1].twinx()
ax_twin3.fill_between(periods_plot, 0, filter_response_plot, alpha=0.2, color='orange', 
                       step='mid', label='Filter Passbands')
ax_twin3.set_ylim(0, 1.0)
ax_twin3.set_ylabel('Filter Response', fontsize=10, color='black')
ax_twin3.tick_params(axis='y', labelcolor='black')

# Add peak markers
for label, period, amp in zip(data_peaks['labels'][:5], 
                               data_peaks['periods'][:5], 
                               data_peaks['amplitudes'][:5]):
    if 6 <= period <= 50:
        period_idx = np.argmin(np.abs(periods_plot - period))
        psd_val = filtered_psd_plot[period_idx]
        axes2[1, 1].scatter([period], [psd_val], s=100, c='darkgreen', 
                           zorder=5, marker='o', edgecolors='black', linewidth=1.5)
        if label in TIDAL_CONSTITUENTS:
            axes2[1, 1].annotate(label, xy=(period, psd_val), xytext=(0, 10), 
                               textcoords='offset points', ha='center',
                               fontsize=9, fontweight='bold', color='darkgreen',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

axes2[1, 1].set_xlabel('Period (hours)', fontsize=11)
axes2[1, 1].set_ylabel('PSD ($\mu$Gal$^2$/Hz)', fontsize=11)
axes2[1, 1].set_title('(d) Extracted Tidal Signal PSD', fontsize=12, fontweight='bold')
axes2[1, 1].set_xlim(6, 50)
axes2[1, 1].legend(loc='upper left', fontsize=10)
ax_twin3.legend(loc='upper right', fontsize=10)
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].set_ylim(bottom=None, top=axes2[1, 1].get_ylim()[1] * 1.2)
plt.tight_layout()
fig.savefig('Filter Drsign 2x2.pdf', dpi=nc.NATURE_DPI, bbox_inches='tight', transparent=False)
plt.show()
# ================================
# RAW DATA COMPARISON WITH COMPREHENSIVE MASKING
# ================================

# Apply high-pass for drift removal (for visualisation only)
DRIFT_REMOVAL_CUTOFF_HOURS = 30  # Remove >60h drift for visualisation

y_obs_drift_removed, drift_removed, _ = highpass_filter(
    y_obs, dt_input, DRIFT_REMOVAL_CUTOFF_HOURS
)

# Detect spikes and apply padding

# Use absolute threshold
spike_mask, spike_indices = detect_and_mask_spikes_absolute(
    y_obs_drift_removed, t_obs, threshold_ugal=550, padding_minutes=30
)

# Mask edge effects from filtering
EDGE_TRIM_PERCENT = 6  # Trim 5% from each edge
edge_trim_samples = int(EDGE_TRIM_PERCENT / 100 * len(t_obs))

edge_mask = np.zeros(len(t_obs), dtype=bool)
edge_mask[:edge_trim_samples] = True
edge_mask[-edge_trim_samples:] = True

print(f"\n{'='*80}")
print(f"EDGE EFFECT MASKING")
print(f"{'='*80}")
print(f"  Trimming: {EDGE_TRIM_PERCENT}% from each edge")
print(f"  Edge samples masked: {2 * edge_trim_samples} ({100*np.sum(edge_mask)/len(edge_mask):.2f}%)")
print(f"  Edge duration: {edge_trim_samples * dt_input / 3600:.1f} hours on each side")

# Combine masks
full_mask = spike_mask | edge_mask

print(f"\n{'='*80}")
print(f"TOTAL MASKING SUMMARY")
print(f"{'='*80}")
print(f"  Spike-masked points: {np.sum(spike_mask)} ({100*np.sum(spike_mask)/len(spike_mask):.2f}%)")
print(f"  Edge-masked points: {np.sum(edge_mask)} ({100*np.sum(edge_mask)/len(edge_mask):.2f}%)")
print(f"  Total masked: {np.sum(full_mask)} ({100*np.sum(full_mask)/len(full_mask):.2f}%)")
print(f"  Valid data: {np.sum(~full_mask)} ({100*np.sum(~full_mask)/len(full_mask):.2f}%)")

# Create masked versions of data
y_obs_masked = y_obs_drift_removed.copy()
y_obs_masked[full_mask] = np.nan

filtered_masked = filtered_obs_scaled.copy()
filtered_masked[full_mask] = np.nan

model_masked = y_model_ugal.copy()
model_masked[full_mask] = np.nan

# Define 7-day window for spring tide visualisation (Aug 4-11)
window_start = pd.Timestamp("2025-08-04 00:00:00")
window_end = pd.Timestamp("2025-08-12 00:00:00")

# Create mask for the time window
time_window_mask = (t_obs >= window_start) & (t_obs <= window_end)

# Apply time window to data
t_obs_windowed = t_obs[time_window_mask]
y_obs_windowed = y_obs_masked[time_window_mask]
filtered_windowed = filtered_masked[time_window_mask]
model_windowed = model_masked[time_window_mask]
full_mask_windowed = full_mask[time_window_mask]

print(f"\n{'='*80}")
print(f"TIME WINDOW FOR SPRING TIDE VISUALISATION")
print(f"{'='*80}")
print(f"  Window: {window_start} to {window_end}")
print(f"  Duration: {(window_end - window_start).total_seconds() / 3600:.1f} hours (~7 days)")
print(f"  Data points in window: {np.sum(time_window_mask)}")

# Create single-panel comparison figure with xkcd colors
# REPLACED WITH:
fig_raw, ax = plt.subplots(1, 1, figsize=nc.get_figure_size('double', 0.55))

# Mark masked regions with light gray shaded background FIRST (bottom layer)
in_masked_region = False
for i in range(len(full_mask_windowed)):
    if full_mask_windowed[i] and not in_masked_region:
        start_t = t_obs_windowed[i]
        in_masked_region = True
    elif not full_mask_windowed[i] and in_masked_region:
        end_t = t_obs_windowed[i-1]
        ax.axvspan(start_t, end_t, alpha=0.15, 
          color=nc.GRAYS['very_light'],  # Was 'xkcd:battleship gray'
          zorder=1)
        in_masked_region = False

# Handle case where mask extends to end
if in_masked_region:
    end_t = t_obs_windowed.iloc[-1]
    ax.axvspan(start_t, end_t, alpha=0.15, color='xkcd:battleship gray', zorder=1)

# Plot raw data in subdued blue-gray
ax.plot(t_obs_windowed, y_obs_windowed, linewidth=0.8, 
       color=colors['raw_data'],  # Was 'xkcd:dirty blue'
       alpha=0.4, label='Raw data (drift removed)', zorder=2)

# Plot filtered data in prominent xkcd green
ax.plot(t_obs_windowed, filtered_windowed, linewidth=2.0, 
       color=colors['extracted_tidal'],  # Was 'xkcd:lawn green'
       alpha=0.95, label='Extracted tidal signal', zorder=4)

# Plot model in prominent xkcd red/orange
ax.plot(t_obs_windowed, model_windowed, linewidth=2.0, 
       color=colors['tidal_model'],  # Was 'xkcd:vermillion'
       linestyle='--', alpha=0.95, label='Tsoft tidal model', zorder=3)

# Update labels and title
ax.set_ylabel('Acceleration (µGal)', fontsize=7, fontweight='bold')
ax.set_xlabel('Time (UTC)', fontsize=7, fontweight='bold')
ax.set_title('Tidal Signal Extraction: Spring Tide Period', fontsize=8, fontweight='bold')
ax.legend(loc='upper right', fontsize=6, framealpha=0.95)

# ADD panel label (Nature requirement)
nc.add_panel_labels(ax, start_letter='a')
ax.grid(True, alpha=0.3)
ax.set_xlim(window_start, window_end)

# Calculate correlations on valid data only (within the time window)
valid_mask_windowed = ~full_mask_windowed
if np.sum(valid_mask_windowed) > 0:
    corr_raw_filtered = np.corrcoef(
        y_obs_drift_removed[time_window_mask][valid_mask_windowed], 
        filtered_obs_scaled[time_window_mask][valid_mask_windowed]
    )[0, 1]
    
    corr_raw_model = np.corrcoef(
        y_obs_drift_removed[time_window_mask][valid_mask_windowed], 
        y_model_ugal[time_window_mask][valid_mask_windowed]
    )[0, 1]
    
    corr_filt_model = np.corrcoef(
        filtered_obs_scaled[time_window_mask][valid_mask_windowed], 
        y_model_ugal[time_window_mask][valid_mask_windowed]
    )[0, 1]
    
    print(f"\n{'='*80}")
    print(f"CORRELATIONS (SPRING TIDE WINDOW ONLY)")
    print(f"{'='*80}")
    print(f"  Raw vs Filtered: {corr_raw_filtered:.4f}")
    print(f"  Raw vs Model: {corr_raw_model:.4f}")
    print(f"  Filtered vs Model: {corr_filt_model:.4f}")
else:
    corr_raw_filtered = np.nan
    corr_raw_model = np.nan
    corr_filt_model = np.nan

# Add note about masked regions at bottom
#fig_raw.text(0.5, 0.01, 
#            'Grey shaded regions: masked data (spikes and highpass edge effects excluded)', 
#            ha='center', fontsize=9, style='italic', color='xkcd:vermillion')

plt.tight_layout(rect=[0, 0.02, 1, 1])
# Save as PDF
fig_raw.savefig('tidal_signal_spring_tide_week.pdf', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved as 'tidal_signal_spring_tide_week.pdf'")
plt.show()

# Print summary stats for the window
if np.sum(valid_mask_windowed) > 0:
    print(f"\n{'='*80}")
    print(f"SPRING TIDE WINDOW STATISTICS")
    print(f"{'='*80}")
    print(f"Raw data std (valid only): {np.std(y_obs_drift_removed[time_window_mask][valid_mask_windowed]):.2f} µGal")
    print(f"Extracted tidal std (valid only): {np.std(filtered_obs_scaled[time_window_mask][valid_mask_windowed]):.2f} µGal")
    print(f"Model std (valid only): {np.std(y_model_ugal[time_window_mask][valid_mask_windowed]):.2f} µGal")
else:
    print(f"ERROR: No valid data remaining after masking in the selected window!")

# =================================================================================================================================
# SUN-MOON ALIGNMENT WITH FULL TIDAL COMPARISON
# =======================================================================================================================================

print(f"\n{'='*80}")
print(f"CALCULATING SUN-MOON ALIGNMENT FOR SPRING/NEAP VISUALIZATION")
print(f"{'='*80}")

def calculate_sun_moon_earth_angle(timestamps):
    """
    Calculate the angle between Sun-Earth-Moon alignment
    0° = New Moon (alignment: Sun-Moon-Earth) - SPRING TIDE
    90° = First Quarter (perpendicular) - NEAP TIDE
    180° = Full Moon (alignment: Sun-Earth-Moon) - SPRING TIDE
    270° = Last Quarter (perpendicular) - NEAP TIDE
    
    Uses Sturgeon Moon (Aug 9, 2025 08:55 UTC) as 180° reference
    """
    # Reference: Aug 9, 2025 08:55 UTC - Sturgeon Moon (full moon) = 180°
    reference_full_moon = pd.Timestamp("2025-08-09 00:00:00")
    synodic_month_days = 29.53059  # Average lunar month
    
    # Days since reference full moon
    days_since = (timestamps - reference_full_moon).total_seconds() / (24 * 3600)
    
    # Phase angle: 180° at full moon reference point
    phase_angle = ((days_since / synodic_month_days * 360) + 180) % 360
    
    return phase_angle



def calculate_tidal_amplitude_from_angle(angle):
    """
    Calculate relative tidal amplitude based on Sun-Earth-Moon angle
    
    Tidal force ∝ |cos(angle)| for the solar-lunar gravitational combination
    Maximum at 0° (new moon) and 180° (full moon) - spring tides
    Minimum at 90° and 270° (quarters) - neap tides
    """
    # Convert to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate alignment factor
    # cos(angle) gives: 1 at 0°, -1 at 180°, 0 at 90°/270°
    # We want high amplitude at both 0° and 180°
    # So we use |cos(angle)| to make both peaks positive
    alignment = np.abs(np.cos(angle_rad))
    
    # Scale to realistic tidal range: 
    # Moon contributes ~2.17x more than Sun to tidal forcing
    # At spring: Moon + Sun aligned = 1.0 (normalized)
    # At neap: Moon - Sun perpendicular ≈ 0.46 (from (2.17-1)/(2.17+1))
    tidal_amplitude = 0.46 + 0.54 * alignment
    
    return tidal_amplitude
# Calculate for full time range
sun_moon_angle = calculate_sun_moon_earth_angle(t_obs)
tidal_amplitude = calculate_tidal_amplitude_from_angle(sun_moon_angle)

# Identify spring and neap periods based on angle
# Spring: near 0° or 180° (within ±30°)
spring_mask = (sun_moon_angle <= 30) | (sun_moon_angle >= 330) | ((sun_moon_angle >= 150) & (sun_moon_angle <= 210))
# Neap: near 90° or 270° (within ±30°)
neap_mask = ((sun_moon_angle >= 60) & (sun_moon_angle <= 120)) | ((sun_moon_angle >= 240) & (sun_moon_angle <= 300))

print(f"  Reference: Sturgeon Moon (Full Moon) - Aug 9, 2025 08:55 UTC = 180°")
print(f"  Location: Southampton, UK")
print(f"  Sun-Moon-Earth angles calculated")
print(f"  Spring tide periods: {np.sum(spring_mask)} points ({100*np.sum(spring_mask)/len(spring_mask):.1f}%)")
print(f"  Neap tide periods: {np.sum(neap_mask)} points ({100*np.sum(neap_mask)/len(neap_mask):.1f}%)")

# Calculate approximate dates of key lunar phases
reference_full_moon = pd.Timestamp("2025-08-09 08:55:00")
synodic_month_days = 29.53059

print(f"\n  Approximate lunar phase dates:")
print(f"    Previous New Moon (Spring): {(reference_full_moon - pd.Timedelta(days=synodic_month_days/2)).strftime('%Y-%m-%d %H:%M')}")
print(f"    Full Moon/Sturgeon (Spring): {reference_full_moon.strftime('%Y-%m-%d %H:%M')}")
print(f"    Next Third Quarter (Neap): {(reference_full_moon + pd.Timedelta(days=synodic_month_days/4)).strftime('%Y-%m-%d %H:%M')}")
print(f"    Next New Moon (Spring): {(reference_full_moon + pd.Timedelta(days=synodic_month_days/2)).strftime('%Y-%m-%d %H:%M')}")

# ================================
# FULL DATASET WITH ALIGNMENT TRACE (SINGLE PANEL)
# ================================

print(f"\n{'='*80}")
print(f"CREATING FULL FIGURE WITH SUN-MOON ALIGNMENT TRACE")
print(f"{'='*80}")
window_start = pd.Timestamp("2025-07-25 12:00:00")
window_end = pd.Timestamp("2025-08-19 06:00:00")

# Create mask for the time window
time_window_mask = (t_obs >= window_start) & (t_obs <= window_end)

# Get alignment data for this window
sun_moon_angle_windowed = sun_moon_angle[time_window_mask]
# Use FULL time range (no windowing)
t_obs_plot = t_obs[time_window_mask]
y_obs_plot = y_obs_masked[time_window_mask]
filtered_plot = filtered_masked[time_window_mask]
model_plot = model_masked[time_window_mask]
full_mask_plot =  full_mask[time_window_mask]

print(f"  Full duration: {(t_obs[-1] - t_obs[0]).total_seconds() / 3600:.1f} hours")
print(f"  Data points: {len(t_obs)}")

# Recalculate sun-moon angle for the full dataset (to ensure lengths match)
sun_moon_angle_plot = calculate_sun_moon_earth_angle(t_obs_plot)

# Create figure with single panel
fig_spring, ax_data = plt.subplots(1, 1, figsize=nc.get_figure_size('double', 0.55))

# Create secondary y-axis for alignment labels (but same scale)
ax_align = ax_data.twinx()

# ========================================
# PRIMARY AXIS: Acceleration data (including alignment trace)
# ========================================

# Mark masked regions with light gray shaded background
in_masked_region = False
for i in range(len(full_mask_plot)):
    if full_mask_plot[i] and not in_masked_region:
        start_t = t_obs_plot[i]
        in_masked_region = True
    elif not full_mask_plot[i] and in_masked_region:
        end_t = t_obs_plot[i-1]
        ax_data.axvspan(start_t, end_t, alpha=0.15, color='xkcd:battleship gray', zorder=1)
        in_masked_region = False

if in_masked_region:
    end_t = t_obs_plot[-1]
    ax_data.axvspan(start_t, end_t, alpha=0.15, color='xkcd:battleship gray', zorder=1)

# Calculate alignment trace scaled to ±600 µGal, centered at 0
# cos(2*angle) gives: +1 at 0° and 180° (spring), -1 at 90° and 270° (neap)
ALIGNMENT_AMPLITUDE = 600  # µGal
alignment_normalized = np.cos(2 * np.deg2rad(sun_moon_angle_plot))  # Range: -1 to +1
alignment_trace = ALIGNMENT_AMPLITUDE * alignment_normalized  # Range: -600 to +600 µGal

print(f"\nAlignment trace statistics:")
print(f"  Length: {len(alignment_trace)} (should match t_obs_plot: {len(t_obs_plot)})")
print(f"  Min: {np.min(alignment_trace):.1f} µGal")
print(f"  Max: {np.max(alignment_trace):.1f} µGal")
print(f"  Mean: {np.mean(alignment_trace):.1f} µGal")

# Plot alignment trace FIRST (so it's in the background)
# Alignment trace - UPDATED COLOURBLIND VER
ax_data.plot(t_obs_plot, alignment_trace, linewidth=3.0, 
            color=colors['sun_moon_alignment'],  # Was 'xkcd:golden yellow'
            alpha=0.4, label='Sun-Moon alignment', zorder=2, linestyle='-')

# Fill spring tide regions (alignment > 0)
# Spring tide regions - UPDATED COLOUR
vermillion_rgba = nc.mpl.colors.to_rgba(colors['spring_tide'])
ax_data.fill_between(t_obs_plot, 0, alignment_trace, 
                     where=(alignment_trace >= 0), 
                     alpha=0.05, color=vermillion_rgba[0:3],  # Was 'red'
                     zorder=1, label='Spring tide periods')

# Fill neap tide regions (alignment < 0)
# Neap tide regions - UPDATED COLOUR
sky_blue_rgba = nc.mpl.colors.to_rgba(colors['neap_tide'])
ax_data.fill_between(t_obs_plot, 0, alignment_trace, 
                     where=(alignment_trace < 0), 
                     alpha=0.05, color=sky_blue_rgba[0:3],  # Was 'blue'
                     zorder=1, label='Neap tide periods')

# Plot raw data in subdued blue-gray
ax_data.plot(t_obs_plot, y_obs_plot, linewidth=0.8, 
            color=colors['raw_data'],  # Was 'xkcd:dirty blue'
            alpha=0.2, label='Raw data (drift removed)', zorder=3)

# Plot filtered data in prominent green
ax_data.plot(t_obs_plot, filtered_plot, linewidth=2.0, 
            color=colors['extracted_tidal'],  # Was 'xkcd:lawn green'
            alpha=0.7, label='Extracted tidal signal', zorder=5)

# Plot model in prominent red/orange
ax_data.plot(t_obs_plot, model_plot, linewidth=2.0, 
            color=colors['tidal_model'],  # Was 'xkcd:vermillion'
            linestyle='--', alpha=0.7, label='Tsoft tidal model', zorder=4)

# Mark the Sturgeon Moon
# Reference lines - UPDATED COLOR
reference_full_moon = pd.Timestamp("2025-08-09 00:00:00")
ax_data.axvline(reference_full_moon, 
               color=colors['highlighted_event'],  # Was 'orange'
               linestyle=':', linewidth=2.0, alpha=0.9, zorder=6, 
               label='Sturgeon Moon (Aug 9)')

# Add horizontal reference line at 0
ax_data.axhline(0, color=colors['reference_line'],  # Was 'black'
               linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)

# ========================================
# Axis formatting
# ========================================

# Set x-axis limits
ax_data.set_xlim(t_obs_plot[0], t_obs_plot[-1])

# PRIMARY AXIS (acceleration)
ax_data.set_ylabel('Acceleration (µGal)', fontsize=7, fontweight='bold')
ax_data.set_xlabel('Time (UTC:YYYY-MM-DD)', fontsize=7, fontweight='bold')
ax_data.legend(loc='lower left', fontsize=7, framealpha=0.8, ncol=2)  # 6pt!
ax_data.tick_params(axis='y', labelsize=7)  # 6pt!
ax_data.tick_params(axis='x', labelsize=7)  # 6pt!
ax_data.grid(True, alpha=0.3)

# SECONDARY AXIS - alignment labels
ax_align.set_ylim(ax_data.get_ylim())
ax_align.set_ylabel('Sun-Moon\nAlignment', fontsize=7, fontweight='bold',
                   rotation=270, labelpad=15)
ax_align.set_yticks([600, 0, -600])  # Positions!
ax_align.set_yticklabels(['Spring\n(Aligned)', 'Transition', 'Neap\n(Perpendicular)'], 
                          fontsize=7)  # 6pt, no color argument!
ax_align.tick_params(axis='y')  # No labelcolor argument!

# Title
#ax_data.set_title('Tidal Signal Detection: Spring-Neap Cycle (Southampton, UK)', 
#                 fontsize=14, fontweight='bold')
# ========================================
# INSET: Zoomed view of Aug 5-9 (4 days at 2x magnification)
# ========================================

# Define inset window: midday Aug 5 to midday Aug 9
inset_start = pd.Timestamp("2025-08-05 12:00:00")
inset_end = pd.Timestamp("2025-08-09 12:00:00")
plt.tight_layout(rect=[0, 0.025, 1, 1])
# Create inset axes aligned with right edge of main plot
# Position: [left, bottom, width, height] in figure coordinates
# Get main plot position to align inset with its right edge
main_pos = ax_data.get_position()
inset_width = 0.30
inset_height = 0.30
inset_left = main_pos.x1 - inset_width  # Align right edge with main plot right edge
inset_bottom = 0.60

ax_inset = fig_spring.add_axes([inset_left, inset_bottom, inset_width, inset_height])

# Get data for inset window
inset_mask = (t_obs_plot >= inset_start) & (t_obs_plot <= inset_end)
t_inset = t_obs_plot[inset_mask]
y_obs_inset = y_obs_plot[inset_mask]
filtered_inset = filtered_plot[inset_mask]
model_inset = model_plot[inset_mask]

# Plot data in inset (same colors as main plot, no alignment trace)
ax_inset.plot(t_inset, y_obs_inset, linewidth=0.6, color=colors['raw_data'], 
              alpha=0.4, zorder=3)
ax_inset.plot(t_inset, filtered_inset, linewidth=2.0, color=colors['extracted_tidal'], 
              alpha=0.95, zorder=5)
ax_inset.plot(t_inset, model_inset, linewidth=2.0, color=colors['tidal_model'], 
              linestyle='--', alpha=0.95, zorder=4)

# Add horizontal reference line at 0
ax_inset.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)

# Formatting
ax_inset.set_xlim(inset_start, inset_end)
ax_inset.grid(True, alpha=0.3)
ax_inset.tick_params(labelsize=8)

# Remove first and last y-tick labels to avoid intersection with secondary axis
xticks = ax_inset.get_xticks()
xticklabels = [ax_inset.xaxis.get_major_formatter()(tick) for tick in xticks]
# Set first and last labels to empty string
xticklabels[0] = ''
xticklabels[-1] = ''
ax_inset.set_xticks(xticks)
ax_inset.set_xticklabels(xticklabels)

# Move axes to top and right to unclutter the data
ax_inset.xaxis.tick_top()
ax_inset.yaxis.tick_right()

# Set labels on top and right
ax_inset.xaxis.set_label_position('top')
ax_inset.yaxis.set_label_position('right')
ax_inset.set_xlabel('Aug 5-9, 2025 (UTC)', fontsize=8, fontweight='bold')
ax_inset.set_ylabel('µGal', fontsize=8, fontweight='bold')

# Add border to inset
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# ========================================
# Visual connection: rectangle on main plot with lines to inset
# ========================================

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# Draw rectangle on main plot highlighting the zoomed region
# Use the actual data range in the main plot for the inset time window
y_data_min = np.min([np.min(filtered_plot[inset_mask]), np.min(model_plot[inset_mask])])
y_data_max = np.max([np.max(filtered_plot[inset_mask]), np.max(model_plot[inset_mask])])
# Add small padding (10% of range)
y_range = y_data_max - y_data_min
y_padding = y_range * 0.1
rect_y_min = y_data_min - y_padding
rect_y_max = y_data_max + y_padding

rect = Rectangle((inset_start, rect_y_min), 
                 inset_end - inset_start, 
                 rect_y_max - rect_y_min,
                 linewidth=2.5, edgecolor='black', facecolor='none',
                 linestyle='-', alpha=0.8, zorder=20)
ax_data.add_patch(rect)

# IMPORTANT: Do coordinate transforms AFTER tight_layout
# For now, just store the rectangle parameters
connection_data = {
    'inset_end': inset_end,
    'rect_y_min': rect_y_min,
    'rect_y_max': rect_y_max
}

# Calculate overall statistics
valid_mask_full = ~full_mask
if np.sum(valid_mask_full) > 0:
    filtered_valid = filtered_obs_scaled[valid_mask_full]
    model_valid = y_model_ugal[valid_mask_full]
    
    corr_filt_model = np.corrcoef(filtered_valid, model_valid)[0, 1]
    rms_diff = np.sqrt(np.mean((filtered_valid - model_valid)**2))
    
    print(f"\n{'='*80}")
    print(f"FULL DATASET STATISTICS (VALID DATA ONLY)")
    print(f"{'='*80}")
    print(f"  Filtered vs Model correlation: {corr_filt_model:.4f}")
    print(f"  RMS difference: {rms_diff:.2f} µGal")



# Add connection lines after tight_layout (when coordinates are finalized)
# Convert inset_end to matplotlib date number
inset_end_num = mdates.date2num(connection_data['inset_end'])

# Get the TOP-RIGHT corner of the rectangle in display coordinates
rect_top_right_display = ax_data.transData.transform([[inset_end_num, connection_data['rect_y_max']]])[0]

# Convert to figure coordinates
rect_top_right_fig = fig_spring.transFigure.inverted().transform([rect_top_right_display])[0]

# Get inset corners in figure coordinates
inset_bbox = ax_inset.get_position()
inset_top_left = [inset_bbox.x0, inset_bbox.y1]
inset_bottom_left = [inset_bbox.x0, inset_bbox.y0]

# Draw connection lines from rectangle TOP-RIGHT corner to BOTH inset corners:
# Line 1: Rectangle TOP-RIGHT → Inset TOP-LEFT
line1 = Line2D([rect_top_right_fig[0], inset_top_left[0]], 
               [rect_top_right_fig[1], inset_top_left[1]],
               transform=fig_spring.transFigure, color='black', 
               linestyle='--', linewidth=2, alpha=0.7, zorder=25)
fig_spring.add_artist(line1)

#Line 2
line2 = Line2D([rect_top_right_fig[0], inset_bottom_left[0]], 
               [rect_top_right_fig[1], inset_bottom_left[1]],
               transform=fig_spring.transFigure, color='black',
               linestyle='--', linewidth=2, alpha=0.7, zorder=25)
fig_spring.add_artist(line2)

# Save figure
fig_spring.savefig('tidal_signal_full_with_alignment.pdf', dpi=nc.NATURE_DPI, bbox_inches='tight', transparent=False)
print(f"\n✓ Full alignment figure saved as 'tidal_signal_full_with_alignment.pdf'")

plt.show()

print(f"\n{'='*80}")
print(f"FULL SPRING/NEAP VISUALISATION COMPLETE")
print(f"{'='*80}")
 

#=========================================================================================================
#=========================================================================================================    
# ================================
# STATIC MIRROR BASELINE (LONG-TERM - NO PROCESSING)
# ================================

print(f"\n{'='*80}")
print(f"LOADING LONG-TERM STATIC MIRROR BASELINE DATA")
print(f"{'='*80}")
#Load static mirror data (long-term, binned)
t_static_long, y_static_long = load_with_time(STATIC_MIRROR_FILE, start_time, dt_input)

print(f"  Data points: {len(y_static_long)}")
print(f"  Duration: {len(y_static_long) * dt_input / 3600:.1f} hours")
print(f"  Sampling interval: {dt_input}s ({dt_input/60:.1f} min)")

# Create tau values for long-term static baseline (240s to 250,920s)
tau_min_static_long = 2700  # seconds
tau_max_static_long = 250920  # seconds
n_points_static_long = 50  # Adjust as needed

tau_static_long_seconds = np.logspace(np.log10(tau_min_static_long), 
                                      np.log10(tau_max_static_long), 
                                      n_points_static_long)

print(f"  Tau range: {tau_min_static_long}s to {tau_max_static_long}s")
print(f"  Tau range: {tau_min_static_long/60:.1f}min to {tau_max_static_long/3600:.1f}h")

y_static_long_scaled = y_static_long * FREQ_SCALE_SQ

# Calculate Allan variance (NO spike masking - raw static data)
tau_static_long, avar_static_long = calculate_allan_variance_custom(
    y_static_long_scaled, dt_input, tau_static_long_seconds
)


min_idx_static_long = find_allan_minimum_in_window(tau_static_long, avar_static_long, ALLAN_LONG_MIN_WINDOW_HOURS)
min_static_long = np.sqrt(avar_static_long[min_idx_static_long])

print(f"\nLong-term static mirror baseline:")
print(f"  Computed {len(tau_static_long)} tau points")
print(f"  Minimum: τ={tau_static_long[min_idx_static_long]:.1f}s ({tau_static_long[min_idx_static_long]/3600:.3f}h)")
print(f"  Allan dev at min: {min_static_long:.3f} µGal")

# ================================
# STATIC MIRROR BASELINE (SHORT-TERM - HIGH-RATE)
# ================================

print(f"\n{'='*80}")
print(f"LOADING SHORT-TERM STATIC MIRROR BASELINE DATA")
print(f"{'='*80}")

# Path to short-term static mirror .mat file
SHORT_STATIC_MIRROR_DIR = "D:/USB Drive Backup/System and Elec noise trace files/Background stability processing/Chunk 1"

# Look for the .mat file in the directory
short_static_files = glob.glob(os.path.join(SHORT_STATIC_MIRROR_DIR, "Systematic_noise_chunk_20260108-211743_to_20260108-233103_full_position_trace_corrected.mat"))

if len(short_static_files) > 0:
    SHORT_STATIC_MIRROR_FILE = short_static_files[0]  # Take first .mat file found
    print(f"  Found: {os.path.basename(SHORT_STATIC_MIRROR_FILE)}")
    
    # Load short-term static mirror data (same process as short observed data)
    t_static_short, position_static_short, dt_static_short = load_short_data(
        SHORT_STATIC_MIRROR_FILE, 
        max_duration_hours=None,  # Load all available data
        start_time=None  # Don't need timestamps for Allan variance
    )
    
    # If dt wasn't in file, use the expected value
    if dt_static_short is None:
        dt_static_short = SHORT_DATA_DT
        print(f"  ⚠ Sampling rate not in file, assuming {1.0/dt_static_short:.1f}Hz")
    
    # Convert position to acceleration (same process as observed data)
    accel_static_m_s2, accel_static_ugal = position_to_acceleration_harmonic_oscillator(
        position_static_short, 
        OSCILLATOR_FREQUENCY_HZ, 
        OSCILLATOR_MASS_GRAMS
    )
    
    print(f"\n  Short-term static baseline loaded:")
    print(f"  Sampling interval: {dt_static_short}s ({1.0/dt_static_short:.1f}Hz)")
    print(f"  Data points: {len(accel_static_ugal)}")
    print(f"  Duration: {len(accel_static_ugal) * dt_static_short / 3600:.2f} hours")
    print(f"  RMS: {np.std(accel_static_ugal):.2f} µGal")
    
    # Create tau values for short-term static baseline (same range as short observed)
    tau_min_static_short = SHORT_TAU_MIN_SECONDS
    if SHORT_TAU_MAX_HOURS < 1:
     tau_max_static_short = SHORT_TAU_MAX_HOURS * 3600
    else: 
     tau_max_static_short = 2000
    tau_static_short_seconds = np.logspace(np.log10(tau_min_static_short), 
                                           np.log10(tau_max_static_short), 
                                           N_POINTS_SHORT)
    
    print(f"  Tau range: {tau_min_static_short:.1f}s to {tau_max_static_short:.1f}s")
    print(f"  Tau range: {tau_min_static_short/60:.2f}min to {tau_max_static_short/3600:.2f}h")
    
    # Calculate Allan variance (NO spike masking - raw static baseline)
    tau_static_short, avar_static_short = calculate_allan_variance_custom(
        accel_static_ugal, dt_static_short, tau_static_short_seconds
    )
    
    min_idx_static_short = np.argmin(avar_static_short)
    min_static_short = np.sqrt(avar_static_short[min_idx_static_short])
    
    print(f"\nShort-term static mirror baseline:")
    print(f"  Computed {len(tau_static_short)} tau points")
    print(f"  Minimum: τ={tau_static_short[min_idx_static_short]:.1f}s ({tau_static_short[min_idx_static_short]/60:.2f}min)")
    print(f"  Allan dev at min: {min_static_short:.3f} µGal")
    
    SHORT_STATIC_AVAILABLE = True
    
else:
    print(f"  ⚠ No .mat file found in {SHORT_STATIC_MIRROR_DIR}")
    print(f"  ⚠ Will use long-term static baseline only")
    
    SHORT_STATIC_AVAILABLE = False
    tau_static_short = np.array([])
    avar_static_short = np.array([])

# ================================
# STORE BASELINES SEPARATELY (DO NOT STITCH!)
# ================================

print(f"\n{'='*80}")
print(f"STATIC BASELINES SUMMARY (SEPARATE - NO STITCHING)")
print(f"{'='*80}")
print(f"⚠️  Note: Short and long baselines are kept separate")
print(f"   They represent different processing levels and should not be stitched")

# Save for plotting - keep baselines SEPARATE
static_baseline = {
    'short_available': SHORT_STATIC_AVAILABLE,
    'tau_long': tau_static_long,
    'avar_long': avar_static_long,
    'min_idx_long': min_idx_static_long,
    'min_value_long': min_static_long
}

if SHORT_STATIC_AVAILABLE:
    static_baseline['tau_short'] = tau_static_short
    static_baseline['avar_short'] = avar_static_short
    static_baseline['min_idx_short'] = min_idx_static_short
    static_baseline['min_value_short'] = min_static_short
    
    print(f"\n  Short-term baseline:")
    print(f"    Tau range: {tau_static_short[0]:.1f}s to {tau_static_short[-1]:.1f}s")
    print(f"    Minimum: {min_static_short:.3f} µGal at τ={tau_static_short[min_idx_static_short]:.1f}s")
    
    print(f"\n  Long-term baseline:")
    print(f"    Tau range: {tau_static_long[0]:.1f}s to {tau_static_long[-1]:.1f}s")
    print(f"    Minimum: {min_static_long:.3f} µGal at τ={tau_static_long[min_idx_static_long]:.1f}s")
else:
    print(f"\n  Long-term baseline only:")
    print(f"    Tau range: {tau_static_long[0]:.1f}s to {tau_static_long[-1]:.1f}s")
    print(f"    Minimum: {min_static_long:.3f} µGal at τ={tau_static_long[min_idx_static_long]:.1f}s")

print(f"\n✓ Static mirror baseline(s) ready for plotting (SEPARATE)")
print(f"{'='*80}")

#===========================================================================================
#STABILITY CALCULATION STAGE
#===========================================================================================
# Choose your input signal for stability analysis
signal_for_stability = y_obs  # Raw data for full instrument characterisation
signal_name = "Raw Data"

print(f"\n{'='*80}")
print(f"STABILITY ANALYSIS: {signal_name.upper()}")
print(f"{'='*80}")

# 1. Autocorrelation
print(f"\n1. Calculating autocorrelation...")
lags_auto, autocorr, lags_samples_auto = calculate_autocorrelation(
    signal_for_stability, dt_input, max_lag_hours=100
)

print(f"  Max lag: {np.max(np.abs(lags_auto)):.1f} hours")
print(f"  Zero-lag correlation: {autocorr[len(autocorr)//2]:.4f}")

# Fit decay envelope (positive lags only)
positive_mask = lags_auto > 0
lags_pos = lags_auto[positive_mask]
autocorr_pos = autocorr[positive_mask]

def exp_decay(t, tau, offset):
    return offset + (1 - offset) * np.exp(-t / tau)

try:
    fit_mask = autocorr_pos > 0.1
    if np.sum(fit_mask) > 10:
        popt, _ = opt.curve_fit(exp_decay, lags_pos[fit_mask], autocorr_pos[fit_mask], 
                                p0=[50, 0], bounds=([1, -0.5], [500, 0.5]))
        tau_fit, offset_fit = popt
        decay_envelope = exp_decay(lags_pos, tau_fit, offset_fit)
        print(f"  Decay time constant τ: {tau_fit:.1f} hours")
        print(f"  50% coherence time: {tau_fit * np.log(2):.1f} hours")
    else:
        tau_fit = None
        decay_envelope = None
        print(f"  ⚠ Could not fit decay envelope")
except Exception as e:
    tau_fit = None
    decay_envelope = None
    print(f"  ⚠ Decay fitting failed: {e}")

# 2. Cross-correlation (filtered signal vs model)
print(f"\n2. Calculating cross-correlation...")
lags_cross, xcorr, peak_lag_samples, peak_lag_hours, peak_corr = calculate_cross_correlation(
    filtered_obs_scaled, y_model_ugal, dt_input, max_lag_hours=20
)

print(f"  Peak correlation: {peak_corr:.4f}")
print(f"  Time lag: {peak_lag_hours:.3f} hours ({peak_lag_hours*60:.1f} minutes)")

if abs(peak_lag_hours) < 0.5:
    print(f"  ✓ Excellent phase alignment (lag < 30 minutes)")
elif abs(peak_lag_hours) < 2.0:
    print(f"  ~ Good phase alignment (lag < 2 hours)")
else:
    print(f"  ⚠ Significant phase offset detected")

# 3. Allan Deviation
print(f"\n3. Calculating Allan deviation...")
# ========================================
# CALCULATE ALLAN VARIANCE - LONG TIMESCALE
# ========================================

print(f"\n{'='*80}")
print(f"INITIAL ALLAN VARIANCE SETUP")
print(f"{'='*80}")

print(f"\n1. LONG TIMESCALE (BINNED DATA)")
print(f"{'='*80}")
print(f"  Signal: {LONG_DATA_NAME}")
print(f"  Sampling interval: {LONG_DATA_DT}s ({LONG_DATA_DT/60:.1f} min)")
print(f"  Data points: {len(LONG_DATA_SIGNAL)}")
print(f"  Duration: {len(LONG_DATA_SIGNAL) * LONG_DATA_DT / 3600:.1f} hours")

# Create logarithmically spaced tau values for long timescale
tau_min_long_sec = LONG_TAU_MIN_HOURS * 3600
tau_max_long_sec = LONG_TAU_MAX_HOURS * 3600
tau_long_seconds = np.logspace(np.log10(tau_min_long_sec), 
                                np.log10(tau_max_long_sec), 
                                N_POINTS_LONG)

print(f"  Tau range: {tau_min_long_sec:.0f}s to {tau_max_long_sec:.0f}s")
print(f"  Tau range: {tau_min_long_sec/3600:.2f}h to {tau_max_long_sec/3600:.1f}h")
print(f"  Number of tau points: {N_POINTS_LONG}")

# ========================================
# CALCULATE ALLAN VARIANCE - SHORT TIMESCALE
# ========================================

print(f"\n2. SHORT TIMESCALE (HIGH-RESOLUTION DATA)")
print(f"{'='*80}")

# Check if short data file exists
short_data_path = Path(SHORT_DATA_FILE)
if short_data_path.exists():
    print(f"  Loading: {SHORT_DATA_FILE}")
    
    # Load short timescale data
    t_short, position_short, dt_raw = load_short_data(SHORT_DATA_FILE, max_duration_hours=SHORT_DATA_DURATION_HOURS, start_time=SHORT_DATA_START_TIME)
    
    # Convert position to acceleration
    accel_m_s2, accel_ugal = position_to_acceleration_harmonic_oscillator(
        position_short, 
        OSCILLATOR_FREQUENCY_HZ, 
        OSCILLATOR_MASS_GRAMS
    )
    
    # Note: We keep accel_ugal at FULL resolution (dt_raw) for processing
    # Downsampling will be done per-processing-stage if needed
    
    print(f"\n  Short timescale data loaded:")
    print(f"  Raw sampling interval: {dt_raw}s ({1.0/dt_raw:.1f}Hz)")
    print(f"  Data points: {len(accel_ugal)}")
    print(f"  Duration: {len(accel_ugal) * dt_raw / 3600:.2f} hours")
    
    # Create tau values for short timescale
    tau_min_short_sec = SHORT_TAU_MIN_SECONDS
    tau_max_short_sec = SHORT_TAU_MAX_HOURS * 3600
    tau_short_seconds = np.logspace(np.log10(tau_min_short_sec), 
                                     np.log10(tau_max_short_sec), 
                                     N_POINTS_SHORT)
    
    print(f"  Tau range for Allan calc: {tau_min_short_sec:.0f}s to {tau_max_short_sec:.0f}s")
    print(f"  Tau range: {tau_min_short_sec/60:.1f}min to {tau_max_short_sec/3600:.2f}h")
    print(f"  Number of tau points: {N_POINTS_SHORT}")
    
    SHORT_DATA_AVAILABLE = True
    
else:
    print(f"  ⚠ Short data file not found: {SHORT_DATA_FILE}")
    print(f"  ⚠ Will compute Allan deviations for long timescale only")
    print(f"  ⚠ Update SHORT_DATA_FILE path when high-resolution data is ready")
    
    SHORT_DATA_AVAILABLE = False
    tau_short_seconds = np.array([])
    accel_ugal = None
    dt_raw = None

print(f"\n{'='*80}")
print(f"SETUP COMPLETE - READY FOR WEE-G STYLE ANALYSIS")
print(f"{'='*80}")
# ================================
# ALLAN DEVIATION ANALYSIS (WEE-G STYLE) - WITH STITCHING
# ================================
# Creates 4 progressive Allan deviations with proper short+long stitching:
# 1. Raw data (stitched)
# 2. Tides removed (stitched)
# 3. Drift removed (stitched)
# 4. Drift removed + 4h low-pass (stitched)

print(f"\n{'='*80}")
print(f"ALLAN DEVIATION ANALYSIS - WEE-G METHODOLOGY (WITH STITCHING)")
print(f"{'='*80}")

# Define parameters
LOWPASS_CUTOFF_HOURS = 4.0  # Wee-g's "240 min averaging"
SPIKE_THRESHOLD_UGAL = 1200 * (OSCILLATOR_FREQUENCY_HZ / 0.809) ** 2  # Threshold for spike detection

# ========================================
# SPIKE DETECTION AND MASKING
# ========================================

print(f"\n{'='*80}")
print(f"SPIKE DETECTION FOR ALLAN VARIANCE")
print(f"{'='*80}")
print(f"  Threshold: ±{SPIKE_THRESHOLD_UGAL} µGal")

# Long timescale residual (already calculated earlier)
if ENABLE_HIGHPASS:
    y_obs2= y_obs_highpassed - filtered_obs_scaled
else:
    y_obs2 = y_obs_detrended - filtered_obs_scaled
# Long timescale spike detection
spike_mask_raw_long = detect_spikes_absolute(y_obs, SPIKE_THRESHOLD_UGAL)
spike_mask_residual_long = detect_spikes_absolute(y_obs2, SPIKE_THRESHOLD_UGAL)

print(f"\nLong timescale (10-min bins):")
print(f"  Raw spikes: {np.sum(spike_mask_raw_long)} ({100*np.sum(spike_mask_raw_long)/len(spike_mask_raw_long):.2f}%)")
print(f"  Residual spikes: {np.sum(spike_mask_residual_long)} ({100*np.sum(spike_mask_residual_long)/len(spike_mask_residual_long):.2f}%)")

# Apply masks to long data
y_obs = y_obs * FREQ_SCALE_SQ
y_obs_masked = apply_spike_mask(y_obs, spike_mask_raw_long)
y_obs2_masked = apply_spike_mask(y_obs2, spike_mask_residual_long)

if SHORT_DATA_AVAILABLE:
    # Short timescale spike detection
    spike_mask_raw_short = detect_spikes_absolute(accel_ugal, SPIKE_THRESHOLD_UGAL)
    
    print(f"\nShort timescale (high-res):")
    print(f"  Raw spikes: {np.sum(spike_mask_raw_short)} ({100*np.sum(spike_mask_raw_short)/len(spike_mask_raw_short):.2f}%)")
    
    # Apply mask
    accel_ugal_masked = apply_spike_mask(accel_ugal, spike_mask_raw_short)

print(f"\n✓ Spike masking complete - spikes replaced with NaN")

# ========================================
# 1. RAW ALLAN DEVIATION (STITCHED, SPIKE-MASKED)
# ========================================

print(f"\n{'='*80}")
print(f"1. RAW DATA ALLAN DEVIATION (STITCHED, SPIKE-MASKED)")
print(f"{'='*80}")

# Long timescale
tau_raw_long, avar_raw_long = calculate_allan_variance_custom(
    y_obs_masked, LONG_DATA_DT, tau_long_seconds
)

if SHORT_DATA_AVAILABLE:
    # Short timescale
    tau_raw_short, avar_raw_short = calculate_allan_variance_custom(
        accel_ugal_masked, dt_raw, tau_short_seconds
    )
    # After calculating tau_raw_short/avar_raw_short and tau_raw_long/avar_raw_long:
    if ENABLE_JOIN_CORRECTION:
     tau_raw_short, avar_raw_short, tau_raw_long, avar_raw_long = apply_join_correction(
         tau_raw_short, avar_raw_short, tau_raw_long, avar_raw_long,
         JOIN_SHORT_ANCHOR_SECONDS, JOIN_LONG_ANCHOR_SECONDS
     )

    # Stitch
    tau_raw_stitched = np.concatenate([tau_raw_short, tau_raw_long])
    avar_raw_stitched = np.concatenate([avar_raw_short, avar_raw_long])
    sort_idx = np.argsort(tau_raw_stitched)
    tau_raw = tau_raw_stitched[sort_idx]
    avar_raw = avar_raw_stitched[sort_idx]
    
    print(f"  Short: {len(tau_raw_short)} points")
    print(f"  Long: {len(tau_raw_long)} points")
    print(f"  Stitched: {len(tau_raw)} points")
else:
    tau_raw = tau_raw_long
    avar_raw = avar_raw_long
    print(f"  Long only: {len(tau_raw)} points")

min_idx_raw = np.argmin(avar_raw)
print(f"  Minimum: τ={tau_raw[min_idx_raw]:.1f}s ({tau_raw[min_idx_raw]/3600:.3f}h), σ={np.sqrt(avar_raw[min_idx_raw]):.3f} µGal")

# ========================================
# 2. RESIDUAL (TIDES REMOVED) - STITCHED, SPIKE-MASKED
# ========================================

print(f"\n{'='*80}")
print(f"2. RESIDUAL ALLAN DEVIATION (Tides Removed, Stitched, Spike-Masked)")
print(f"{'='*80}")

y_obs2_masked_scaled = y_obs2_masked * FREQ_SCALE_SQ

# Long timescale residual (already masked)
tau_residual_long, avar_residual_long = calculate_allan_variance_custom(
    y_obs2_masked_scaled, LONG_DATA_DT, tau_long_seconds
)

if SHORT_DATA_AVAILABLE:
    # Need to extract tides from short data too
    print(f"  Extracting tidal signal from short timescale data...")
    
    # Create filter at short timescale resolution
    n_short = len(accel_ugal)
    freqs_short = fftfreq(n_short, dt_raw)
    
    filter_mask_short, _ = create_narrowband_constituent_filter(
        model_peaks, n_short, dt_raw, 
        bandwidth_hours=SPECTRAL_PEAK_BANDWIDTH_HOURS
    )
    
    # Add custom 8.2h band
    custom_period = 8.1772
    custom_bandwidth = 0.03
    period_low = custom_period - custom_bandwidth
    period_high = custom_period + custom_bandwidth
    freq_high = 1.0 / (period_low * 3600)
    freq_low = 1.0 / (period_high * 3600)
    pos_mask = (freqs_short >= freq_low) & (freqs_short <= freq_high)
    neg_mask = (freqs_short >= -freq_high) & (freqs_short <= -freq_low)
    filter_mask_short[pos_mask | neg_mask] = 1.0
    
    # Apply filter (to masked data)
    filtered_short, _, _ = apply_filter(accel_ugal_masked, filter_mask_short)
    
    # Residual: raw - tides
    residual_short = accel_ugal_masked - filtered_short
    
    # Detect spikes in residual
    spike_mask_residual_short = detect_spikes_absolute(
        residual_short[~np.isnan(residual_short)], SPIKE_THRESHOLD_UGAL
    )
    
    print(f"  Short residual spikes: {np.sum(spike_mask_residual_short)} points")
    print(f"  Short residual std: {np.nanstd(residual_short):.2f} µGal")
    
    # Calculate Allan variance (NaN-aware)
    tau_residual_short, avar_residual_short = calculate_allan_variance_custom(
        residual_short, dt_raw, tau_short_seconds
    )
    if ENABLE_JOIN_CORRECTION:
     tau_residual_short, avar_residual_short, tau_residual_long, avar_residual_long = apply_join_correction(
         tau_residual_short, avar_residual_short, tau_residual_long, avar_residual_long,
         JOIN_SHORT_ANCHOR_SECONDS, JOIN_LONG_ANCHOR_SECONDS
         )
    # Stitch
    tau_residual_stitched = np.concatenate([tau_residual_short, tau_residual_long])
    avar_residual_stitched = np.concatenate([avar_residual_short, avar_residual_long])
    sort_idx = np.argsort(tau_residual_stitched)
    tau_residual = tau_residual_stitched[sort_idx]
    avar_residual = avar_residual_stitched[sort_idx]
    
    print(f"  Stitched: {len(tau_residual)} points")
else:
    tau_residual = tau_residual_long
    avar_residual = avar_residual_long

min_idx_residual = np.argmin(avar_residual)
print(f"  Minimum: τ={tau_residual[min_idx_residual]:.1f}s ({tau_residual[min_idx_residual]/3600:.3f}h), σ={np.sqrt(avar_residual[min_idx_residual]):.3f} µGal")

# ========================================
# 3. DRIFT REMOVED - TWO-STAGE, STITCHED, SPIKE-MASKED
# ========================================

print(f"\n{'='*80}")
print(f"3. DRIFT-CORRECTED ALLAN DEVIATION (TWO-STAGE, STITCHED, SPIKE-MASKED)")
print(f"{'='*80}")

# STAGE 1: Linear drift removal
print(f"\nSTAGE 1: Linear Drift Removal")
print(f"{'-'*80}")

# Long timescale: remove linear drift from masked residual
# Need to work with valid (non-NaN) data for fitting
valid_mask_long = ~np.isnan(y_obs2_masked)
t_valid_long = np.arange(len(y_obs2_masked))[valid_mask_long] * LONG_DATA_DT / 86400  # days
y_valid_long = y_obs2_masked[valid_mask_long]

linear_coeffs_long = np.polyfit(t_valid_long, y_valid_long, 1)
# Reconstruct full trend (including NaN positions)
t_full_long = np.arange(len(y_obs2_masked)) * LONG_DATA_DT / 86400
linear_trend_long = np.polyval(linear_coeffs_long, t_full_long)
linear_detrended_long = y_obs2_masked - linear_trend_long  # NaN preserved

drift_rate_linear_long = linear_coeffs_long[0]  # µGal/day

# SAVE FOR DRIFT FIGURE (which comes after this section)
drift_rate_ugal_per_day = drift_rate_linear_long
linear_coeffs = linear_coeffs_long
y_linear_fit = linear_trend_long
residuals_linear = linear_detrended_long

# Calculate stats for drift figure
r_squared = 1 - (np.nanvar(residuals_linear) / np.nanvar(y_obs2_masked))
rms_residual = np.sqrt(np.nanmean(residuals_linear**2))
residual_std = np.nanstd(residuals_linear)
max_residual = np.nanmax(np.abs(residuals_linear))
drift_rate_uncertainty = 1.0  # Simplified

print(f"  Long: Linear drift = {drift_rate_linear_long:.2f} µGal/day")
print(f"  Long: Removed {np.nanmax(linear_trend_long) - np.nanmin(linear_trend_long):.1f} µGal total")
print(f"  Long: R² = {r_squared:.5f}")

if SHORT_DATA_AVAILABLE:
    # Short timescale: remove linear drift from masked residual
    valid_mask_short = ~np.isnan(residual_short)
    t_valid_short = np.arange(len(residual_short))[valid_mask_short] * dt_raw / 86400
    y_valid_short = residual_short[valid_mask_short]
    
    if len(y_valid_short) > 10:  # Need enough points
        linear_coeffs_short = np.polyfit(t_valid_short, y_valid_short, 1)
        t_full_short = np.arange(len(residual_short)) * dt_raw / 86400
        linear_trend_short = np.polyval(linear_coeffs_short, t_full_short)
        linear_detrended_short = residual_short - linear_trend_short
        
        drift_rate_short = linear_coeffs_short[0]  # µGal/day
        print(f"  Short: Linear drift = {drift_rate_short:.2f} µGal/day")
        print(f"  Short: Removed {np.nanmax(linear_trend_short) - np.nanmin(linear_trend_short):.1f} µGal total")
    else:
        print(f"  Short: ⚠️ Not enough valid points for linear fit")
        linear_detrended_short = residual_short

# STAGE 2: High-pass filter for non-linear drift
if REMOVE_DRIFT_ORDER is not None and REMOVE_DRIFT_ORDER > 1:
    print(f"\nSTAGE 2: High-Pass Filter for Non-Linear Drift")
    print(f"{'='*80}")
    print(f"  Cutoff: >{HIGHPASS_CUTOFF_HOURS}h periods removed")
    
    def highpass_filter_with_nan(signal, dt, cutoff_period_hours):
        """High-pass filter that handles NaN values"""
        # Remove NaN for FFT
        valid_mask = ~np.isnan(signal)
        signal_valid = signal[valid_mask]
        
        if len(signal_valid) < 100:
            return signal, None, None  # Not enough data
        
        n = len(signal_valid)
        signal_centered = signal_valid - np.mean(signal_valid)
        signal_fft = fft(signal_centered)
        freqs = fftfreq(n, dt)
        
        cutoff_freq = 1.0 / (cutoff_period_hours * 3600)
        highpass_mask = np.abs(freqs) >= cutoff_freq
        
        filtered_fft = signal_fft * highpass_mask
        filtered_signal_valid = np.real(ifft(filtered_fft))
        
        # Calculate removed component
        removed_component_valid = signal_centered - filtered_signal_valid
        
        # Reconstruct with NaN in original positions
        filtered_signal = np.full(len(signal), np.nan)
        filtered_signal[valid_mask] = filtered_signal_valid
        
        removed_component = np.full(len(signal), np.nan)
        removed_component[valid_mask] = removed_component_valid
        
        return filtered_signal, removed_component, highpass_mask
    
    # Long: high-pass the linearly-detrended data
    data_highpassed_long, nonlinear_drift_long, _ = highpass_filter_with_nan(
        linear_detrended_long, LONG_DATA_DT, HIGHPASS_CUTOFF_HOURS
    )
    
    if nonlinear_drift_long is not None:
        print(f"  Long: Additional non-linear drift removed = {np.nanstd(nonlinear_drift_long):.1f} µGal")
        print(f"  Long: Final std after both stages = {np.nanstd(data_highpassed_long):.1f} µGal")
        data_drift_removed_long = data_highpassed_long
    else:
        print(f"  Long: ⚠️ High-pass failed, using linear detrend only")
        data_drift_removed_long = linear_detrended_long
    
    if SHORT_DATA_AVAILABLE:
        # Short: high-pass the linearly-detrended data
        data_highpassed_short, nonlinear_drift_short, _ = highpass_filter_with_nan(
            linear_detrended_short, dt_raw, HIGHPASS_CUTOFF_HOURS
        )
        
        if nonlinear_drift_short is not None:
            print(f"  Short: Additional non-linear drift removed = {np.nanstd(nonlinear_drift_short):.1f} µGal")
            print(f"  Short: Final std after both stages = {np.nanstd(data_highpassed_short):.1f} µGal")
            data_drift_removed_short = data_highpassed_short
        else:
            print(f"  Short: ⚠️ High-pass failed, using linear detrend only")
            data_drift_removed_short = linear_detrended_short
    
else:
    print(f"\nSTAGE 2: Skipped (REMOVE_DRIFT_ORDER = {REMOVE_DRIFT_ORDER})")
    print(f"  Using linear detrend only")
    data_drift_removed_long = linear_detrended_long
    if SHORT_DATA_AVAILABLE:
        data_drift_removed_short = linear_detrended_short

print(f"\nFinal detrended std (excluding NaN):")
print(f"  Long: {np.nanstd(data_drift_removed_long):.2f} µGal")
if SHORT_DATA_AVAILABLE:
    print(f"  Short: {np.nanstd(data_drift_removed_short):.2f} µGal")
    
data_drift_removed_scaled = data_drift_removed_long * FREQ_SCALE_SQ
# Calculate Allan variance on drift-corrected data
tau_detrended_long, avar_detrended_long = calculate_allan_variance_custom(
    data_drift_removed_scaled, LONG_DATA_DT, tau_long_seconds
)


if SHORT_DATA_AVAILABLE:
    tau_detrended_short, avar_detrended_short = calculate_allan_variance_custom(
        data_drift_removed_short, dt_raw, tau_short_seconds
    )
    if ENABLE_JOIN_CORRECTION:
     tau_detrended_short, avar_detrended_short, tau_detrended_long, avar_detrended_long = apply_join_correction(
         tau_detrended_short, avar_detrended_short, tau_detrended_long, avar_detrended_long,
         JOIN_SHORT_ANCHOR_SECONDS, JOIN_LONG_ANCHOR_SECONDS
         )
    # Stitch
    tau_detrended_stitched = np.concatenate([tau_detrended_short, tau_detrended_long])
    avar_detrended_stitched = np.concatenate([avar_detrended_short, avar_detrended_long])
    sort_idx = np.argsort(tau_detrended_stitched)
    tau_detrended = tau_detrended_stitched[sort_idx]
    avar_detrended = avar_detrended_stitched[sort_idx]
else:
    tau_detrended = tau_detrended_long
    avar_detrended = avar_detrended_long

min_idx_detrended = np.argmin(avar_detrended)
print(f"\n  Minimum: τ={tau_detrended[min_idx_detrended]:.1f}s ({tau_detrended[min_idx_detrended]/3600:.3f}h), σ={np.sqrt(avar_detrended[min_idx_detrended]):.3f} µGal")

# ========================================
# 4. DRIFT REMOVED + 4H LOW-PASS - STITCHED, SPIKE-MASKED
# ========================================

print(f"\n{'='*80}")
print(f"4. DRIFT + WHITE NOISE REMOVED (Wee-g Style, Stitched, Spike-Masked)")
print(f"{'='*80}")

# Long timescale: apply low-pass (handles NaN internally via FFT)
def apply_lowpass_filter_with_nan(signal, dt, cutoff_period_hours):
    """Apply low-pass filter, handling NaN values"""
    # Remove NaN for FFT
    valid_mask = ~np.isnan(signal)
    signal_valid = signal[valid_mask]
    
    if len(signal_valid) < 100:
        return signal, None  # Not enough data
    
    n = len(signal_valid)
    signal_centered = signal_valid - np.mean(signal_valid)
    signal_fft = fft(signal_centered)
    freqs = fftfreq(n, dt)
    
    cutoff_freq = 1.0 / (cutoff_period_hours * 3600)
    lowpass_mask = np.abs(freqs) <= cutoff_freq
    
    filtered_fft = signal_fft * lowpass_mask
    filtered_signal_valid = np.real(ifft(filtered_fft))
    
    # Reconstruct with NaN in original positions
    filtered_signal = np.full(len(signal), np.nan)
    filtered_signal[valid_mask] = filtered_signal_valid
    
    return filtered_signal, lowpass_mask

data_detrended_lowpassed_long, _ = apply_lowpass_filter_with_nan(
    data_drift_removed_long, LONG_DATA_DT, LOWPASS_CUTOFF_HOURS
)

print(f"  Long: {LOWPASS_CUTOFF_HOURS}h low-pass applied")
print(f"  Long std after: {np.nanstd(data_detrended_lowpassed_long):.2f} µGal")

tau_final_long, avar_final_long = calculate_allan_variance_custom(
    data_detrended_lowpassed_long, LONG_DATA_DT, tau_long_seconds
)

if SHORT_DATA_AVAILABLE:
    # Apply low-pass to short data
    print(f"  Applying {LOWPASS_CUTOFF_HOURS}h low-pass to short timescale data...")
    
    data_detrended_lowpassed_short, _ = apply_lowpass_filter_with_nan(
        data_drift_removed_short, dt_raw, LOWPASS_CUTOFF_HOURS
    )
    
    print(f"  Short std after: {np.nanstd(data_detrended_lowpassed_short):.2f} µGal")
    
    # Calculate Allan variance
    tau_final_short, avar_final_short = calculate_allan_variance_custom(
        data_detrended_lowpassed_short, dt_raw, tau_short_seconds
    )
    
    # Stitch
    tau_final_stitched = np.concatenate([tau_final_short, tau_final_long])
    avar_final_stitched = np.concatenate([avar_final_short, avar_final_long])
    sort_idx = np.argsort(tau_final_stitched)
    tau_final = tau_final_stitched[sort_idx]
    avar_final = avar_final_stitched[sort_idx]
    
    print(f"  Stitched: {len(tau_final)} points")
else:
    tau_final = tau_final_long
    avar_final = avar_final_long

min_idx_final = np.argmin(avar_final)
print(f"  Minimum: τ={tau_final[min_idx_final]:.1f}s ({tau_final[min_idx_final]/3600:.3f}h), σ={np.sqrt(avar_final[min_idx_final]):.3f} µGal")

# ========================================
# SUMMARY
# ========================================

print(f"\n{'='*80}")
print(f"STITCHED ALLAN DEVIATION SUMMARY")
print(f"{'='*80}")
print(f"\n{'Processing Stage':<35}{'Min Allan Dev':<20}{'τ at min':<20}{'Improvement':<15}")
print(f"{'-'*90}")

min_raw = np.sqrt(avar_raw[min_idx_raw])
min_residual = np.sqrt(avar_residual[min_idx_residual])
min_detrended = np.sqrt(avar_detrended[min_idx_detrended])
min_final = np.sqrt(avar_final[min_idx_final])

tau_at_min_raw = tau_raw[min_idx_raw]
tau_at_min_residual = tau_residual[min_idx_residual]
tau_at_min_detrended = tau_detrended[min_idx_detrended]
tau_at_min_final = tau_final[min_idx_final]

print(f"{'1. Raw':<35}{min_raw:<20.3f}{f'{tau_at_min_raw:.1f}s':<20}{'-':<15}")
print(f"{'2. Tides removed':<35}{min_residual:<20.3f}{f'{tau_at_min_residual:.1f}s':<20}{f'{(1-min_residual/min_raw)*100:.1f}%':<15}")
print(f"{'3. Drift removed':<35}{min_detrended:<20.3f}{f'{tau_at_min_detrended:.1f}s':<20}{f'{(1-min_detrended/min_raw)*100:.1f}%':<15}")
print(f"{'4. Drift + 4h low-pass':<35}{min_final:<20.3f}{f'{tau_at_min_final:.1f}s':<20}{f'{(1-min_final/min_raw)*100:.1f}%':<15}")

if SHORT_DATA_AVAILABLE:
    print(f"\n✓ All Allan deviations include short timescale stitching")
    print(f"  Short range: {tau_raw_short[0]:.1f}s to {tau_raw_short[-1]:.1f}s")
    print(f"  Long range: {tau_raw_long[0]:.1f}s to {tau_raw_long[-1]:.1f}s")
else:
    print(f"\n⚠️  Short timescale data not available - long timescale only")

print(f"\n{'='*80}")

# ================================
# PLOT-STAGE FREQUENCY SCALING
# ================================
# Change Z_MODE_FREQ_HZ to test different z-mode candidates.
# 0.809 = baseline (no change)
# 4.2 Hz 
# 5.8 Hz candidate 
# All processing remains in raw µGal — scaling only applied at plot time.

Z_MODE_FREQ_HZ = 0.809  # <-- Change this to test candidates
PLOT_FREQ_SCALE = (Z_MODE_FREQ_HZ / 0.809) ** 2  # 1.0 at baseline

# Helper: multiply adev by sqrt(PLOT_FREQ_SCALE) since we are in adev not variance space
def scaled_adev(avar):
    return np.sqrt(avar) * np.sqrt(PLOT_FREQ_SCALE)

# Plot limits
tau_min_plot = 5       # seconds
tau_max_plot = 4e5     # seconds
avar_min = 1e1 * np.sqrt(PLOT_FREQ_SCALE)
avar_max = 1e3 * np.sqrt(PLOT_FREQ_SCALE)
ELECTRONIC_NOISE_FLOOR_PLOT = ELECTRONIC_NOISE_FLOOR_UGAL * np.sqrt(PLOT_FREQ_SCALE)

print(f"\n{'='*80}")
print(f"CREATING 3x2 COMPARISON FIGURE WITH TWO-COLOR ALLAN PLOTS")
print(f"{'='*80}")
print(f"  Z-mode frequency: {Z_MODE_FREQ_HZ} Hz")
print(f"  Plot scale factor (adev): {np.sqrt(PLOT_FREQ_SCALE):.4f}x")

fig_comparison, axes = plt.subplots(3, 2, figsize=nc.get_figure_size('double', 0.93))

t_plot = t_obs
time_label = 'Time (UTC)'

# ========================================
# ROW 1: RAW DATA
# ========================================
color_short_raw    = 'xkcd:light blue'
color_long_raw     = 'xkcd:royal blue'
color_static_short = 'xkcd:light gray'
color_static_long  = 'xkcd:dark gray'

# Left: Raw trace (unscaled — lives in raw µGal space)
# Row 1: Raw data - UPDATED COLORS
axes[0, 0].plot(t_plot, y_obs_masked*PLOT_FREQ_SCALE, 
               linewidth=0.8, alpha=0.7, 
               color=colors['raw_data'])  # Was 'xkcd:blue'
axes[0, 0].set_ylabel('Acceleration (µGal)', fontweight='bold')
axes[0, 0].set_title('(a) Raw Data', fontsize=8, fontweight='bold', loc='left')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].text(0.02, 0.98, f'Std: {np.nanstd(y_obs_masked):.1f} µGal',
                transform=axes[0, 0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Right: Raw Allan deviation
# Raw Allan deviation - UPDATED COLORS
if SHORT_DATA_AVAILABLE:
    axes[0, 1].loglog(tau_raw_short, scaled_adev(avar_raw_short), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_short_1'],  # Was 'xkcd:light blue'
                     alpha=0.8, label='Short timescale', zorder=3)
    axes[0, 1].loglog(tau_raw_long, scaled_adev(avar_raw_long), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_long_1'],  # Was 'xkcd:royal blue'
                     alpha=0.8, label='Long timescale', zorder=3)
    min_idx_raw_short = np.argmin(avar_raw_short)
    min_idx_raw_long  = find_allan_minimum_in_window(tau_raw_long, avar_raw_long, ALLAN_LONG_MIN_WINDOW_HOURS)
    axes[0, 1].scatter([tau_raw_long[min_idx_raw_long]],
                       [scaled_adev(avar_raw_long)[min_idx_raw_long]],
                       s=60, color=color_long_raw, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = (f'Min (short): {scaled_adev(avar_raw_short)[min_idx_raw_short]:.1f} µGal'
                f' @ τ={tau_raw_short[min_idx_raw_short]:.2f}s\n'
                f'Min (long):  {scaled_adev(avar_raw_long)[min_idx_raw_long]:.1f} µGal'
                f' @ τ={tau_raw_long[min_idx_raw_long]/3600:.2f}h')
else:
    axes[0, 1].loglog(tau_raw, scaled_adev(avar_raw), 'o-', linewidth=2, markersize=4,
                      color=color_long_raw, alpha=0.8)
    min_idx_raw = np.argmin(avar_raw)
    axes[0, 1].scatter([tau_raw[min_idx_raw]], [scaled_adev(avar_raw)[min_idx_raw]],
                       s=60, color=color_long_raw, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = f'Min: {scaled_adev(avar_raw)[min_idx_raw]:.1f} µGal\nτ={tau_raw[min_idx_raw]/3600:.2f}h'

axes[0, 1].set_ylabel('Allan deviation (µGal)', fontweight='bold')
axes[0, 1].set_title('(b) Allan Deviation - Raw',fontsize=8, fontweight='bold', loc='left')
axes[0, 1].grid(True, alpha=0.3, which='both')
axes[0, 1].set_xlim([tau_min_plot, tau_max_plot])
axes[0, 1].set_ylim([avar_min, avar_max])
axes[0, 1].text(0.02, 0.98, min_text,
                transform=axes[0, 1].transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Static baselines - UPDATED COLORS
if static_baseline['short_available']:
    axes[0, 1].loglog(static_baseline['tau_short'], 
                     scaled_adev(static_baseline['avar_short']),
                     '--', linewidth=2.0, alpha=0.7, 
                     color=colors['static_baseline_short'],  # Was 'xkcd:light gray'
                     label='Static mirror: Short', zorder=1)
    axes[0, 1].loglog(static_baseline['tau_long'], 
                     scaled_adev(static_baseline['avar_long']),
                     '--', linewidth=2.0, alpha=0.7, 
                     color=colors['static_baseline_long'],  # Was 'xkcd:dark gray'
                     label='Static mirror: Long', zorder=1)
    axes[0, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
    axes[0, 1].fill_between(static_baseline['tau_short'],
                            scaled_adev(static_baseline['avar_short']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
else:
    axes[0, 1].loglog(static_baseline['tau_long'], scaled_adev(static_baseline['avar_long']),
                      'k--', linewidth=2.5, alpha=0.7, label='Static mirror', zorder=1)
    axes[0, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
axes[0, 1].legend(loc='upper right')

# ========================================
# ROW 2: RESIDUAL (TIDES REMOVED)
# ========================================
color_short_residual = 'xkcd:light green'
color_long_residual  = 'xkcd:forest green'

# Left: Residual trace (unscaled)
# Row 2: Residual (Tides Removed) - UPDATED COLORS
axes[1, 0].plot(t_plot, y_obs2_masked*PLOT_FREQ_SCALE, 
               linewidth=0.8, alpha=0.7, 
               color=colors['allan_long_2'])  # Was 'xkcd:green'
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1, 0].set_ylabel('Acceleration (µGal)', fontweight='bold')
axes[1, 0].set_title('(c) Residual (Tides Removed)', fontsize=8, fontweight='bold', loc='left')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(0.02, 0.98, f'Std: {np.nanstd(y_obs2_masked):.1f} µGal',
                transform=axes[1, 0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Right: Residual Allan deviation
if SHORT_DATA_AVAILABLE:
    axes[1, 1].loglog(tau_residual_short, scaled_adev(avar_residual_short), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_short_2'],  # Was 'xkcd:light green'
                     alpha=0.8, label='Short timescale')
    axes[1, 1].loglog(tau_residual_long, scaled_adev(avar_residual_long), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_long_2'],  # Was 'xkcd:forest green'
                     alpha=0.8, label='Long timescale')
    min_idx_residual_short = np.argmin(avar_residual_short)
    min_idx_residual_long  = find_allan_minimum_in_window(tau_residual_long, avar_residual_long, ALLAN_LONG_MIN_WINDOW_HOURS)
    axes[1, 1].scatter([tau_residual_long[min_idx_residual_long]],
                       [scaled_adev(avar_residual_long)[min_idx_residual_long]],
                       s=60, color=color_long_residual, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = (f'Min (short): {scaled_adev(avar_residual_short)[min_idx_residual_short]:.1f} µGal'
                f' @ τ={tau_residual_short[min_idx_residual_short]:.2f}s\n'
                f'Min (long):  {scaled_adev(avar_residual_long)[min_idx_residual_long]:.1f} µGal'
                f' @ τ={tau_residual_long[min_idx_residual_long]/3600:.2f}h')
else:
    axes[1, 1].loglog(tau_residual, scaled_adev(avar_residual), 'o-', linewidth=2, markersize=4,
                      color=color_long_residual, alpha=0.8)
    min_idx_residual = np.argmin(avar_residual)
    axes[1, 1].scatter([tau_residual[min_idx_residual]], [scaled_adev(avar_residual)[min_idx_residual]],
                       s=120, color=color_long_residual, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = f'Min: {scaled_adev(avar_residual)[min_idx_residual]:.1f} µGal\nτ={tau_residual[min_idx_residual]/3600:.2f}h'

axes[1, 1].set_ylabel('Allan deviation (µGal)', fontweight='bold')
axes[1, 1].set_title('(d) Allan Deviation - Residual', fontsize=8, fontweight='bold', loc='left')
axes[1, 1].grid(True, alpha=0.3, which='both')
axes[1, 1].set_xlim([tau_min_plot, tau_max_plot])
axes[1, 1].set_ylim([avar_min, avar_max])
axes[1, 1].text(0.02, 0.98, min_text,
                transform=axes[1, 1].transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

if static_baseline['short_available']:
    axes[1, 1].loglog(static_baseline['tau_short'], scaled_adev(static_baseline['avar_short']),
                      '--', linewidth=2.5, alpha=0.7, color=color_static_short,
                      label='Static mirror: Short', zorder=1)
    axes[1, 1].loglog(static_baseline['tau_long'],  scaled_adev(static_baseline['avar_long']),
                      '--', linewidth=2.5, alpha=0.7, color=color_static_long,
                      label='Static mirror: Long',  zorder=1)
    axes[1, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
    axes[1, 1].fill_between(static_baseline['tau_short'],
                            scaled_adev(static_baseline['avar_short']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
else:
    axes[1, 1].loglog(static_baseline['tau_long'], scaled_adev(static_baseline['avar_long']),
                      'k--', linewidth=2.5, alpha=0.7, label='Static mirror', zorder=1)
    axes[1, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
axes[1, 1].legend(loc='upper right')
axes[1, 1].axhline(ELECTRONIC_NOISE_FLOOR_PLOT,
                   color='red', linestyle=':', linewidth=2.5, alpha=0.7,
                   label='Electronic noise', zorder=0.5)

# ========================================
# ROW 3: DRIFT REMOVED
# ========================================
color_short_detrended = 'xkcd:light orange'
color_long_detrended  = 'xkcd:dark orange'

# Left: Detrended trace (unscaled)
# Row 3: Drift Removed - UPDATED COLORS
axes[2, 0].plot(t_plot, data_drift_removed_long*PLOT_FREQ_SCALE, 
               linewidth=0.8, alpha=0.7, 
               color=colors['allan_long_3'])  # Was 'xkcd:orange'
axes[2, 0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[2, 0].set_ylabel('Acceleration (µGal)', fontweight='bold')
axes[2, 0].set_xlabel(time_label, fontweight='bold')
axes[2, 0].set_title('(e) Drift Removed', fontsize=8, fontweight='bold', loc='left')
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_ylim(-600*PLOT_FREQ_SCALE, 600*PLOT_FREQ_SCALE)
axes[2, 0].text(0.02, 0.98,
                f'Std: {np.nanstd(data_drift_removed_long):.1f} µGal\nDrift: {drift_rate_ugal_per_day:.1f} µGal/day',
                transform=axes[2, 0].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right: Detrended Allan deviation
# Detrended Allan - UPDATED COLORS
if SHORT_DATA_AVAILABLE:
    axes[2, 1].loglog(tau_detrended_short, scaled_adev(avar_detrended_short), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_short_3'],  # Was 'xkcd:light orange'
                     alpha=0.8, label='Short timescale')
    axes[2, 1].loglog(tau_detrended_long, scaled_adev(avar_detrended_long), 'o-', 
                     linewidth=1.5, markersize=3,
                     color=colors['allan_long_3'],  # Was 'xkcd:dark orange'
                     alpha=0.8, label='Long timescale')
    min_idx_detrended_short = np.argmin(avar_detrended_short)
    min_idx_detrended_long  = find_allan_minimum_in_window(tau_detrended_long, avar_detrended_long, ALLAN_LONG_MIN_WINDOW_HOURS)
    axes[2, 1].scatter([tau_detrended_long[min_idx_detrended_long]],
                       [scaled_adev(avar_detrended_long)[min_idx_detrended_long]],
                       s=60, color=color_long_detrended, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = (f'Min (short): {scaled_adev(avar_detrended_short)[min_idx_detrended_short]:.1f} µGal'
                f' @ τ={tau_detrended_short[min_idx_detrended_short]:.2f}s\n'
                f'Min (long):  {scaled_adev(avar_detrended_long)[min_idx_detrended_long]:.1f} µGal'
                f' @ τ={tau_detrended_long[min_idx_detrended_long]/3600:.2f}h')
else:
    axes[2, 1].loglog(tau_detrended, scaled_adev(avar_detrended), 'o-', linewidth=2, markersize=4,
                      color=color_long_detrended, alpha=0.8)
    min_idx_detrended = np.argmin(avar_detrended)
    axes[2, 1].scatter([tau_detrended[min_idx_detrended]], [scaled_adev(avar_detrended)[min_idx_detrended]],
                       s=60, color=color_long_detrended, marker='o', zorder=5, edgecolors='black', linewidth=2)
    min_text = f'Min: {scaled_adev(avar_detrended)[min_idx_detrended]:.1f} µGal\nτ={tau_detrended[min_idx_detrended]/3600:.2f}h'

axes[2, 1].set_ylabel('Allan deviation (µGal)', fontweight='bold')
axes[2, 1].set_xlabel('Integration time, τ (s)', fontweight='bold')
axes[2, 1].set_title('(f) Allan Deviation - Detrended', fontsize=8, fontweight='bold', loc='left')
axes[2, 1].grid(True, alpha=0.3, which='both')
axes[2, 1].set_xlim([tau_min_plot, tau_max_plot])
axes[2, 1].set_ylim([avar_min, avar_max])
axes[2, 1].text(0.02, 0.98, min_text,
                transform=axes[2, 1].transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

if static_baseline['short_available']:
    axes[2, 1].loglog(static_baseline['tau_short'], scaled_adev(static_baseline['avar_short']),
                      '--', linewidth=2.5, alpha=0.7, color=color_static_short,
                      label='Static mirror: Short', zorder=1)
    axes[2, 1].loglog(static_baseline['tau_long'],  scaled_adev(static_baseline['avar_long']),
                      '--', linewidth=2.5, alpha=0.7, color=color_static_long,
                      label='Static mirror: Long',  zorder=1)
    axes[2, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
    axes[2, 1].fill_between(static_baseline['tau_short'],
                            scaled_adev(static_baseline['avar_short']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
else:
    axes[2, 1].loglog(static_baseline['tau_long'], scaled_adev(static_baseline['avar_long']),
                      'k--', linewidth=2.5, alpha=0.7, label='Static mirror', zorder=1)
    axes[2, 1].fill_between(static_baseline['tau_long'],
                            scaled_adev(static_baseline['avar_long']), avar_min,
                            alpha=0.1, color='gray', zorder=0)
axes[2, 1].legend(loc='upper right')
axes[2, 1].axhline(ELECTRONIC_NOISE_FLOOR_PLOT,
                   color='red', linestyle=':', linewidth=2.5, alpha=0.7,
                   label='Electronic noise', zorder=0.5)
# Update all axis labels to proper size
for ax in axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
    ax.legend( framealpha=0.95)

# ADD panel labels (Nature requirement)
nc.add_panel_labels(axes, start_letter='a', fontsize=8)
# ========================================
# FINALISE FIGURE
# ========================================
fig_comparison.suptitle(
    f'Progressive Processing: Traces and Allan Deviation Analysis'
    f'  [Z-mode: 5.85 Hz]',
    fontsize=8, fontweight='bold', y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.99])
fig_comparison.savefig('allan_progression_comparison.pdf', dpi=nc.NATURE_DPI, bbox_inches='tight', transparent=False)
print(f"✓ Comparison figure saved: 'allan_progression_comparison.pdf'")
plt.show()

print(f"\n{'='*80}")
print(f"TWO-COLOR ALLAN DEVIATION ANALYSIS COMPLETE")
print(f"  Z-mode used: {Z_MODE_FREQ_HZ} Hz")
print(f"  Plot scale:  {PLOT_FREQ_SCALE:.4f}x variance  /  {np.sqrt(PLOT_FREQ_SCALE):.4f}x adev")
print(f"{'='*80}")
#N_POINTS_LONG
#PARAMETERS
#SHORT_TAU_MIN_FACTOR
#SHORT_DATA_FILE
#SHORT_DATA_DURATION_HOURS
#OSCILLATOR_FREQUENCY_HZ
#Progressive Processing: Traces and Allan Deviation Analysis
print(f"\n{'='*80}")
print(f"STABILITY ANALYSIS COMPLETE")
print(f"{'='*80}")