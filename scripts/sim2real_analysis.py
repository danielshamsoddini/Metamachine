#!/usr/bin/env python3
"""
Sim-to-Real Actuation Analysis Tool

This script compares joint tracking data from simulation and real robot
to diagnose actuation differences and reality gap issues.

Analysis includes:
1. Tracking error comparison (RMSE, max error, delay)
2. Frequency response analysis
3. Step response characteristics
4. Joint-by-joint comparison plots

Usage:
    python scripts/sim2real_analysis.py sim_robot_tracking.npz real_robot_tracking.npz
    python scripts/sim2real_analysis.py sim.npz real.npz --output analysis_results/
    python scripts/sim2real_analysis.py sim.npz real.npz --joint 0 1 2  # specific joints

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr


def load_tracking_data(filepath: str) -> dict:
    """Load tracking data from .npz or .pkl file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = dict(np.load(filepath, allow_pickle=True))
        # Handle numpy object arrays for lists
        for key in ['joint_names', 'default_dof_pos']:
            if key in data and isinstance(data[key], np.ndarray):
                if data[key].ndim == 0:
                    data[key] = data[key].item()
    elif filepath.suffix == '.pkl':
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return data


def compute_tracking_metrics(commands: np.ndarray, positions: np.ndarray, 
                             timestamps: np.ndarray) -> dict:
    """Compute tracking performance metrics for each joint.
    
    Args:
        commands: Joint commands array (T, N)
        positions: Actual positions array (T, N)
        timestamps: Time array (T,)
        
    Returns:
        Dictionary of metrics per joint
    """
    num_joints = commands.shape[1]
    tracking_error = positions - commands
    
    metrics = {
        'rmse': [],
        'max_error': [],
        'mean_error': [],
        'std_error': [],
        'delay_samples': [],
        'delay_time': [],
        'correlation': [],
    }
    
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
    
    for j in range(num_joints):
        error = tracking_error[:, j]
        cmd = commands[:, j]
        pos = positions[:, j]
        
        # Basic error metrics
        metrics['rmse'].append(np.sqrt(np.mean(error ** 2)))
        metrics['max_error'].append(np.max(np.abs(error)))
        metrics['mean_error'].append(np.mean(error))
        metrics['std_error'].append(np.std(error))
        
        # Estimate delay using cross-correlation
        if np.std(cmd) > 1e-6 and np.std(pos) > 1e-6:
            correlation = np.correlate(pos - np.mean(pos), cmd - np.mean(cmd), mode='full')
            delay_idx = np.argmax(correlation) - (len(cmd) - 1)
            metrics['delay_samples'].append(delay_idx)
            metrics['delay_time'].append(delay_idx * dt)
            
            # Pearson correlation
            corr, _ = pearsonr(cmd, pos)
            metrics['correlation'].append(corr)
        else:
            metrics['delay_samples'].append(0)
            metrics['delay_time'].append(0.0)
            metrics['correlation'].append(0.0)
    
    return metrics


def compute_frequency_response(commands: np.ndarray, positions: np.ndarray,
                               timestamps: np.ndarray) -> dict:
    """Compute frequency response characteristics.
    
    Args:
        commands: Joint commands array (T, N)
        positions: Actual positions array (T, N)
        timestamps: Time array (T,)
        
    Returns:
        Dictionary with frequency analysis results
    """
    num_joints = commands.shape[1]
    dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
    fs = 1.0 / dt
    
    results = {
        'frequencies': [],
        'cmd_psd': [],
        'pos_psd': [],
        'coherence': [],
        'gain': [],
        'phase': [],
    }
    
    for j in range(num_joints):
        cmd = commands[:, j]
        pos = positions[:, j]
        
        # Power spectral density
        f_cmd, psd_cmd = signal.welch(cmd, fs=fs, nperseg=min(256, len(cmd)//2))
        f_pos, psd_pos = signal.welch(pos, fs=fs, nperseg=min(256, len(pos)//2))
        
        results['frequencies'].append(f_cmd)
        results['cmd_psd'].append(psd_cmd)
        results['pos_psd'].append(psd_pos)
        
        # Coherence and transfer function estimate
        if len(cmd) > 256:
            f_coh, coh = signal.coherence(cmd, pos, fs=fs, nperseg=min(256, len(cmd)//2))
            results['coherence'].append((f_coh, coh))
            
            # Transfer function estimate (gain and phase)
            f_tf, Pxy = signal.csd(cmd, pos, fs=fs, nperseg=min(256, len(cmd)//2))
            _, Pxx = signal.welch(cmd, fs=fs, nperseg=min(256, len(cmd)//2))
            
            H = Pxy / (Pxx + 1e-10)  # Avoid division by zero
            gain = np.abs(H)
            phase = np.angle(H, deg=True)
            results['gain'].append((f_tf, gain))
            results['phase'].append((f_tf, phase))
        else:
            results['coherence'].append((np.array([]), np.array([])))
            results['gain'].append((np.array([]), np.array([])))
            results['phase'].append((np.array([]), np.array([])))
    
    return results


def plot_tracking_comparison(sim_data: dict, real_data: dict,
                             joint_indices: Optional[list] = None,
                             output_dir: Optional[str] = None) -> None:
    """Plot side-by-side tracking comparison between sim and real.
    
    Args:
        sim_data: Simulation tracking data
        real_data: Real robot tracking data
        joint_indices: List of joint indices to plot (None = all)
        output_dir: Directory to save plots
    """
    sim_times = sim_data['timestamps']
    sim_commands = sim_data['commands']
    sim_positions = sim_data['positions']
    
    real_times = real_data['timestamps']
    real_commands = real_data['commands']
    real_positions = real_data['positions']
    
    num_joints = sim_commands.shape[1]
    joint_names = sim_data.get('joint_names', [f'Joint {i}' for i in range(num_joints)])
    if isinstance(joint_names, np.ndarray):
        joint_names = joint_names.tolist()
    
    if joint_indices is None:
        joint_indices = list(range(num_joints))
    
    # Create figure with subplots: 2 columns (sim, real) x N joints
    fig, axes = plt.subplots(len(joint_indices), 2, figsize=(16, 3 * len(joint_indices)))
    if len(joint_indices) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sim vs Real: Joint Tracking Comparison', fontsize=14, fontweight='bold')
    
    for idx, j in enumerate(joint_indices):
        # Simulation subplot
        ax_sim = axes[idx, 0]
        ax_sim.plot(sim_times, sim_commands[:, j], 'g-', label='Command', linewidth=1.5)
        ax_sim.plot(sim_times, sim_positions[:, j], 'r-', label='Actual', linewidth=1.5, alpha=0.8)
        ax_sim.fill_between(sim_times, sim_commands[:, j], sim_positions[:, j], 
                           alpha=0.3, color='orange', label='Error')
        ax_sim.set_title(f'{joint_names[j]} - SIMULATION', fontsize=10)
        ax_sim.set_xlabel('Time (s)')
        ax_sim.set_ylabel('Position (rad)')
        ax_sim.legend(loc='upper right', fontsize=8)
        ax_sim.grid(True, alpha=0.3)
        
        # Real robot subplot
        ax_real = axes[idx, 1]
        ax_real.plot(real_times, real_commands[:, j], 'g-', label='Command', linewidth=1.5)
        ax_real.plot(real_times, real_positions[:, j], 'r-', label='Actual', linewidth=1.5, alpha=0.8)
        ax_real.fill_between(real_times, real_commands[:, j], real_positions[:, j],
                            alpha=0.3, color='orange', label='Error')
        ax_real.set_title(f'{joint_names[j]} - REAL', fontsize=10)
        ax_real.set_xlabel('Time (s)')
        ax_real.set_ylabel('Position (rad)')
        ax_real.legend(loc='upper right', fontsize=8)
        ax_real.grid(True, alpha=0.3)
        
        # Match y-axis limits
        y_min = min(ax_sim.get_ylim()[0], ax_real.get_ylim()[0])
        y_max = max(ax_sim.get_ylim()[1], ax_real.get_ylim()[1])
        ax_sim.set_ylim(y_min, y_max)
        ax_real.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'tracking_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {Path(output_dir) / 'tracking_comparison.png'}")
    
    plt.show()


def plot_error_comparison(sim_data: dict, real_data: dict,
                          joint_indices: Optional[list] = None,
                          output_dir: Optional[str] = None) -> None:
    """Plot tracking error comparison between sim and real.
    
    Args:
        sim_data: Simulation tracking data
        real_data: Real robot tracking data
        joint_indices: List of joint indices to plot
        output_dir: Directory to save plots
    """
    sim_times = sim_data['timestamps']
    sim_error = sim_data['positions'] - sim_data['commands']
    
    real_times = real_data['timestamps']
    real_error = real_data['positions'] - real_data['commands']
    
    num_joints = sim_error.shape[1]
    joint_names = sim_data.get('joint_names', [f'Joint {i}' for i in range(num_joints)])
    if isinstance(joint_names, np.ndarray):
        joint_names = joint_names.tolist()
    
    if joint_indices is None:
        joint_indices = list(range(num_joints))
    
    # Create figure
    fig, axes = plt.subplots(len(joint_indices), 1, figsize=(14, 2.5 * len(joint_indices)))
    if len(joint_indices) == 1:
        axes = [axes]
    
    fig.suptitle('Sim vs Real: Tracking Error Comparison', fontsize=14, fontweight='bold')
    
    for idx, j in enumerate(joint_indices):
        ax = axes[idx]
        
        # Plot both errors on same axis
        ax.plot(sim_times, sim_error[:, j], 'b-', label='Sim Error', linewidth=1.5, alpha=0.8)
        ax.plot(real_times, real_error[:, j], 'r-', label='Real Error', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add RMSE annotations
        sim_rmse = np.sqrt(np.mean(sim_error[:, j] ** 2))
        real_rmse = np.sqrt(np.mean(real_error[:, j] ** 2))
        
        ax.text(0.02, 0.98, 
                f'Sim RMSE: {sim_rmse:.4f} rad\nReal RMSE: {real_rmse:.4f} rad\nRatio: {real_rmse/sim_rmse:.2f}x',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{joint_names[j]} - Tracking Error', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (rad)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'error_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {Path(output_dir) / 'error_comparison.png'}")
    
    plt.show()


def plot_metrics_comparison(sim_metrics: dict, real_metrics: dict,
                            joint_names: list,
                            joint_indices: Optional[list] = None,
                            output_dir: Optional[str] = None) -> None:
    """Plot bar chart comparison of tracking metrics.
    
    Args:
        sim_metrics: Simulation metrics dict
        real_metrics: Real robot metrics dict
        joint_names: List of joint names
        joint_indices: Joint indices to plot
        output_dir: Directory to save plots
    """
    if joint_indices is None:
        joint_indices = list(range(len(joint_names)))
    
    selected_names = [joint_names[i] for i in joint_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sim vs Real: Tracking Metrics Comparison', fontsize=14, fontweight='bold')
    
    x = np.arange(len(joint_indices))
    width = 0.35
    
    # RMSE comparison
    ax = axes[0, 0]
    sim_rmse = [sim_metrics['rmse'][i] for i in joint_indices]
    real_rmse = [real_metrics['rmse'][i] for i in joint_indices]
    bars1 = ax.bar(x - width/2, sim_rmse, width, label='Simulation', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, real_rmse, width, label='Real', color='red', alpha=0.7)
    ax.set_ylabel('RMSE (rad)')
    ax.set_title('Root Mean Square Error')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Max error comparison
    ax = axes[0, 1]
    sim_max = [sim_metrics['max_error'][i] for i in joint_indices]
    real_max = [real_metrics['max_error'][i] for i in joint_indices]
    ax.bar(x - width/2, sim_max, width, label='Simulation', color='blue', alpha=0.7)
    ax.bar(x + width/2, real_max, width, label='Real', color='red', alpha=0.7)
    ax.set_ylabel('Max Error (rad)')
    ax.set_title('Maximum Absolute Error')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Delay comparison
    ax = axes[1, 0]
    sim_delay = [sim_metrics['delay_time'][i] * 1000 for i in joint_indices]  # Convert to ms
    real_delay = [real_metrics['delay_time'][i] * 1000 for i in joint_indices]
    ax.bar(x - width/2, sim_delay, width, label='Simulation', color='blue', alpha=0.7)
    ax.bar(x + width/2, real_delay, width, label='Real', color='red', alpha=0.7)
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Estimated Tracking Delay')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Correlation comparison
    ax = axes[1, 1]
    sim_corr = [sim_metrics['correlation'][i] for i in joint_indices]
    real_corr = [real_metrics['correlation'][i] for i in joint_indices]
    ax.bar(x - width/2, sim_corr, width, label='Simulation', color='blue', alpha=0.7)
    ax.bar(x + width/2, real_corr, width, label='Real', color='red', alpha=0.7)
    ax.set_ylabel('Correlation')
    ax.set_title('Command-Position Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(selected_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {Path(output_dir) / 'metrics_comparison.png'}")
    
    plt.show()


def plot_frequency_comparison(sim_freq: dict, real_freq: dict,
                              joint_names: list,
                              joint_indices: Optional[list] = None,
                              output_dir: Optional[str] = None) -> None:
    """Plot frequency response comparison.
    
    Args:
        sim_freq: Simulation frequency analysis
        real_freq: Real robot frequency analysis
        joint_names: List of joint names
        joint_indices: Joint indices to plot
        output_dir: Directory to save plots
    """
    if joint_indices is None:
        joint_indices = list(range(len(joint_names)))
    
    fig, axes = plt.subplots(len(joint_indices), 2, figsize=(14, 3 * len(joint_indices)))
    if len(joint_indices) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Sim vs Real: Frequency Response Comparison', fontsize=14, fontweight='bold')
    
    for idx, j in enumerate(joint_indices):
        # PSD comparison
        ax_psd = axes[idx, 0]
        
        sim_f = sim_freq['frequencies'][j]
        sim_psd = sim_freq['pos_psd'][j]
        real_f = real_freq['frequencies'][j]
        real_psd = real_freq['pos_psd'][j]
        
        ax_psd.semilogy(sim_f, sim_psd, 'b-', label='Sim Position PSD', linewidth=1.5)
        ax_psd.semilogy(real_f, real_psd, 'r-', label='Real Position PSD', linewidth=1.5)
        ax_psd.set_title(f'{joint_names[j]} - Power Spectral Density', fontsize=10)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('PSD')
        ax_psd.legend(loc='upper right', fontsize=8)
        ax_psd.grid(True, alpha=0.3)
        ax_psd.set_xlim(0, min(sim_f[-1], real_f[-1]) if len(sim_f) > 0 and len(real_f) > 0 else 10)
        
        # Coherence comparison
        ax_coh = axes[idx, 1]
        
        if len(sim_freq['coherence'][j][0]) > 0:
            sim_f_coh, sim_coh = sim_freq['coherence'][j]
            ax_coh.plot(sim_f_coh, sim_coh, 'b-', label='Sim Coherence', linewidth=1.5)
        
        if len(real_freq['coherence'][j][0]) > 0:
            real_f_coh, real_coh = real_freq['coherence'][j]
            ax_coh.plot(real_f_coh, real_coh, 'r-', label='Real Coherence', linewidth=1.5)
        
        ax_coh.set_title(f'{joint_names[j]} - Command-Position Coherence', fontsize=10)
        ax_coh.set_xlabel('Frequency (Hz)')
        ax_coh.set_ylabel('Coherence')
        ax_coh.legend(loc='upper right', fontsize=8)
        ax_coh.grid(True, alpha=0.3)
        ax_coh.set_ylim(0, 1.1)
        ax_coh.set_xlim(0, 10)  # Focus on low frequencies
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(Path(output_dir) / 'frequency_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {Path(output_dir) / 'frequency_comparison.png'}")
    
    plt.show()


def print_analysis_report(sim_metrics: dict, real_metrics: dict,
                          joint_names: list) -> None:
    """Print a text summary of the sim-to-real analysis."""
    
    print("\n" + "=" * 70)
    print("SIM-TO-REAL ACTUATION ANALYSIS REPORT")
    print("=" * 70)
    
    num_joints = len(joint_names)
    
    # Overall summary
    sim_avg_rmse = np.mean(sim_metrics['rmse'])
    real_avg_rmse = np.mean(real_metrics['rmse'])
    rmse_ratio = real_avg_rmse / sim_avg_rmse if sim_avg_rmse > 0 else float('inf')
    
    print(f"\n{'OVERALL SUMMARY':^70}")
    print("-" * 70)
    print(f"  Average RMSE - Sim: {sim_avg_rmse:.4f} rad, Real: {real_avg_rmse:.4f} rad")
    print(f"  Reality Gap Ratio (Real/Sim): {rmse_ratio:.2f}x")
    
    sim_avg_delay = np.mean(sim_metrics['delay_time']) * 1000
    real_avg_delay = np.mean(real_metrics['delay_time']) * 1000
    print(f"  Average Delay - Sim: {sim_avg_delay:.1f} ms, Real: {real_avg_delay:.1f} ms")
    
    # Per-joint summary
    print(f"\n{'PER-JOINT COMPARISON':^70}")
    print("-" * 70)
    print(f"{'Joint':<15} {'Sim RMSE':>10} {'Real RMSE':>10} {'Ratio':>8} {'Sim Delay':>10} {'Real Delay':>10}")
    print("-" * 70)
    
    for j in range(num_joints):
        name = joint_names[j] if j < len(joint_names) else f"Joint {j}"
        sim_rmse = sim_metrics['rmse'][j]
        real_rmse = real_metrics['rmse'][j]
        ratio = real_rmse / sim_rmse if sim_rmse > 0 else float('inf')
        sim_delay = sim_metrics['delay_time'][j] * 1000
        real_delay = real_metrics['delay_time'][j] * 1000
        
        print(f"{name:<15} {sim_rmse:>10.4f} {real_rmse:>10.4f} {ratio:>8.2f}x {sim_delay:>9.1f}ms {real_delay:>9.1f}ms")
    
    # Diagnosis
    print(f"\n{'DIAGNOSIS':^70}")
    print("-" * 70)
    
    # Check for problematic joints
    problematic = []
    for j in range(num_joints):
        ratio = real_metrics['rmse'][j] / sim_metrics['rmse'][j] if sim_metrics['rmse'][j] > 0 else float('inf')
        if ratio > 2.0:
            problematic.append((joint_names[j], ratio))
    
    if problematic:
        print("  ⚠️  Joints with significant reality gap (>2x RMSE):")
        for name, ratio in problematic:
            print(f"      - {name}: {ratio:.2f}x worse in real")
    else:
        print("  ✓ All joints have acceptable tracking performance")
    
    # Check for delay issues
    delay_issues = []
    for j in range(num_joints):
        real_delay = real_metrics['delay_time'][j] * 1000
        if real_delay > 50:  # More than 50ms delay
            delay_issues.append((joint_names[j], real_delay))
    
    if delay_issues:
        print("\n  ⚠️  Joints with high latency (>50ms):")
        for name, delay in delay_issues:
            print(f"      - {name}: {delay:.1f}ms")
    
    # Recommendations
    print(f"\n{'RECOMMENDATIONS':^70}")
    print("-" * 70)
    
    if rmse_ratio > 1.5:
        print("  • Consider increasing PD gains in simulation to match real tracking error")
        print("  • Add actuation delay/latency to simulation")
    
    if real_avg_delay > sim_avg_delay + 10:
        print(f"  • Real robot has {real_avg_delay - sim_avg_delay:.1f}ms more delay than sim")
        print("    Consider adding latency simulation with latency_scheme config")
    
    if any(real_metrics['correlation'][j] < 0.8 for j in range(num_joints)):
        print("  • Some joints have low command-position correlation")
        print("    Check for mechanical issues or communication delays")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Sim-to-Real Actuation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic comparison
    python scripts/sim2real_analysis.py sim_robot_tracking.npz real_robot_tracking.npz
    
    # Save plots to directory
    python scripts/sim2real_analysis.py sim.npz real.npz --output ./analysis/
    
    # Analyze specific joints only
    python scripts/sim2real_analysis.py sim.npz real.npz --joints 0 1 2
    
    # Skip interactive plots (just save)
    python scripts/sim2real_analysis.py sim.npz real.npz --output ./analysis/ --no-show
        """
    )
    
    parser.add_argument("sim_file", type=str, help="Path to simulation tracking data (.npz or .pkl)")
    parser.add_argument("real_file", type=str, help="Path to real robot tracking data (.npz or .pkl)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for saving plots and reports")
    parser.add_argument("--joints", "-j", type=int, nargs="+", default=None,
                        help="Specific joint indices to analyze (default: all)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show interactive plots (just save)")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"Loading simulation data from: {args.sim_file}")
    sim_data = load_tracking_data(args.sim_file)
    
    print(f"Loading real robot data from: {args.real_file}")
    real_data = load_tracking_data(args.real_file)
    
    # Get joint info
    num_joints = sim_data['commands'].shape[1]
    joint_names = sim_data.get('joint_names', [f'Joint {i}' for i in range(num_joints)])
    if isinstance(joint_names, np.ndarray):
        joint_names = joint_names.tolist()
    
    joint_indices = args.joints if args.joints else list(range(num_joints))
    
    print(f"\nAnalyzing {len(joint_indices)} joints: {[joint_names[i] for i in joint_indices]}")
    print(f"Simulation: {len(sim_data['timestamps'])} timesteps")
    print(f"Real robot: {len(real_data['timestamps'])} timesteps")
    
    # Compute metrics
    print("\nComputing tracking metrics...")
    sim_metrics = compute_tracking_metrics(
        sim_data['commands'], sim_data['positions'], sim_data['timestamps']
    )
    real_metrics = compute_tracking_metrics(
        real_data['commands'], real_data['positions'], real_data['timestamps']
    )
    
    # Compute frequency response
    print("Computing frequency response...")
    sim_freq = compute_frequency_response(
        sim_data['commands'], sim_data['positions'], sim_data['timestamps']
    )
    real_freq = compute_frequency_response(
        real_data['commands'], real_data['positions'], real_data['timestamps']
    )
    
    # Print text report
    print_analysis_report(sim_metrics, real_metrics, joint_names)
    
    # Save report to file if output specified
    if args.output:
        report_path = Path(args.output) / 'analysis_report.txt'
        import io
        import sys
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        print_analysis_report(sim_metrics, real_metrics, joint_names)
        report_text = buffer.getvalue()
        sys.stdout = old_stdout
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nSaved report to: {report_path}")
    
    # Generate plots
    if not args.no_show or args.output:
        print("\nGenerating comparison plots...")
        
        # Temporarily disable interactive mode if no-show
        if args.no_show:
            plt.ioff()
        
        plot_tracking_comparison(sim_data, real_data, joint_indices, args.output)
        plot_error_comparison(sim_data, real_data, joint_indices, args.output)
        plot_metrics_comparison(sim_metrics, real_metrics, joint_names, joint_indices, args.output)
        plot_frequency_comparison(sim_freq, real_freq, joint_names, joint_indices, args.output)
        
        if args.no_show:
            plt.ion()
    
    print("\n✓ Analysis complete!")
    
    if args.output:
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
