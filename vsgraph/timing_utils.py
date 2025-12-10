"""
Timing utilities for performance comparison between different implementations.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
import numpy as np


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.verbose:
            print(f"Starting: {self.name}")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"Completed: {self.name} in {self.elapsed:.4f}s")
        return False


class PerformanceTimer:
    """
    Timer for tracking multiple operations and computing statistics.

    Usage:
        timer = PerformanceTimer()

        with timer.time("operation1"):
            # code to time

        with timer.time("operation2"):
            # more code

        results = timer.get_summary()
    """

    def __init__(self, name: str = "Performance"):
        self.name = name
        self.timings: Dict[str, List[float]] = {}
        self.current_operation: Optional[str] = None

    @contextmanager
    def time(self, operation: str, verbose: bool = False):
        """Time a code block."""
        if operation not in self.timings:
            self.timings[operation] = []

        if verbose:
            print(f"[{self.name}] Starting: {operation}")

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[operation].append(elapsed)
            if verbose:
                print(f"[{self.name}] Completed: {operation} in {elapsed:.4f}s")

    def get_summary(self) -> Dict:
        """Get summary statistics for all timed operations."""
        summary = {}
        for operation, times in self.timings.items():
            times_array = np.array(times)
            summary[operation] = {
                'count': len(times),
                'total': float(np.sum(times_array)),
                'mean': float(np.mean(times_array)),
                'std': float(np.std(times_array)),
                'min': float(np.min(times_array)),
                'max': float(np.max(times_array)),
                'median': float(np.median(times_array))
            }
        return summary

    def print_summary(self, sort_by: str = 'total'):
        """Print formatted summary of timings."""
        summary = self.get_summary()

        print(f"\n{'='*70}")
        print(f"{self.name} - Timing Summary")
        print(f"{'='*70}")

        if not summary:
            print("No timings recorded.")
            return

        # Sort operations
        if sort_by in ['total', 'mean', 'max']:
            sorted_ops = sorted(summary.items(), key=lambda x: x[1][sort_by], reverse=True)
        else:
            sorted_ops = sorted(summary.items())

        print(f"{'Operation':<30} {'Count':<8} {'Total (s)':<12} {'Mean (s)':<12} {'Std (s)':<12}")
        print(f"{'-'*70}")

        for operation, stats in sorted_ops:
            print(f"{operation:<30} {stats['count']:<8} {stats['total']:>10.4f}  "
                  f"{stats['mean']:>10.4f}  {stats['std']:>10.4f}")

        print(f"{'='*70}\n")

    def reset(self):
        """Clear all timing data."""
        self.timings.clear()


class ComparisonTimer:
    """
    Timer for comparing performance between two implementations.

    Usage:
        timer = ComparisonTimer()

        with timer.time_version("original"):
            # original implementation

        with timer.time_version("optimized"):
            # optimized implementation

        timer.print_comparison()
    """

    def __init__(self, name: str = "Comparison"):
        self.name = name
        self.versions: Dict[str, PerformanceTimer] = {}

    @contextmanager
    def time_version(self, version: str, operation: str = "default", verbose: bool = False):
        """Time a specific version of an operation."""
        if version not in self.versions:
            self.versions[version] = PerformanceTimer(version)

        with self.versions[version].time(operation, verbose=verbose):
            yield

    def get_comparison(self, operation: str = "default") -> Dict:
        """Get comparison between versions for a specific operation."""
        comparison = {}

        for version, timer in self.versions.items():
            if operation in timer.timings:
                times = np.array(timer.timings[operation])
                comparison[version] = {
                    'count': len(times),
                    'total': float(np.sum(times)),
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times))
                }

        # Calculate speedup if we have exactly 2 versions
        if len(comparison) == 2:
            versions = list(comparison.keys())
            v1, v2 = versions[0], versions[1]

            comparison['speedup'] = {
                f'{v1}_vs_{v2}': comparison[v1]['mean'] / comparison[v2]['mean'],
                f'{v2}_vs_{v1}': comparison[v2]['mean'] / comparison[v1]['mean']
            }

        return comparison

    def print_comparison(self, operation: str = "default"):
        """Print formatted comparison between versions."""
        comparison = self.get_comparison(operation)

        print(f"\n{'='*70}")
        print(f"{self.name} - Performance Comparison")
        print(f"Operation: {operation}")
        print(f"{'='*70}")

        if not comparison:
            print("No comparison data available.")
            return

        # Print individual version stats
        print(f"\n{'Version':<20} {'Count':<8} {'Mean (s)':<15} {'Std (s)':<15}")
        print(f"{'-'*70}")

        for version, stats in comparison.items():
            if version == 'speedup':
                continue
            print(f"{version:<20} {stats['count']:<8} {stats['mean']:>13.6f}  {stats['std']:>13.6f}")

        # Print speedup if available
        if 'speedup' in comparison:
            print(f"\n{'Speedup Comparison':<40}")
            print(f"{'-'*70}")
            for comparison_name, speedup in comparison['speedup'].items():
                print(f"  {comparison_name:<35}: {speedup:>8.2f}×")

        print(f"{'='*70}\n")

    def print_all_comparisons(self):
        """Print comparisons for all operations."""
        # Get all unique operations
        all_operations = set()
        for timer in self.versions.values():
            all_operations.update(timer.timings.keys())

        for operation in sorted(all_operations):
            self.print_comparison(operation)


@contextmanager
def time_operation(name: str, verbose: bool = True):
    """
    Simple context manager to time a single operation.

    Usage:
        with time_operation("my_function"):
            # code to time
    """
    if verbose:
        print(f"Starting: {name}...", end=" ", flush=True)

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"Done in {elapsed:.4f}s")


def time_function(func):
    """
    Decorator to time function execution.

    Usage:
        @time_function
        def my_function():
            # code
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} completed in {elapsed:.4f}s")
        return result
    return wrapper
