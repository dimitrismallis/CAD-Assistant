#!/usr/bin/env python3
"""
SGP-Bench Dataset Loader
========================

A comprehensive data loader for the SGP-bench dataset from Hugging Face.
Supports single-threaded and parallel execution with debugging capabilities.

Dataset: https://huggingface.co/datasets/sgp-bench/sgp-bench
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pandas as pd

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ Hugging Face datasets not installed. Run: pip install datasets")


@dataclass
class SGPSample:
    """Represents a single sample from the SGP-bench dataset."""
    pid: int
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str
    subject: str
    
    @property
    def options(self) -> Dict[str, str]:
        """Get all options as a dictionary."""
        return {
            'A': self.option_a,
            'B': self.option_b,
            'C': self.option_c,
            'D': self.option_d
        }
    
    @property
    def correct_option(self) -> str:
        """Get the text of the correct answer."""
        return self.options.get(self.answer, "Unknown")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'pid': self.pid,
            'question': self.question,
            'options': self.options,
            'answer': self.answer,
            'correct_option': self.correct_option,
            'subject': self.subject
        }


class SGPBenchLoader:
    """
    Data loader for SGP-bench dataset with support for single-threaded 
    and parallel execution.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, debug: bool = False):
        """
        Initialize the SGP-bench loader.
        
        Args:
            cache_dir: Directory to cache the dataset
            debug: Enable debug logging
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("Hugging Face datasets library is required. Run: pip install datasets")
        
        self.cache_dir = cache_dir
        self.debug = debug
        self.dataset = None
        self.splits = {}
        
        print("🚀 Initializing SGP-bench loader...")
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            print("📥 Loading SGP-bench CAD split from Hugging Face...")
            # Only load the CAD split to save memory and time
            self.dataset = load_dataset("sgp-bench/sgp-bench", cache_dir=self.cache_dir, split={'cad': 'cad'})
            
            # Only track the CAD split
            self.splits['cad'] = len(self.dataset['cad'])
            if self.debug:
                print(f"  📂 CAD split: {self.splits['cad']} samples")
            
            print(f"✅ CAD split loaded successfully! {self.splits['cad']:,} samples")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
    
    def get_split_info(self) -> Dict[str, int]:
        """Get information about available splits."""
        return self.splits.copy()
    
    def get_sample(self, split: str, index: int) -> SGPSample:
        """
        Get a single sample from the dataset.
        
        Args:
            split: Dataset split name (only 'cad' is supported)
            index: Sample index
            
        Returns:
            SGPSample object
        """
        if split != 'cad':
            raise ValueError(f"Only 'cad' split is supported. Requested: '{split}'")
        
        if index >= len(self.dataset['cad']):
            raise IndexError(f"Index {index} out of range for CAD split (size: {len(self.dataset['cad'])})")
        
        row = self.dataset['cad'][index]
        
        return SGPSample(
            pid=row['PID'],
            question=row['Question'],
            option_a=row['A'],
            option_b=row['B'],
            option_c=row['C'],
            option_d=row['D'],
            answer=row['Answer'],
            subject=row['Subject']
        )
    
    def get_samples(self, split: str, indices: Optional[List[int]] = None, 
                   limit: Optional[int] = None) -> List[SGPSample]:
        """
        Get multiple samples from the dataset.
        
        Args:
            split: Dataset split name (only 'cad' is supported)
            indices: Specific indices to retrieve (if None, get all or up to limit)
            limit: Maximum number of samples to retrieve
            
        Returns:
            List of SGPSample objects
        """
        if split != 'cad':
            raise ValueError(f"Only 'cad' split is supported. Requested: '{split}'")
        
        split_data = self.dataset['cad']
        
        if indices is not None:
            # Get specific indices
            samples = []
            for idx in indices:
                if idx < len(split_data):
                    samples.append(self.get_sample(split, idx))
                else:
                    print(f"⚠️ Index {idx} out of range for split '{split}'")
        else:
            # Get all samples or up to limit
            max_samples = len(split_data) if limit is None else min(limit, len(split_data))
            samples = [self.get_sample(split, i) for i in range(max_samples)]
        
        return samples
    
    def iterate_split(self, split: str, batch_size: int = 1) -> Iterator[List[SGPSample]]:
        """
        Iterate through a split in batches.
        
        Args:
            split: Dataset split name (only 'cad' is supported)
            batch_size: Number of samples per batch
            
        Yields:
            Batches of SGPSample objects
        """
        if split != 'cad':
            raise ValueError(f"Only 'cad' split is supported. Requested: '{split}'")
        
        split_size = len(self.dataset['cad'])
        
        for start_idx in range(0, split_size, batch_size):
            end_idx = min(start_idx + batch_size, split_size)
            batch = [self.get_sample(split, i) for i in range(start_idx, end_idx)]
            yield batch
    
    def process_samples_sequential(self, samples: List[SGPSample], 
                                 processor_func: Callable[[SGPSample], Any]) -> List[Any]:
        """
        Process samples sequentially (single-threaded).
        
        Args:
            samples: List of samples to process
            processor_func: Function to process each sample
            
        Returns:
            List of processing results
        """
        print(f"🔄 Processing {len(samples)} samples sequentially...")
        start_time = time.time()
        
        results = []
        for i, sample in enumerate(samples):
            if self.debug:
                # Handle both individual samples and tuples (sample, index)
                if isinstance(sample, tuple):
                    actual_sample = sample[0]
                    print(f"  Processing sample {i+1}/{len(samples)} (PID: {actual_sample.pid})")
                else:
                    print(f"  Processing sample {i+1}/{len(samples)} (PID: {sample.pid})")
            
            try:
                result = processor_func(sample)
                results.append(result)
            except Exception as e:
                # Handle both individual samples and tuples (sample, index)
                if isinstance(sample, tuple):
                    actual_sample = sample[0]
                    print(f"❌ Error processing sample {actual_sample.pid}: {e}")
                else:
                    print(f"❌ Error processing sample {sample.pid}: {e}")
                results.append(None)
        
        elapsed = time.time() - start_time
        print(f"✅ Sequential processing completed in {elapsed:.2f}s ({elapsed/len(samples):.3f}s per sample)")
        
        return results
    
    def process_samples_parallel(self, samples: List[SGPSample], 
                               processor_func: Callable[[SGPSample], Any],
                               max_workers: Optional[int] = None,
                               use_processes: bool = False) -> List[Any]:
        """
        Process samples in parallel (multi-threaded or multi-process).
        
        Args:
            samples: List of samples to process
            processor_func: Function to process each sample
            max_workers: Maximum number of workers (default: CPU count)
            use_processes: Use ProcessPoolExecutor instead of ThreadPoolExecutor
            
        Returns:
            List of processing results
        """
        if max_workers is None:
            max_workers = min(cpu_count(), len(samples))
        
        executor_type = "process" if use_processes else "thread"
        print(f"⚡ Processing {len(samples)} samples in parallel ({executor_type}-based, {max_workers} workers)...")
        start_time = time.time()
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        results = [None] * len(samples)
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(processor_func, sample): i 
                for i, sample in enumerate(samples)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completed += 1
                
                try:
                    result = future.result()
                    results[index] = result
                    
                    if self.debug:
                        sample = samples[index]
                        # Handle both individual samples and tuples (sample, index)
                        if isinstance(sample, tuple):
                            actual_sample = sample[0]
                            print(f"  ✅ Completed sample {completed}/{len(samples)} (PID: {actual_sample.pid})")
                        else:
                            print(f"  ✅ Completed sample {completed}/{len(samples)} (PID: {sample.pid})")
                        
                except Exception as e:
                    sample = samples[index]
                    # Handle both individual samples and tuples (sample, index)
                    if isinstance(sample, tuple):
                        actual_sample = sample[0]
                        print(f"❌ Error processing sample {actual_sample.pid}: {e}")
                    else:
                        print(f"❌ Error processing sample {sample.pid}: {e}")
                    results[index] = None
        
        elapsed = time.time() - start_time
        print(f"✅ Parallel processing completed in {elapsed:.2f}s ({elapsed/len(samples):.3f}s per sample)")
        
        return results
    
    def debug_samples(self, num_samples: int = 3):
        """
        Debug by examining a few samples from the CAD dataset.
        
        Args:
            num_samples: Number of samples to examine
        """
        print(f"🔍 Debugging {num_samples} CAD samples...")
        
        samples = self.get_samples("cad", limit=num_samples)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n📋 Sample {i} (PID: {sample.pid}):")
            print(f"  Subject: {sample.subject}")
            print(f"  Question: {sample.question[:200]}{'...' if len(sample.question) > 200 else ''}")
            print(f"  Options:")
            for key, value in sample.options.items():
                marker = "✅" if key == sample.answer else "  "
                print(f"    {marker} {key}: {value}")
            print(f"  Correct Answer: {sample.answer} ({sample.correct_option})")
    
    def save_samples(self, samples: List[SGPSample], filename: str):
        """
        Save samples to a JSON file.
        
        Args:
            samples: List of samples to save
            filename: Output filename
        """
        data = [sample.to_dict() for sample in samples]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(samples)} samples to {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics for CAD split."""
        stats = {
            'total_samples': self.splits['cad'],
            'splits': self.splits.copy(),
            'subjects': {}
        }
        
        # Get subject distribution for CAD split
        split_data = self.dataset['cad']
        subjects = {}
        
        for row in split_data:
            subject = row['Subject']
            subjects[subject] = subjects.get(subject, 0) + 1
        
        stats['subjects']['cad'] = subjects
        
        return stats


def demo_processor_function(sample: SGPSample) -> Dict[str, Any]:
    """
    Demo processor function that analyzes a sample.
    This is an example of how you might process samples.
    """
    # Simulate some processing time
    time.sleep(0.1)
    
    # Extract some basic information
    result = {
        'pid': sample.pid,
        'subject': sample.subject,
        'question_length': len(sample.question),
        'has_cad_code': 'SOL;' in sample.question,
        'num_options': len(sample.options),
        'answer': sample.answer,
        'processed_at': time.time()
    }
    
    return result


def main():
    """Main function to demonstrate the SGP-bench loader."""
    print("🧪 SGP-Bench Loader Demo")
    print("=" * 50)
    
    try:
        # Initialize loader
        loader = SGPBenchLoader(debug=True)
        
        # Show dataset statistics
        print("\n📊 CAD Dataset Statistics:")
        stats = loader.get_statistics()
        print(f"Total CAD samples: {stats['total_samples']:,}")
        
        # Debug a few samples
        print("\n" + "="*50)
        loader.debug_samples(num_samples=2)
        
        # Test sequential processing
        print("\n" + "="*50)
        test_samples = loader.get_samples("cad", limit=5)
        sequential_results = loader.process_samples_sequential(test_samples, demo_processor_function)
        
        # Test parallel processing
        print("\n" + "="*50)
        parallel_results = loader.process_samples_parallel(test_samples, demo_processor_function, max_workers=2)
        
        # Save results
        print("\n" + "="*50)
        loader.save_samples(test_samples, "sgp_bench_debug_samples.json")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 