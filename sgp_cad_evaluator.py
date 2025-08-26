#!/usr/bin/env python3
"""
SGP-Bench CAD Evaluator
=======================

Evaluation script for SGP-bench CAD split using the CAD Assistant system.
Supports single-threaded and parallel evaluation with detailed metrics.
"""

import os
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback

from sgp_bench_loader import SGPBenchLoader, SGPSample
from cad_assistant.core import CADAssistantCore
from cad_assistant.openai_planner import OpenAIPlannerChain


@dataclass
class EvaluationResult:
    """Results from evaluating a single CAD sample."""
    pid: int
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    reasoning: Optional[str] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    assistant_logs: Optional[str] = None
    # Majority voting fields
    trials: int = 1
    all_predictions: Optional[List[str]] = None
    vote_counts: Optional[Dict[str, int]] = None
    confidence: Optional[float] = None  # Fraction of trials that agreed with final answer


def evaluate_single_sample_isolated(sample_config_sketches_tuple):
    """
    Isolated function for evaluating a single sample in a separate process.
    This ensures complete isolation between parallel evaluations.
    
    Args:
        sample_config_sketches_tuple: Tuple of (sample, config_path, sketches_dir)
        
    Returns:
        EvaluationResult object
    """
    sample, config_path, sketches_dir = sample_config_sketches_tuple
    start_time = time.time()
    
    try:
        # Import here to ensure fresh imports in each process
        import json
        from cad_assistant.core import CADAssistantCore
        from cad_assistant.openai_planner import OpenAIPlannerChain
        
        # Load config to get max_steps
        with open(config_path, 'r') as f:
            config = json.load(f)
        max_steps = config.get("max_steps", 10)
        
        # Construct sketch file path using new directory structure: subject_PID/subject_PID.FCStd
        dir_name = f"{sample.subject}_{sample.pid:04d}"  # e.g., "2D_0001"
        fcstd_filename = f"{dir_name}.FCStd"
        sketch_file = os.path.join(sketches_dir, dir_name, fcstd_filename)
        
        if not os.path.exists(sketch_file):
            raise FileNotFoundError(f"Sketch file not found: {sketch_file}")
        
        # Create fresh assistant instance for this process
        planner = OpenAIPlannerChain(config_path=config_path)
        assistant = CADAssistantCore(
            planner_chain=planner,
            prompt_name="sgp_cad_evaluation",
            project_file=sketch_file
        )
        
        # Format question based on subject type
        question_pattern = re.compile(f"{re.escape('Question:')}.*", re.DOTALL)
        question_match = question_pattern.search(sample.question)
        question_only = question_match.group(0) if question_match else sample.question
        
        # Format options
        options_text = "\n".join([
            f"{key}: {value}" for key, value in sample.options.items()
        ])
        
        # Only handle 2D subjects now
        formatted_question = ("Answer the following multiple choice question. The last line of your response "
                            "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n\n"
                            "You are given a FreeCAD project file that includes a 2D CAD sketch. The FreeCAD project is "
                            "already loaded for you. Examine the sketch carefully to understand the 2D object it generates "
                            "and answer the question based on your interpretation of the rendered image of that object.\n\n" +
                            question_only + '\n' + options_text)
        
        # Run the assistant
        result = assistant.step(formatted_question, max_iterations=max_steps)
        
        # Extract prediction from the assistant's response
        response_text = result.get('output', '') or result.get('plan', '')
        
        # Extract answer using the same logic as the main class
        predicted_answer = None
        reasoning = response_text
        
        if response_text:
            # Look for the specific "Answer: X" format we requested
            answer_pattern = re.compile(r'Answer:\s*([ABCD])', re.IGNORECASE)
            match = answer_pattern.search(response_text)
            
            if match:
                predicted_answer = match.group(1).upper()
            else:
                # Fallback: look for the last occurrence of A, B, C, or D
                for char in reversed(response_text):
                    if char in ['A', 'B', 'C', 'D']:
                        predicted_answer = char
                        break
        
        # Check if correct
        is_correct = predicted_answer == sample.answer if predicted_answer else False
        
        execution_time = time.time() - start_time
        
        # Get assistant logs
        assistant_logs = assistant.get_log_directory() if hasattr(assistant, 'get_log_directory') else None
        
        return EvaluationResult(
            pid=sample.pid,
            predicted_answer=predicted_answer,
            correct_answer=sample.answer,
            is_correct=is_correct,
            reasoning=reasoning,
            execution_time=execution_time,
            assistant_logs=assistant_logs,
            trials=1
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error evaluating sample {sample.pid}: {str(e)}"
        
        return EvaluationResult(
            pid=sample.pid,
            predicted_answer=None,
            correct_answer=sample.answer,
            is_correct=False,
            execution_time=execution_time,
            error=error_msg,
            trials=1
        )


class SGPCADEvaluator:
    """
    Evaluator for SGP-bench CAD questions using CAD Assistant.
    """
    
    def __init__(self, config_path: str = "config.json", debug: bool = False):
        """
        Initialize the CAD evaluator.
        
        Args:
            config_path: Path to OpenAI config file
            debug: Enable debug logging
        """
        self.config_path = config_path
        self.debug = debug
        
        # Load config to get max_steps
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.max_steps = config.get("max_steps", 10)
        
        # Initialize data loader
        print("📥 Initializing SGP-bench loader...")
        self.loader = SGPBenchLoader(debug=debug)
        
        # Check if config exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"✅ Evaluator initialized for CAD split ({self.loader.splits['cad']:,} samples)")

    def _create_cad_assistant(self, project_file: str) -> CADAssistantCore:
        """Create a new CAD Assistant instance."""
        try:
            planner = OpenAIPlannerChain(config_path=self.config_path)
            assistant = CADAssistantCore(
                planner_chain=planner,
                prompt_name="sgp_cad_evaluation",
                project_file=project_file
            )
            return assistant
        except Exception as e:
            raise RuntimeError(f"Failed to create CAD Assistant: {e}")
    
    def _format_2d_question(self, sample: SGPSample) -> str:
        """
        Format the SGP 2D question for the assistant.
        
        Args:
            sample: SGP sample with 2D question
            
        Returns:
            Formatted question string
        """
        # Extract question using the same patterns as in process_2d_samples.py
        question_pattern = re.compile(f"{re.escape('Question:')}.*", re.DOTALL)
        question_match = question_pattern.search(sample.question)
        question_only = question_match.group(0) if question_match else sample.question
        
        # Format options
        options_text = "\n".join([
            f"{key}: {value}" for key, value in sample.options.items()
        ])
        
        formatted_question = ("Answer the following multiple choice question. The last line of your response "
                            "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n\n"
                            "You are given a FreeCAD project file that includes a 2D CAD sketch. The FreeCAD project is "
                            "already loaded for you. Examine the sketch carefully to understand the 2D object it generates "
                            "and answer the question based on your interpretation of the rendered image of that object.\n\n" +
                            question_only + '\n' + options_text)
        
        return formatted_question
    

    

    
    def _extract_answer_from_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the predicted answer and reasoning from assistant response.
        
        Args:
            response_text: Assistant's response text
            
        Returns:
            Tuple of (predicted_answer, reasoning)
        """
        if not response_text:
            return None, None
        
        # Look for the specific "Answer: X" format we requested
        answer_pattern = re.compile(r'Answer:\s*([ABCD])', re.IGNORECASE)
        match = answer_pattern.search(response_text)
        
        if match:
            predicted_answer = match.group(1).upper()
        else:
            # Fallback: look for the last occurrence of A, B, C, or D
            predicted_answer = None
            for char in reversed(response_text):
                if char in ['A', 'B', 'C', 'D']:
                    predicted_answer = char
                    break
        
        return predicted_answer, response_text
    
    def _perform_majority_vote(self, results: List[EvaluationResult], sample: SGPSample) -> EvaluationResult:
        """
        Perform majority voting on multiple evaluation results for the same sample.
        
        Args:
            results: List of evaluation results from multiple trials
            sample: The original sample being evaluated
            
        Returns:
            Single EvaluationResult with majority vote
        """
        if not results:
            return EvaluationResult(
                pid=sample.pid,
                predicted_answer=None,
                correct_answer=sample.answer,
                is_correct=False,
                error="No trial results available",
                trials=0
            )
        
        # Filter out failed results
        valid_results = [r for r in results if r.predicted_answer is not None and r.error is None]
        
        if not valid_results:
            # All trials failed
            return EvaluationResult(
                pid=sample.pid,
                predicted_answer=None,
                correct_answer=sample.answer,
                is_correct=False,
                error=f"All {len(results)} trials failed",
                trials=len(results),
                all_predictions=[r.predicted_answer for r in results]
            )
        
        # Count votes
        vote_counts = {}
        all_predictions = []
        total_execution_time = 0
        all_reasoning = []
        
        for result in valid_results:
            pred = result.predicted_answer
            all_predictions.append(pred)
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
            total_execution_time += result.execution_time
            if result.reasoning:
                all_reasoning.append(f"Trial: {result.reasoning}")
        
        # Add None predictions from failed trials
        for result in results:
            if result.predicted_answer is None:
                all_predictions.append(None)
        
        # Find majority answer
        if vote_counts:
            majority_answer = max(vote_counts.items(), key=lambda x: x[1])[0]
            majority_count = vote_counts[majority_answer]
        else:
            majority_answer = None
            majority_count = 0
        
        # Calculate confidence (fraction of valid trials that agreed)
        confidence = majority_count / len(valid_results) if valid_results else 0.0
        
        # Check if correct
        is_correct = majority_answer == sample.answer if majority_answer else False
        
        # Combine reasoning from all trials
        combined_reasoning = "\n\n---\n\n".join(all_reasoning) if all_reasoning else None
        
        return EvaluationResult(
            pid=sample.pid,
            predicted_answer=majority_answer,
            correct_answer=sample.answer,
            is_correct=is_correct,
            reasoning=combined_reasoning,
            execution_time=total_execution_time,
            trials=len(results),
            all_predictions=all_predictions,
            vote_counts=vote_counts,
            confidence=confidence
        )
    
    def evaluate_sample(self, sample: SGPSample, sketches_dir: str = "sgp_bench_samples") -> EvaluationResult:
        """
        Evaluate a single 2D CAD sample using the assistant.
        
        Args:
            sample: SGP sample to evaluate
            sketches_dir: Directory containing pre-generated FreeCAD files
            
        Returns:
            EvaluationResult object
        """
        start_time = time.time()
        
        try:
            if self.debug:
                print(f"🔍 Evaluating sample {sample.pid}")
            
            # Construct sketch file path using new directory structure: subject_PID/subject_PID.FCStd
            dir_name = f"{sample.subject}_{sample.pid:04d}"  # e.g., "2D_0001"
            fcstd_filename = f"{dir_name}.FCStd"
            sketch_file = os.path.join(sketches_dir, dir_name, fcstd_filename)
            
            if not os.path.exists(sketch_file):
                raise FileNotFoundError(f"Sketch file not found: {sketch_file}")
            
            if self.debug:
                print(f"  📁 Loading sketch: {sketch_file}")
            
            # Create assistant with the specific sketch file
            assistant = self._create_cad_assistant(project_file=sketch_file)
            
            # Format question for 2D samples only
            formatted_question = self._format_2d_question(sample)
            
            # Run the assistant
            result = assistant.step(formatted_question, max_iterations=self.max_steps)
            
            # Extract prediction from the assistant's response
            response_text = result.get('output', '') or result.get('plan', '')
            predicted_answer, reasoning = self._extract_answer_from_response(response_text)
            
            # Check if correct
            is_correct = predicted_answer == sample.answer if predicted_answer else False
            
            execution_time = time.time() - start_time
            
            # Get assistant logs
            assistant_logs = assistant.get_log_directory() if hasattr(assistant, 'get_log_directory') else None
            
            return EvaluationResult(
                pid=sample.pid,
                predicted_answer=predicted_answer,
                correct_answer=sample.answer,
                is_correct=is_correct,
                reasoning=reasoning,
                execution_time=execution_time,
                assistant_logs=assistant_logs,
                trials=1
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error evaluating sample {sample.pid}: {str(e)}"
            
            if self.debug:
                print(f"❌ {error_msg}")
                traceback.print_exc()
            
            return EvaluationResult(
                pid=sample.pid,
                predicted_answer=None,
                correct_answer=sample.answer,
                is_correct=False,
                execution_time=execution_time,
                error=error_msg,
                trials=1
            )
    
    def evaluate_samples(self, 
                        sample_indices: Optional[List[int]] = None,
                        limit: Optional[int] = None,
                        parallel: bool = False,
                        max_workers: Optional[int] = None,
                        sketches_dir: str = "sgp_bench_samples",
                        trials: int = 1,
                        subject: str = "2D") -> List[EvaluationResult]:
        """
        Evaluate multiple CAD samples.
        
        Args:
            sample_indices: Specific sample indices to evaluate
            limit: Maximum number of samples to evaluate  
            parallel: Use parallel processing
            max_workers: Number of parallel workers
            sketches_dir: Directory containing pre-generated FreeCAD files
            trials: Number of trials per sample for majority voting
            subject: Subject type to evaluate (2D only)
            
        Returns:
            List of evaluation results
        """
        # Get all samples and filter by subject (only 2D supported)
        all_samples = self.loader.get_samples("cad", limit=None)
        filtered_samples = [s for s in all_samples if s.subject == "2D"]
        
        if sample_indices:
            # If specific indices provided, get those samples
            samples = [filtered_samples[i] for i in sample_indices if i < len(filtered_samples)]
        else:
            # Otherwise use limit
            samples = filtered_samples[:limit] if limit else filtered_samples
        
        print(f"🔍 Evaluating {len(samples)} 2D CAD samples...")
        if trials > 1:
            print(f"🗳️  Using majority voting with {trials} trials per sample")
            print(f"📊 Total evaluations: {len(samples) * trials}")
        print(f"📊 Mode: {'Parallel' if parallel else 'Sequential'}")
        
        if trials == 1:
            # Single trial - use existing logic
            return self._evaluate_samples_single_trial(samples, parallel, max_workers, sketches_dir)
        else:
            # Multiple trials - implement majority voting
            return self._evaluate_samples_with_majority_vote(samples, trials, parallel, max_workers, sketches_dir)
    
    def _evaluate_samples_single_trial(self, samples, parallel, max_workers, sketches_dir):
        """Evaluate samples with single trial (original logic)."""
        if parallel:
            print("⚡ Parallel execution enabled - using process-based parallelism for CAD operation isolation")
            print("🔒 Each process will have completely isolated CAD environment to prevent conflicts")
            # Limit workers to avoid overwhelming the system
            if max_workers is None:
                max_workers = min(4, len(samples))  # Conservative default for CAD operations
            elif max_workers > 6:
                print(f"⚠️  Warning: {max_workers} workers may cause resource contention with CAD operations. Consider using ≤6 workers.")
            
            # Prepare data for isolated process execution
            sample_configs = [
                (sample, self.config_path, sketches_dir) 
                for sample in samples
            ]
            
            # Use process-based parallelism for complete isolation
            results = self.loader.process_samples_parallel(
                sample_configs,
                evaluate_single_sample_isolated,  # Use the isolated function
                max_workers=max_workers,
                use_processes=True  # Force process-based execution for CAD safety
            )
        else:
            # Sequential evaluation function
            def eval_func(sample):
                return self.evaluate_sample(sample, sketches_dir=sketches_dir)
            
            results = self.loader.process_samples_sequential(
                samples,
                eval_func
            )
        
        return results
    
    def _evaluate_samples_with_majority_vote(self, samples, trials, parallel, max_workers, sketches_dir):
        """Evaluate samples with multiple trials and majority voting."""
        if parallel:
            print("⚡ Parallel execution enabled - using process-based parallelism for CAD operation isolation")
            print("🔒 Each process will have completely isolated CAD environment to prevent conflicts")
            # Limit workers to avoid overwhelming the system  
            if max_workers is None:
                max_workers = min(4, len(samples) * trials)  # Conservative default for CAD operations
            elif max_workers > 6:
                print(f"⚠️  Warning: {max_workers} workers may cause resource contention with CAD operations. Consider using ≤6 workers.")
            
            # Create multiple trials for each sample
            sample_configs = []
            sample_trial_map = {}  # Map config index to (sample_index, trial_number)
            
            for sample_idx, sample in enumerate(samples):
                for trial in range(trials):
                    config_idx = len(sample_configs)
                    sample_configs.append((sample, self.config_path, sketches_dir))
                    sample_trial_map[config_idx] = (sample_idx, trial)
            
            print(f"🔄 Running {len(sample_configs)} total evaluations ({len(samples)} samples × {trials} trials)")
            
            # Use process-based parallelism for complete isolation
            all_trial_results = self.loader.process_samples_parallel(
                sample_configs,
                evaluate_single_sample_isolated,
                max_workers=max_workers,
                use_processes=True
            )
            
            # Group results by sample and perform majority voting
            sample_results = [[] for _ in range(len(samples))]
            for config_idx, result in enumerate(all_trial_results):
                if result is not None:
                    sample_idx, trial_num = sample_trial_map[config_idx]
                    sample_results[sample_idx].append(result)
            
            # Perform majority voting for each sample
            final_results = []
            for sample_idx, sample in enumerate(samples):
                trial_results = sample_results[sample_idx]
                majority_result = self._perform_majority_vote(trial_results, sample)
                final_results.append(majority_result)
                
                if self.debug and majority_result.trials > 1:
                    print(f"  🗳️  Sample {sample.pid}: {majority_result.vote_counts} → {majority_result.predicted_answer} (confidence: {majority_result.confidence:.1%})")
            
            return final_results
            
        else:
            # Sequential evaluation with multiple trials
            print(f"🔄 Running sequential evaluation with {trials} trials per sample")
            
            final_results = []
            for sample_idx, sample in enumerate(samples):
                if self.debug:
                    print(f"  Processing sample {sample_idx+1}/{len(samples)} (PID: {sample.pid}) with {trials} trials")
                
                # Run multiple trials for this sample
                trial_results = []
                for trial in range(trials):
                    if self.debug:
                        print(f"    Trial {trial+1}/{trials}")
                    result = self.evaluate_sample(sample, sketches_dir=sketches_dir)
                    trial_results.append(result)
                
                # Perform majority voting
                majority_result = self._perform_majority_vote(trial_results, sample)
                final_results.append(majority_result)
                
                if self.debug and majority_result.trials > 1:
                    print(f"    🗳️  Votes: {majority_result.vote_counts} → Final: {majority_result.predicted_answer} (confidence: {majority_result.confidence:.1%})")
            
            return final_results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of metrics
        """
        total_samples = len(results)
        
        # Count correct predictions
        correct_predictions = sum(1 for r in results if r.is_correct)
        
        # Count errors
        error_count = sum(1 for r in results if r.error is not None)
        
        # Count predictions made
        predictions_made = sum(1 for r in results if r.predicted_answer is not None)
        
        # Calculate accuracy
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Calculate accuracy among predictions made
        prediction_accuracy = (correct_predictions / predictions_made 
                             if predictions_made > 0 else 0.0)
        
        # Average execution time
        avg_execution_time = (sum(r.execution_time for r in results) / total_samples 
                            if total_samples > 0 else 0.0)
        
        # Answer distribution
        answer_distribution = {}
        for r in results:
            if r.predicted_answer:
                answer_distribution[r.predicted_answer] = answer_distribution.get(r.predicted_answer, 0) + 1
        
        # Majority voting statistics
        voting_stats = {}
        multi_trial_results = [r for r in results if r.trials > 1]
        if multi_trial_results:
            total_trials = sum(r.trials for r in multi_trial_results)
            avg_confidence = sum(r.confidence for r in multi_trial_results if r.confidence is not None) / len(multi_trial_results)
            
            # Count unanimous vs split decisions
            unanimous_decisions = sum(1 for r in multi_trial_results if r.confidence == 1.0)
            split_decisions = len(multi_trial_results) - unanimous_decisions
            
            voting_stats = {
                'samples_with_voting': len(multi_trial_results),
                'avg_trials_per_sample': total_trials / len(multi_trial_results),
                'avg_confidence': avg_confidence,
                'unanimous_decisions': unanimous_decisions,
                'split_decisions': split_decisions,
                'unanimous_rate': unanimous_decisions / len(multi_trial_results) if multi_trial_results else 0.0
            }
        
        return {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'error_count': error_count,
            'predictions_made': predictions_made,
            'accuracy': accuracy,
            'prediction_accuracy': prediction_accuracy,
            'avg_execution_time': avg_execution_time,
            'answer_distribution': answer_distribution,
            'completion_rate': predictions_made / total_samples if total_samples > 0 else 0.0,
            'voting_stats': voting_stats
        }
    
    def save_results(self, results: List[EvaluationResult], filename: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            filename: Output filename
        """
        # Convert results to dict format
        results_data = []
        for result in results:
            result_dict = {
                'pid': result.pid,
                'predicted_answer': result.predicted_answer,
                'correct_answer': result.correct_answer,
                'is_correct': result.is_correct,
                'reasoning': result.reasoning,
                'execution_time': result.execution_time,
                'error': result.error,
                'assistant_logs': result.assistant_logs
            }
            
            # Add voting fields if this was a multi-trial result
            if result.trials > 1:
                result_dict.update({
                    'trials': result.trials,
                    'all_predictions': result.all_predictions,
                    'vote_counts': result.vote_counts,
                    'confidence': result.confidence
                })
            
            results_data.append(result_dict)
        
        # Add metadata
        output_data = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset': 'sgp-bench',
            'split': 'cad',
            'subject': '2D',
            'metrics': self.calculate_metrics(results),
            'results': results_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved evaluation results to {filename}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary."""
        metrics = self.calculate_metrics(results)
        
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Samples: {metrics['total_samples']:,}")
        print(f"Correct Predictions: {metrics['correct_predictions']:,}")
        print(f"Accuracy: {metrics['accuracy']:.1%}")
        print(f"Completion Rate: {metrics['completion_rate']:.1%}")
        print(f"Prediction Accuracy: {metrics['prediction_accuracy']:.1%}")
        print(f"Errors: {metrics['error_count']:,}")
        print(f"Avg Execution Time: {metrics['avg_execution_time']:.1f}s")
        
        if metrics['answer_distribution']:
            print(f"\nAnswer Distribution:")
            for answer, count in sorted(metrics['answer_distribution'].items()):
                print(f"  {answer}: {count:,} ({count/metrics['predictions_made']:.1%})")
        
        # Display voting statistics if majority voting was used
        if metrics['voting_stats']:
            voting = metrics['voting_stats']
            print(f"\n🗳️  Majority Voting Statistics:")
            print(f"  Samples with voting: {voting['samples_with_voting']:,}")
            print(f"  Avg trials per sample: {voting['avg_trials_per_sample']:.1f}")
            print(f"  Avg confidence: {voting['avg_confidence']:.1%}")
            print(f"  Unanimous decisions: {voting['unanimous_decisions']:,} ({voting['unanimous_rate']:.1%})")
            print(f"  Split decisions: {voting['split_decisions']:,}")
        
        print("="*60)


def main():
    """Main function for CAD evaluation demo."""
    print("🧪 SGP-Bench CAD Evaluation Demo")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        evaluator = SGPCADEvaluator(debug=True)
        
        # Quick test with a few samples
        print("\n🔬 Quick test with 3 2D samples...")
        results = evaluator.evaluate_samples(limit=3, parallel=False, subject="2D")
        
        # Print individual results
        print("\n📋 Individual Results:")
        for result in results:
            status = "✅" if result.is_correct else "❌"
            pred = result.predicted_answer or "None"
            print(f"  {status} PID {result.pid}: {pred} (correct: {result.correct_answer}) - {result.execution_time:.1f}s")
            if result.error:
                print(f"      Error: {result.error}")
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        evaluator.save_results(results, "sgp_cad_evaluation_demo.json")
        
        print(f"\n🎉 Demo completed!")
        print(f"💡 To run full evaluation:")
        print(f"   - Small batch: evaluator.evaluate_samples(limit=50)")
        print(f"   - Full dataset: evaluator.evaluate_samples()")
        print(f"   - Parallel: evaluator.evaluate_samples(limit=20, parallel=True)")
        
    except Exception as e:
        print(f"💥 Demo failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 