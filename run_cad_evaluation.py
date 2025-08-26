#!/usr/bin/env python3
"""
CAD Evaluation Runner
====================

Simple script to run 2D CAD evaluations on SGP-bench CAD split.
Evaluates 2D samples using pre-generated FreeCAD sketch files.
"""

import argparse
import os
import json
from sgp_cad_evaluator import SGPCADEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run 2D CAD evaluation on SGP-bench CAD split")
    
    # Basic options
    parser.add_argument('--limit', type=int, default=5, 
                       help='Number of samples to evaluate (default: 5 for quick test)')
    parser.add_argument('--subject', type=str, default='2D', choices=['2D'],
                       help='Subject type to evaluate (2D only for now)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to OpenAI config file (default: config.json)')
    parser.add_argument('--output', type=str, default='cad_evaluation_results.json',
                       help='Output file for results (default: cad_evaluation_results.json)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--trials', type=int, default=1,
                       help='Number of trials per sample for majority voting (default: 1, recommend: 3 or 5)')
    
    # Sample selection
    parser.add_argument('--indices', type=str, default=None,
                       help='Specific sample indices to evaluate (comma-separated, e.g., "0,1,2")')
    
    # Sketches directory
    parser.add_argument('--sketches-dir', type=str, default='sgp_bench_samples',
                       help='Directory containing FreeCAD project directories (default: sgp_bench_samples)')
    
    args = parser.parse_args()
    
    # Validate trials argument
    if args.trials < 1:
        print("❌ Error: --trials must be at least 1")
        return 1
    elif args.trials > 1:
        print(f"🗳️  Majority voting enabled: {args.trials} trials per sample")
        if args.trials % 2 == 0:
            print("⚠️  Warning: Even number of trials may result in ties")
    
    # Show subject selection and validate
    print(f"🎯 Subject: {args.subject}")
    
    # Validate subject (should always be 2D now)
    if args.subject != '2D':
        print(f"❌ Invalid subject: {args.subject}")
        print("💡 Only 2D evaluation is supported in this version")
        return 1
    
    # Check config file
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("Create config.json with your OpenAI API key:")
        print("""
{
  "openai_api_key": "your-api-key-here",
  "openai_model": "gpt-4o-mini"
}
""")
        return 1
    
    # Check sketches directory
    if not os.path.exists(args.sketches_dir):
        print(f"❌ Sketches directory not found: {args.sketches_dir}")
        print(f"💡 The {args.sketches_dir} directory should be included in the release. Check your installation.")
        return 1
    
    try:
        # Initialize evaluator
        print(f"🔧 Initializing {args.subject} CAD evaluator...")
        evaluator = SGPCADEvaluator(config_path=args.config, debug=args.debug)
        
        # Determine sample indices
        sample_indices = None
        if args.indices:
            sample_indices = [int(x.strip()) for x in args.indices.split(',')]
            print(f"📋 Evaluating specific indices: {sample_indices}")
        
        print(f"📁 Using {args.subject} sample directories from: {args.sketches_dir}")
        
        # Run evaluation
        print(f"🎯 Starting {args.subject} evaluation...")
        
        results = evaluator.evaluate_samples(
            sample_indices=sample_indices,
            limit=args.limit,
            parallel=args.parallel,
            max_workers=args.workers,
            sketches_dir=args.sketches_dir,
            trials=args.trials,
            subject=args.subject
        )
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        evaluator.save_results(results, args.output)
        
        print(f"\n✅ Evaluation completed!")
        print(f"📊 Results saved to: {args.output}")
        
        # Show some example results
        print(f"\n📋 Example Results:")
        for i, result in enumerate(results[:3]):
            status = "✅" if result.is_correct else "❌"
            pred = result.predicted_answer or "None"
            
            if result.trials > 1:
                # Show voting information
                confidence_str = f" (confidence: {result.confidence:.1%})" if result.confidence else ""
                votes_str = f" {result.vote_counts}" if result.vote_counts else ""
                print(f"  {status} Sample {i+1}: Predicted {pred}, Correct {result.correct_answer}{votes_str}{confidence_str}")
            else:
                print(f"  {status} Sample {i+1}: Predicted {pred}, Correct {result.correct_answer}")
            
    except Exception as e:
        print(f"💥 Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 