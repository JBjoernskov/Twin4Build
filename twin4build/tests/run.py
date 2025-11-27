#!/usr/bin/env python
"""
Script to run unittest discovery with coverage measurement.
Equivalent to: coverage run -m unittest discover
"""

import sys
import os
import unittest
import webbrowser

try:
    import coverage
except ImportError:
    print("Error: coverage module not installed.")
    print("Install it with: pip install coverage")
    sys.exit(1)


def main():
    # Initialize coverage
    cov = coverage.Coverage()
    
    # Start measuring coverage
    cov.start()
    
    try:
        # Get the directory containing the tests
        test_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a test loader and discover tests
        loader = unittest.TestLoader()
        
        # Discover all tests in the current directory and subdirectories
        # Start directory: current test directory
        # Pattern: test*.py (default pattern for unittest discovery)

        tests = loader.discover(start_dir=test_dir, pattern='test_*.py')
        # tests = loader.discover(start_dir=test_dir, pattern='test_simulation_model.py')
        # tests = loader.discover(start_dir=test_dir, pattern='test_components.py')
        # tests = loader.discover(start_dir=test_dir, pattern='test_translator.py')
        # tests = loader.discover(start_dir=test_dir, pattern='test_optimizer.py')
        # tests = loader.discover(start_dir=test_dir, pattern='test_semantic_model.py')
        tests = loader.discover(start_dir=test_dir, pattern='test_types.py')

        
        # Create a test runner
        runner = unittest.TextTestRunner(verbosity=2)
        
        # Run the tests
        result = runner.run(tests)


        cov.stop()


        if True:
            
            # Stop measuring coverage
            
            
            # Save coverage data
            cov.save()
            
            print("\n" + "="*70)
            print("Generating coverage reports...")
            print("="*70)
            
            # Generate text report to console
            print("\nCoverage Summary:")
            print("-"*70)
            cov.report()
            
            # Generate HTML report
            print("\n" + "="*70)
            print("Generating HTML coverage report...")
            
            # html_report() returns the coverage percentage, not the directory
            coverage_percent = cov.html_report()
            
            # The HTML report is generated in the 'htmlcov' directory by default
            html_dir = 'htmlcov'
            index_path = os.path.join(html_dir, 'index.html')
            abs_index_path = os.path.abspath(index_path)
            
            print(f"HTML report generated in: {os.path.abspath(html_dir)}")
            print(f"Coverage: {coverage_percent:.1f}%")
            print(f"Opening report in browser...")
            print("="*70)
            
            # Open the HTML report in the default browser
            webbrowser.open(f'file://{abs_index_path}')
            
        # Exit with appropriate code based on test results
        if result.wasSuccessful():
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        # Stop coverage even if there's an error
        cov.stop()
        cov.save()
        print(f"\nError occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

