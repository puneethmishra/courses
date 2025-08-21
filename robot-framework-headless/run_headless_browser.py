#!/usr/bin/env python3
"""
Python script to run Robot Framework headless browser tests programmatically
"""

import os
import sys
import argparse
from datetime import datetime
from robot import run
from robot.api import ExecutionResult, ResultVisitor


class TestResultsAnalyzer(ResultVisitor):
    """Visitor class to analyze test results"""
    
    def __init__(self):
        self.passed_tests = []
        self.failed_tests = []
        self.total_time = 0
        
    def visit_test(self, test):
        """Visit each test case"""
        if test.passed:
            self.passed_tests.append(test.name)
        else:
            self.failed_tests.append(test.name)
        self.total_time += test.elapsedtime


def run_headless_tests(test_file=None, include_tags=None, exclude_tags=None, 
                      output_dir=None, browser='chrome', headless=True):
    """
    Run Robot Framework tests with headless browser configuration
    
    Args:
        test_file: Path to test file or directory
        include_tags: Tags to include
        exclude_tags: Tags to exclude
        output_dir: Output directory for results
        browser: Browser to use (chrome/firefox)
        headless: Run in headless mode
    
    Returns:
        int: Return code (0 for success, non-zero for failure)
    """
    
    # Default test file
    if not test_file:
        test_file = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Create output directory with timestamp
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 
                                 f'results/python_run_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build arguments for robot run
    args = [
        '--outputdir', output_dir,
        '--variable', f'BROWSER:{"headless" + browser if headless else browser}',
    ]
    
    if include_tags:
        args.extend(['--include', include_tags])
    
    if exclude_tags:
        args.extend(['--exclude', exclude_tags])
    
    # Add test file/directory
    args.append(test_file)
    
    print(f"Running Robot Framework tests...")
    print(f"Test source: {test_file}")
    print(f"Output directory: {output_dir}")
    print(f"Browser: {'headless ' + browser if headless else browser}")
    
    # Run tests
    return_code = run(*args)
    
    # Analyze results
    result_file = os.path.join(output_dir, 'output.xml')
    if os.path.exists(result_file):
        result = ExecutionResult(result_file)
        analyzer = TestResultsAnalyzer()
        result.visit(analyzer)
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total tests: {len(analyzer.passed_tests) + len(analyzer.failed_tests)}")
        print(f"Passed: {len(analyzer.passed_tests)}")
        print(f"Failed: {len(analyzer.failed_tests)}")
        print(f"Total execution time: {analyzer.total_time/1000:.2f} seconds")
        
        if analyzer.passed_tests:
            print("\nPassed tests:")
            for test in analyzer.passed_tests:
                print(f"  ✓ {test}")
        
        if analyzer.failed_tests:
            print("\nFailed tests:")
            for test in analyzer.failed_tests:
                print(f"  ✗ {test}")
        
        print("\nDetailed reports:")
        print(f"  - Report: {os.path.join(output_dir, 'report.html')}")
        print(f"  - Log: {os.path.join(output_dir, 'log.html')}")
    
    return return_code


def run_specific_test_cases():
    """Example of running specific test cases programmatically"""
    
    print("Running specific test cases...\n")
    
    # Example 1: Run only Chrome tests
    print("1. Running Chrome tests only:")
    run_headless_tests(
        test_file='tests/headless_browser_tests.robot',
        include_tags='chrome',
        browser='chrome'
    )
    
    # Example 2: Run scraping tests
    print("\n2. Running web scraping tests:")
    run_headless_tests(
        test_file='tests/headless_browser_tests.robot',
        include_tags='scraping',
        browser='chrome'
    )
    
    # Example 3: Run all tests except slow ones
    print("\n3. Running all tests except slow ones:")
    run_headless_tests(
        test_file='tests/headless_browser_tests.robot',
        exclude_tags='slow',
        browser='firefox'
    )


def create_custom_test():
    """Create and run a custom test dynamically"""
    
    custom_test = """*** Settings ***
Library    SeleniumLibrary

*** Variables ***
${URL}    https://www.python.org

*** Test Cases ***
Dynamic Python.org Test
    [Documentation]    Dynamically created test
    ${options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    Call Method    ${options}    add_argument    --headless
    Call Method    ${options}    add_argument    --no-sandbox
    Create Webdriver    Chrome    options=${options}
    Go To    ${URL}
    Title Should Be    Welcome to Python.org
    Page Should Contain    Python
    Capture Page Screenshot    dynamic_test.png
    [Teardown]    Close Browser
"""
    
    # Write custom test to file
    custom_test_file = 'tests/dynamic_test.robot'
    with open(custom_test_file, 'w') as f:
        f.write(custom_test)
    
    print("Running dynamically created test...")
    result = run_headless_tests(test_file=custom_test_file)
    
    # Clean up
    os.remove(custom_test_file)
    
    return result


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Run Robot Framework headless browser tests'
    )
    parser.add_argument(
        'test_file',
        nargs='?',
        default='tests/headless_browser_tests.robot',
        help='Test file or directory to run'
    )
    parser.add_argument(
        '-i', '--include',
        help='Include tests with these tags'
    )
    parser.add_argument(
        '-e', '--exclude',
        help='Exclude tests with these tags'
    )
    parser.add_argument(
        '-d', '--outputdir',
        help='Output directory for results'
    )
    parser.add_argument(
        '-b', '--browser',
        choices=['chrome', 'firefox'],
        default='chrome',
        help='Browser to use'
    )
    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Run browser in normal (non-headless) mode'
    )
    parser.add_argument(
        '--examples',
        action='store_true',
        help='Run example test cases'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Create and run a dynamic test'
    )
    
    args = parser.parse_args()
    
    if args.examples:
        run_specific_test_cases()
    elif args.dynamic:
        create_custom_test()
    else:
        return_code = run_headless_tests(
            test_file=args.test_file,
            include_tags=args.include,
            exclude_tags=args.exclude,
            output_dir=args.outputdir,
            browser=args.browser,
            headless=not args.no_headless
        )
        
        sys.exit(return_code)


if __name__ == '__main__':
    main()