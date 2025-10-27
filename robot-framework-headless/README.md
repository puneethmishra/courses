# Robot Framework Headless Browser Testing

This project demonstrates headless browser testing using Robot Framework with Selenium WebDriver. It includes comprehensive test cases for various web automation scenarios running in headless mode.

## Features

- ✅ Headless Chrome and Firefox browser support
- ✅ Web scraping capabilities
- ✅ Form interaction and submission
- ✅ JavaScript execution
- ✅ Screenshot capture
- ✅ Cookie management
- ✅ Performance monitoring
- ✅ Responsive design testing
- ✅ File download handling
- ✅ Custom wait conditions

## Project Structure

```
robot-framework-headless/
├── tests/
│   └── headless_browser_tests.robot    # Main test suite
├── resources/
│   └── browser_keywords.robot          # Reusable keywords
├── results/
│   └── screenshots/                    # Screenshot storage
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Prerequisites

- Python 3.8 or higher
- Google Chrome (for Chrome tests)
- Mozilla Firefox (for Firefox tests)

## Installation

1. Clone or navigate to the project directory:
```bash
cd /workspace/robot-framework-headless
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install browser drivers:
```bash
# The webdriver-manager will automatically download drivers when tests run
# Or manually install:
webdriver-manager update --chrome
webdriver-manager update --firefox
```

## Running Tests

### Run all tests:
```bash
robot tests/headless_browser_tests.robot
```

### Run specific test by name:
```bash
robot -t "Test Headless Chrome Browser" tests/headless_browser_tests.robot
```

### Run tests with specific tags:
```bash
# Run only Chrome tests
robot -i chrome tests/headless_browser_tests.robot

# Run scraping tests
robot -i scraping tests/headless_browser_tests.robot

# Exclude specific tags
robot -e firefox tests/headless_browser_tests.robot
```

### Run with custom output directory:
```bash
robot -d results/test_run_$(date +%Y%m%d_%H%M%S) tests/headless_browser_tests.robot
```

### Run with different log levels:
```bash
robot -L DEBUG tests/headless_browser_tests.robot
```

## Test Cases Overview

### 1. Basic Headless Browser Tests
- **Test Headless Chrome Browser**: Verifies Chrome headless functionality
- **Test Headless Firefox Browser**: Verifies Firefox headless functionality

### 2. Interaction Tests
- **Test Page Navigation And Screenshots**: Navigation and screenshot capture
- **Test Form Interaction In Headless Mode**: Form filling and submission
- **Test JavaScript Execution**: Execute JavaScript in headless mode

### 3. Advanced Features
- **Test Web Scraping**: Extract data from web pages
- **Test Responsive Design**: Test different viewport sizes
- **Test Wait Conditions**: Various wait strategies
- **Test Cookie Management**: Cookie operations
- **Test Network Performance**: Monitor page load times

## Configuration

### Browser Options

The tests use predefined browser options for headless mode. You can customize them in the resource file:

```robot
# Chrome options
--headless
--no-sandbox
--disable-dev-shm-usage
--disable-gpu
--window-size=1920,1080

# Firefox options
--headless
--width=1920
--height=1080
```

### Timeout Settings

Modify timeout values in the Variables section:
```robot
${TIMEOUT}              10    # Default timeout in seconds
${IMPLICIT_WAIT}        5     # Implicit wait time
```

## Custom Keywords

The project includes reusable keywords in `resources/browser_keywords.robot`:

- `Open Headless Chrome With Custom Options`: Open Chrome with custom arguments
- `Open Headless Firefox With Custom Options`: Open Firefox with custom arguments
- `Wait For Element And Click`: Smart click with wait
- `Take Screenshot With Timestamp`: Screenshot with timestamp
- `Check Page Load Performance`: Get page load metrics
- `Extract Table Data`: Extract data from HTML tables
- `Download File In Headless Mode`: Configure downloads in headless mode

## Debugging Tips

1. **Remove headless mode for debugging**:
   - Comment out the `--headless` argument in browser options
   
2. **Enable debug logging**:
   ```bash
   robot -L DEBUG -b debug.log tests/headless_browser_tests.robot
   ```

3. **Take screenshots on failure**:
   - Already configured in Test Teardown

4. **Check browser console logs**:
   ```robot
   ${logs}=    Get Browser Logs
   Log Many    @{logs}
   ```

## CI/CD Integration

### GitHub Actions Example:
```yaml
- name: Run Robot Framework Tests
  run: |
    pip install -r requirements.txt
    robot -d results tests/headless_browser_tests.robot
    
- name: Upload test results
  uses: actions/upload-artifact@v2
  if: always()
  with:
    name: test-results
    path: results/
```

### Jenkins Example:
```groovy
stage('Run Tests') {
    steps {
        sh 'robot -d results tests/headless_browser_tests.robot'
    }
    post {
        always {
            robot outputPath: 'results',
                  outputFileName: 'output.xml',
                  reportFileName: 'report.html',
                  logFileName: 'log.html'
        }
    }
}
```

## Troubleshooting

### Common Issues:

1. **ChromeDriver not found**:
   ```bash
   webdriver-manager update --chrome
   ```

2. **Timeout errors**:
   - Increase timeout values in test variables
   - Check network connectivity

3. **Element not found**:
   - Verify selectors are correct
   - Add explicit waits before interacting with elements

4. **JavaScript errors**:
   - Check browser console logs
   - Ensure JavaScript is enabled

## Best Practices

1. **Use Page Object Model** for complex applications
2. **Implement proper wait strategies** instead of sleep
3. **Take screenshots** at critical points
4. **Use descriptive test and keyword names**
5. **Handle exceptions gracefully**
6. **Clean up resources** in teardown

## Contributing

1. Follow Robot Framework style guide
2. Add documentation to all keywords
3. Write meaningful test descriptions
4. Include tags for test categorization

## License

This project is open source and available under the MIT License.

## Additional Resources

- [Robot Framework Documentation](https://robotframework.org/)
- [SeleniumLibrary Documentation](https://robotframework.org/SeleniumLibrary/)
- [Selenium Documentation](https://www.selenium.dev/documentation/)
- [Robot Framework User Guide](https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html)