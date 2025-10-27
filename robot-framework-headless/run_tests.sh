#!/bin/bash

# Robot Framework Headless Browser Test Runner

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TEST_FILE="tests/headless_browser_tests.robot"
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
LOG_LEVEL="INFO"
TAGS=""
EXCLUDE_TAGS=""
TEST_NAME=""

# Function to display help
show_help() {
    echo "Robot Framework Headless Browser Test Runner"
    echo ""
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -t, --test NAME     Run specific test by name"
    echo "  -i, --include TAGS  Include tests with specific tags"
    echo "  -e, --exclude TAGS  Exclude tests with specific tags"
    echo "  -f, --file FILE     Specify test file (default: tests/headless_browser_tests.robot)"
    echo "  -d, --dir DIR       Output directory (default: results/timestamp)"
    echo "  -l, --log LEVEL     Log level (TRACE, DEBUG, INFO, WARN)"
    echo "  -s, --simple        Run simple example test"
    echo "  --setup             Install Python dependencies"
    echo ""
    echo "Examples:"
    echo "  ./run_tests.sh                    # Run all tests"
    echo "  ./run_tests.sh -t \"Test Headless Chrome Browser\""
    echo "  ./run_tests.sh -i chrome -e slow"
    echo "  ./run_tests.sh -s                 # Run simple example"
    echo "  ./run_tests.sh --setup            # Install dependencies"
}

# Function to install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${RED}Failed to install dependencies!${NC}"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--test)
            TEST_NAME="$2"
            shift 2
            ;;
        -i|--include)
            TAGS="$2"
            shift 2
            ;;
        -e|--exclude)
            EXCLUDE_TAGS="$2"
            shift 2
            ;;
        -f|--file)
            TEST_FILE="$2"
            shift 2
            ;;
        -d|--dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--log)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -s|--simple)
            TEST_FILE="tests/simple_example.robot"
            shift
            ;;
        --setup)
            install_dependencies
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if Robot Framework is installed
if ! command -v robot &> /dev/null; then
    echo -e "${RED}Robot Framework is not installed!${NC}"
    echo "Run './run_tests.sh --setup' to install dependencies"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build robot command
ROBOT_CMD="robot"
ROBOT_CMD="$ROBOT_CMD -d $OUTPUT_DIR"
ROBOT_CMD="$ROBOT_CMD -L $LOG_LEVEL"

if [ -n "$TEST_NAME" ]; then
    ROBOT_CMD="$ROBOT_CMD -t \"$TEST_NAME\""
fi

if [ -n "$TAGS" ]; then
    ROBOT_CMD="$ROBOT_CMD -i $TAGS"
fi

if [ -n "$EXCLUDE_TAGS" ]; then
    ROBOT_CMD="$ROBOT_CMD -e $EXCLUDE_TAGS"
fi

ROBOT_CMD="$ROBOT_CMD $TEST_FILE"

# Display test information
echo -e "${YELLOW}Running Robot Framework Tests${NC}"
echo "Test file: $TEST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Log level: $LOG_LEVEL"
if [ -n "$TEST_NAME" ]; then
    echo "Test name: $TEST_NAME"
fi
if [ -n "$TAGS" ]; then
    echo "Include tags: $TAGS"
fi
if [ -n "$EXCLUDE_TAGS" ]; then
    echo "Exclude tags: $EXCLUDE_TAGS"
fi
echo ""

# Run tests
echo -e "${YELLOW}Executing: $ROBOT_CMD${NC}"
echo ""
eval $ROBOT_CMD

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed successfully!${NC}"
    echo "Results available in: $OUTPUT_DIR"
    echo "- Report: $OUTPUT_DIR/report.html"
    echo "- Log: $OUTPUT_DIR/log.html"
else
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    echo "Check the detailed report at: $OUTPUT_DIR/report.html"
fi