## Robot Framework Headless Browsing Suite

This suite uses SeleniumLibrary to run headless browser tests (Chrome or Firefox).

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r robot/requirements.txt
```

Ensure you have a compatible browser installed (Chrome or Firefox). Selenium 4+ will use Selenium Manager to resolve drivers automatically.

### Run (Headless by default)

```bash
robot -d robot/results robot/headless_tests.robot
```

### Override variables

- Browser (chrome|firefox), URL, headless flag:

```bash
robot -d robot/results -v BROWSER:chrome -v HEADLESS:True -v BASE_URL:https://example.com robot/headless_tests.robot
```

### Notes

- Chrome headless uses `--headless=new`.
- For limited shared memory environments (containers/CI), `--no-sandbox` and `--disable-dev-shm-usage` flags are included.
