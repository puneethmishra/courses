*** Settings ***
Library  SeleniumLibrary

*** Variables ***
${BROWSER}    chrome
${HEADLESS}   True
${BASE_URL}   https://example.com

*** Keywords ***
Open Browser In Headless Mode
    IF    '${BROWSER}'=='chrome'
        ${options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys, selenium.webdriver
        IF    ${HEADLESS}
            Call Method    ${options}    add_argument    --headless=new
        END
        Call Method    ${options}    add_argument    --no-sandbox
        Call Method    ${options}    add_argument    --disable-dev-shm-usage
        Create Webdriver    Chrome    options=${options}
    ELSE IF    '${BROWSER}'=='firefox'
        ${options}=    Evaluate    sys.modules['selenium.webdriver'].FirefoxOptions()    sys, selenium.webdriver
        IF    ${HEADLESS}
            Call Method    ${options}    add_argument    -headless
        END
        Create Webdriver    Firefox    options=${options}
    ELSE
        Fail    Unsupported browser: ${BROWSER}
    END
    Set Selenium Implicit Wait    5s

Go To Base Url
    Go To    ${BASE_URL}

Close Browser Session
    Close All Browsers

*** Test Cases ***
Example Page Loads Headless
    Open Browser In Headless Mode
    Go To Base Url
    Wait Until Page Contains    Example Domain    10s
    [Teardown]    Close Browser Session

