*** Settings ***
Documentation     Simple example of headless browser testing
Library           SeleniumLibrary

*** Variables ***
${URL}            https://www.google.com
${BROWSER}        headlesschrome

*** Test Cases ***
Simple Google Search Test
    [Documentation]    A simple test that searches Google in headless mode
    Open Browser In Headless Mode
    Go To    ${URL}
    Wait Until Page Contains Element    name:q    timeout=10
    Input Text    name:q    Robot Framework
    Press Keys    name:q    RETURN
    Wait Until Page Contains    results    timeout=10
    Page Should Contain    Robot Framework
    Capture Page Screenshot    google_search_result.png
    [Teardown]    Close Browser

*** Keywords ***
Open Browser In Headless Mode
    [Documentation]    Opens Chrome browser in headless mode
    ${chrome_options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    Call Method    ${chrome_options}    add_argument    --headless
    Call Method    ${chrome_options}    add_argument    --no-sandbox
    Call Method    ${chrome_options}    add_argument    --disable-dev-shm-usage
    Create Webdriver    Chrome    options=${chrome_options}