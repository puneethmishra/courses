*** Settings ***
Documentation     Headless Browser Testing with Robot Framework
...               This test suite demonstrates various headless browser capabilities
...               including Chrome and Firefox headless modes, page interactions,
...               screenshots, and web scraping.

Library           SeleniumLibrary
Library           OperatingSystem
Library           Collections
Library           String
Resource          ../resources/browser_keywords.robot

Suite Setup       Setup Test Environment
Suite Teardown    Close All Browsers
Test Setup        Open Headless Browser
Test Teardown     Capture Page Screenshot

*** Variables ***
${BROWSER}              headlesschrome
${TIMEOUT}              10
${SCREENSHOT_DIR}       ${CURDIR}/../results/screenshots
${TEST_URL}             https://www.example.com
${SEARCH_URL}           https://www.google.com

*** Test Cases ***
Test Headless Chrome Browser
    [Documentation]    Test basic headless Chrome functionality
    [Tags]    chrome    headless
    Go To    ${TEST_URL}
    Title Should Be    Example Domain
    Page Should Contain    Example Domain
    ${page_source}=    Get Source
    Should Contain    ${page_source}    This domain is for use in illustrative examples

Test Headless Firefox Browser
    [Documentation]    Test headless Firefox functionality
    [Tags]    firefox    headless
    [Setup]    Open Headless Firefox
    Go To    ${TEST_URL}
    Title Should Be    Example Domain
    Element Should Be Visible    xpath://h1
    ${h1_text}=    Get Text    xpath://h1
    Should Be Equal    ${h1_text}    Example Domain

Test Page Navigation And Screenshots
    [Documentation]    Test navigation between pages and capture screenshots
    [Tags]    navigation    screenshot
    Go To    ${SEARCH_URL}
    Wait Until Page Contains Element    name:q    timeout=${TIMEOUT}
    Input Text    name:q    Robot Framework headless browser
    Capture Element Screenshot    name:q    search_input.png
    Press Keys    name:q    RETURN
    Wait Until Page Contains    results    timeout=${TIMEOUT}
    Capture Full Page Screenshot

Test Form Interaction In Headless Mode
    [Documentation]    Test form filling and submission in headless mode
    [Tags]    forms    interaction
    Go To    https://httpbin.org/forms/post
    Wait Until Element Is Visible    name:custname    timeout=${TIMEOUT}
    Input Text    name:custname    Robot Framework User
    Input Text    name:custtel    +1234567890
    Input Text    name:custemail    robot@framework.test
    Select From List By Value    name:size    large
    Select Checkbox    name:topping    value=cheese
    Select Checkbox    name:topping    value=mushroom
    Input Text    name:comments    Testing headless form submission
    Submit Form    xpath://form
    Wait Until Page Contains    Robot Framework User    timeout=${TIMEOUT}

Test JavaScript Execution In Headless Browser
    [Documentation]    Test JavaScript execution capabilities
    [Tags]    javascript    headless
    Go To    ${TEST_URL}
    ${result}=    Execute Javascript    return document.title;
    Should Be Equal    ${result}    Example Domain
    Execute Javascript    document.body.style.backgroundColor = 'lightblue';
    ${bg_color}=    Execute Javascript    return window.getComputedStyle(document.body).backgroundColor;
    Log    Background color changed to: ${bg_color}

Test Web Scraping With Headless Browser
    [Documentation]    Demonstrate web scraping capabilities
    [Tags]    scraping    data
    Go To    https://quotes.toscrape.com/
    ${quotes}=    Get WebElements    xpath://div[@class='quote']
    ${quote_count}=    Get Length    ${quotes}
    Should Be True    ${quote_count} > 0
    
    ${first_quote_text}=    Get Text    xpath:(//span[@class='text'])[1]
    ${first_quote_author}=    Get Text    xpath:(//small[@class='author'])[1]
    Log    First Quote: ${first_quote_text} - ${first_quote_author}
    
    @{all_quotes}=    Create List
    FOR    ${index}    IN RANGE    1    4
        ${quote}=    Get Text    xpath:(//span[@class='text'])[${index}]
        ${author}=    Get Text    xpath:(//small[@class='author'])[${index}]
        ${quote_dict}=    Create Dictionary    quote=${quote}    author=${author}
        Append To List    ${all_quotes}    ${quote_dict}
    END
    Log Many    @{all_quotes}

Test Responsive Design In Different Viewports
    [Documentation]    Test different viewport sizes in headless mode
    [Tags]    responsive    viewport
    # Mobile viewport
    Set Window Size    375    667
    Go To    ${TEST_URL}
    Capture Page Screenshot    mobile_view.png
    
    # Tablet viewport
    Set Window Size    768    1024
    Capture Page Screenshot    tablet_view.png
    
    # Desktop viewport
    Set Window Size    1920    1080
    Capture Page Screenshot    desktop_view.png

Test Wait Conditions In Headless Mode
    [Documentation]    Test various wait conditions
    [Tags]    wait    conditions
    Go To    https://httpbin.org/delay/2
    Wait Until Page Contains    origin    timeout=10
    ${response_text}=    Get Text    xpath://pre
    Should Contain    ${response_text}    origin
    Should Contain    ${response_text}    url

Test Cookie Management In Headless Browser
    [Documentation]    Test cookie operations in headless mode
    [Tags]    cookies    session
    Go To    ${TEST_URL}
    Add Cookie    test_cookie    test_value
    Add Cookie    session_id    abc123    path=/
    ${cookies}=    Get Cookies
    Log    All cookies: ${cookies}
    ${cookie_value}=    Get Cookie    test_cookie
    Should Be Equal    ${cookie_value.value}    test_value
    Delete Cookie    test_cookie
    Delete All Cookies

Test Network Performance Monitoring
    [Documentation]    Monitor page load performance
    [Tags]    performance    metrics
    ${start_time}=    Get Time    epoch
    Go To    ${TEST_URL}
    ${end_time}=    Get Time    epoch
    ${load_time}=    Evaluate    ${end_time} - ${start_time}
    Log    Page load time: ${load_time} seconds
    Should Be True    ${load_time} < 5    Page should load within 5 seconds

*** Keywords ***
Setup Test Environment
    [Documentation]    Setup the test environment
    Create Directory    ${SCREENSHOT_DIR}
    Set Screenshot Directory    ${SCREENSHOT_DIR}
    Log    Test environment setup completed

Open Headless Browser
    [Documentation]    Open browser in headless mode with Chrome
    ${chrome_options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    Call Method    ${chrome_options}    add_argument    --headless
    Call Method    ${chrome_options}    add_argument    --no-sandbox
    Call Method    ${chrome_options}    add_argument    --disable-dev-shm-usage
    Call Method    ${chrome_options}    add_argument    --disable-gpu
    Call Method    ${chrome_options}    add_argument    --window-size=1920,1080
    Create Webdriver    Chrome    options=${chrome_options}
    Set Selenium Timeout    ${TIMEOUT}

Open Headless Firefox
    [Documentation]    Open Firefox in headless mode
    ${firefox_options}=    Evaluate    sys.modules['selenium.webdriver'].FirefoxOptions()    sys
    Call Method    ${firefox_options}    add_argument    --headless
    Call Method    ${firefox_options}    add_argument    --width=1920
    Call Method    ${firefox_options}    add_argument    --height=1080
    Create Webdriver    Firefox    options=${firefox_options}
    Set Selenium Timeout    ${TIMEOUT}

Capture Full Page Screenshot
    [Documentation]    Capture screenshot with timestamp
    ${timestamp}=    Get Time    epoch
    ${screenshot_name}=    Set Variable    fullpage_${timestamp}.png
    Capture Page Screenshot    ${screenshot_name}