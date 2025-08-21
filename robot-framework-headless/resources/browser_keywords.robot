*** Settings ***
Documentation     Common browser keywords and utilities for headless testing
Library           SeleniumLibrary
Library           DateTime
Library           OperatingSystem

*** Variables ***
${DEFAULT_TIMEOUT}      10
${IMPLICIT_WAIT}        5
${POLL_FREQUENCY}       0.5

*** Keywords ***
Open Headless Chrome With Custom Options
    [Documentation]    Open Chrome with custom headless options
    [Arguments]    ${additional_args}=@{EMPTY}
    ${options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    Call Method    ${options}    add_argument    --headless
    Call Method    ${options}    add_argument    --no-sandbox
    Call Method    ${options}    add_argument    --disable-dev-shm-usage
    Call Method    ${options}    add_argument    --disable-gpu
    Call Method    ${options}    add_argument    --disable-web-security
    Call Method    ${options}    add_argument    --disable-features=VizDisplayCompositor
    Call Method    ${options}    add_argument    --window-size=1920,1080
    
    # Add user agent
    Call Method    ${options}    add_argument    --user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36
    
    # Add additional arguments if provided
    FOR    ${arg}    IN    @{additional_args}
        Call Method    ${options}    add_argument    ${arg}
    END
    
    # Add experimental options for better performance
    ${prefs}=    Create Dictionary    
    ...    profile.default_content_setting_values.notifications=2
    ...    profile.default_content_settings.popups=0
    ...    profile.managed_default_content_settings.images=2
    Call Method    ${options}    add_experimental_option    prefs    ${prefs}
    
    Create Webdriver    Chrome    options=${options}
    Set Selenium Implicit Wait    ${IMPLICIT_WAIT}
    Set Selenium Timeout    ${DEFAULT_TIMEOUT}

Open Headless Firefox With Custom Options
    [Documentation]    Open Firefox with custom headless options
    [Arguments]    ${additional_args}=@{EMPTY}
    ${options}=    Evaluate    sys.modules['selenium.webdriver'].FirefoxOptions()    sys
    Call Method    ${options}    add_argument    --headless
    Call Method    ${options}    add_argument    --width=1920
    Call Method    ${options}    add_argument    --height=1080
    
    # Add additional arguments if provided
    FOR    ${arg}    IN    @{additional_args}
        Call Method    ${options}    add_argument    ${arg}
    END
    
    # Set Firefox preferences
    Call Method    ${options}    set_preference    browser.download.folderList    2
    Call Method    ${options}    set_preference    browser.download.manager.showWhenStarting    False
    Call Method    ${options}    set_preference    browser.helperApps.neverAsk.saveToDisk    application/pdf,text/csv,application/csv
    
    Create Webdriver    Firefox    options=${options}
    Set Selenium Implicit Wait    ${IMPLICIT_WAIT}
    Set Selenium Timeout    ${DEFAULT_TIMEOUT}

Open Headless Browser With Proxy
    [Documentation]    Open headless browser with proxy settings
    [Arguments]    ${browser}=chrome    ${proxy_server}=${EMPTY}
    ${options}=    Run Keyword If    '${browser.lower()}' == 'chrome'
    ...    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    ...    ELSE    Evaluate    sys.modules['selenium.webdriver'].FirefoxOptions()    sys
    
    Call Method    ${options}    add_argument    --headless
    Call Method    ${options}    add_argument    --no-sandbox
    
    Run Keyword If    '${proxy_server}' != '${EMPTY}'
    ...    Call Method    ${options}    add_argument    --proxy-server=${proxy_server}
    
    Create Webdriver    ${browser}    options=${options}

Wait For Element And Click
    [Documentation]    Wait for element to be clickable and click it
    [Arguments]    ${locator}    ${timeout}=${DEFAULT_TIMEOUT}
    Wait Until Element Is Visible    ${locator}    timeout=${timeout}
    Wait Until Element Is Enabled    ${locator}    timeout=${timeout}
    Scroll Element Into View    ${locator}
    Click Element    ${locator}

Wait For Element And Input Text
    [Documentation]    Wait for element and input text
    [Arguments]    ${locator}    ${text}    ${clear}=${TRUE}    ${timeout}=${DEFAULT_TIMEOUT}
    Wait Until Element Is Visible    ${locator}    timeout=${timeout}
    Wait Until Element Is Enabled    ${locator}    timeout=${timeout}
    Run Keyword If    ${clear}    Clear Element Text    ${locator}
    Input Text    ${locator}    ${text}

Scroll To Element
    [Documentation]    Scroll to make element visible
    [Arguments]    ${locator}
    Execute Javascript    
    ...    var element = document.evaluate('${locator}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    ...    if(element) element.scrollIntoView({behavior: 'smooth', block: 'center'});

Take Screenshot With Timestamp
    [Documentation]    Take screenshot with timestamp in filename
    [Arguments]    ${prefix}=screenshot
    ${timestamp}=    Get Current Date    result_format=%Y%m%d_%H%M%S
    ${filename}=    Set Variable    ${prefix}_${timestamp}.png
    Capture Page Screenshot    ${filename}
    [Return]    ${filename}

Get Element Attribute Value
    [Documentation]    Get attribute value of an element
    [Arguments]    ${locator}    ${attribute}
    ${value}=    Get Element Attribute    ${locator}    ${attribute}
    [Return]    ${value}

Wait For Ajax Requests To Complete
    [Documentation]    Wait for all jQuery AJAX requests to complete
    [Arguments]    ${timeout}=${DEFAULT_TIMEOUT}
    Wait For Condition    
    ...    return (typeof jQuery !== 'undefined' && jQuery.active === 0) || (typeof jQuery === 'undefined')
    ...    timeout=${timeout}

Execute Async JavaScript
    [Documentation]    Execute asynchronous JavaScript code
    [Arguments]    ${script}
    ${result}=    Execute Async Javascript    ${script}
    [Return]    ${result}

Get All Links From Page
    [Documentation]    Get all links from the current page
    ${links}=    Get WebElements    xpath://a[@href]
    @{link_urls}=    Create List
    FOR    ${link}    IN    @{links}
        ${url}=    Get Element Attribute    ${link}    href
        Append To List    ${link_urls}    ${url}
    END
    [Return]    @{link_urls}

Check Page Load Performance
    [Documentation]    Check page load performance metrics
    ${performance}=    Execute Javascript
    ...    return JSON.stringify(window.performance.timing);
    ${timing}=    Evaluate    json.loads('''${performance}''')    json
    ${load_time}=    Evaluate    
    ...    (${timing['loadEventEnd']} - ${timing['navigationStart']}) / 1000.0
    Log    Page load time: ${load_time} seconds
    [Return]    ${load_time}

Verify Page Is Loaded Completely
    [Documentation]    Verify that the page is completely loaded
    [Arguments]    ${timeout}=${DEFAULT_TIMEOUT}
    Wait For Condition    
    ...    return document.readyState === 'complete'
    ...    timeout=${timeout}
    Wait For Ajax Requests To Complete    ${timeout}

Handle Alert If Present
    [Documentation]    Handle JavaScript alert if present
    ${alert_present}=    Run Keyword And Return Status    Alert Should Be Present    timeout=2
    Run Keyword If    ${alert_present}    Handle Alert    accept

Extract Table Data
    [Documentation]    Extract data from HTML table
    [Arguments]    ${table_locator}
    ${rows}=    Get WebElements    ${table_locator}//tr
    @{table_data}=    Create List
    FOR    ${row}    IN    @{rows}
        ${cells}=    Get WebElements    ${row}//td|${row}//th
        @{row_data}=    Create List
        FOR    ${cell}    IN    @{cells}
            ${text}=    Get Text    ${cell}
            Append To List    ${row_data}    ${text}
        END
        Append To List    ${table_data}    ${row_data}
    END
    [Return]    @{table_data}

Download File In Headless Mode
    [Documentation]    Configure browser to download files in headless mode
    [Arguments]    ${download_dir}
    ${chrome_options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys
    ${prefs}=    Create Dictionary
    ...    download.default_directory=${download_dir}
    ...    download.prompt_for_download=False
    ...    download.directory_upgrade=True
    ...    safebrowsing.enabled=False
    Call Method    ${chrome_options}    add_experimental_option    prefs    ${prefs}
    Call Method    ${chrome_options}    add_argument    --headless
    Call Method    ${chrome_options}    add_argument    --no-sandbox
    Call Method    ${chrome_options}    add_argument    --disable-dev-shm-usage
    Create Webdriver    Chrome    options=${chrome_options}

Wait Until File Downloaded
    [Documentation]    Wait until file is downloaded
    [Arguments]    ${download_dir}    ${file_pattern}    ${timeout}=30
    Wait Until Keyword Succeeds    ${timeout}s    1s
    ...    File Should Exist    ${download_dir}/${file_pattern}