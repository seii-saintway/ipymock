# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/2_automation.ipynb (unless otherwise specified).

__all__ = ['driver', 'device_pixel_ratio', 'init', 'quit', 'ok', 'last', 'new', 'close', 'find_elements',
           'find_elements', 'find_element', 'click', 'input', 'get_html_hash', 'wait', 'screen_hash', 'watch',
           'try_log_screen', 'find_position', 'inject', 'get_mouse_position', 'move_to_center', 'move_and_click',
           'exists', 'touch', 'fill', 'convert_md_with_ruby_to_html', 'convert_md_content_to_html', 'save_file',
           'convert_html_with_ruby_to_png', 'convert_md_with_ruby_to_png', 'dialog_for_printing',
           'convert_html_with_ruby_to_pdf', 'convert_md_with_ruby_to_pdf']

# Internal Cell
import os, typing
from selenium.webdriver.remote.webdriver import WebDriver

# Internal Cell
import undetected_chromedriver
from webdriver_manager.chrome import ChromeDriverManager

# Cell
driver: typing.Optional[WebDriver] = None
device_pixel_ratio = 1

# Cell
def init(*arguments):
    chrome_options = undetected_chromedriver.ChromeOptions()
    for argument in arguments:
        if isinstance(argument, str):
            chrome_options.add_argument(argument)
    global driver
    driver = undetected_chromedriver.Chrome(
        options = chrome_options,
        driver_executable_path = os.path.join(
            os.path.dirname(ChromeDriverManager().install()), 'chromedriver.exe' if os.name == 'nt' else 'chromedriver'
        )
    )
    global device_pixel_ratio
    device_pixel_ratio = driver.execute_script('return window.devicePixelRatio;')

# Cell
def quit():
    global driver
    global device_pixel_ratio
    driver.quit()
    driver = None
    device_pixel_ratio = 1

# Internal Cell
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt = '[%(asctime)s][%(levelname)s]<%(name)s> %(message)s',
    datefmt = '%H:%M:%S'
))
logger.addHandler(handler)

# Internal Cell
from selenium.common.exceptions import WebDriverException

# Cell
def ok():
    if driver is None:
        return False
    try:
        if driver.window_handles == []:
            return False
    except WebDriverException:
        return False
    return True

# Cell
def last():
    if not ok():
        return
    driver.switch_to.window(driver.window_handles[-1])

# Cell
def new(url):
    last()
    if not ok():
        init()
    if 'data:,' not in driver.current_url and 'chrome://new-tab-page/' not in driver.current_url:
        driver.switch_to.new_window('tab')
    driver.get(url)

# Cell
def close():
    global driver
    global device_pixel_ratio
    driver.close()
    if not ok():
        driver = None
        device_pixel_ratio = 1

# Internal Cell
from selenium.webdriver.common.by import By

# Internal Cell
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions

# Cell
def find_elements(prompt, exactly = True):
    return [elem for elem in WebDriverWait(driver, 10).until(
        expected_conditions.presence_of_all_elements_located((By.XPATH, f'//*[not(contains(text(), "\n")) and contains(., "{prompt}")]'))
    ) if elem.is_displayed() and (elem.text == prompt or not exactly)]

# Internal Cell
from selenium.common.exceptions import NoSuchElementException

# Cell
def find_elements(prompt, exactly = True):
    for scope in range(1, 5):
        elements = driver.find_elements(By.XPATH, f'//*[.{"/*" * (scope - 1)} and not(.{"/*" * scope}) and contains(., "{prompt}")]')
        elements = [elem for elem in elements if elem.is_displayed() and (elem.text == prompt or not exactly)]
        logger.info(f'Search for text "{prompt}" in scope {scope}: found {len(elements)} element(s)')
        if elements:
            return elements
        if exactly:
            elements = driver.find_elements(By.XPATH, f'//*[.{"/*" * (scope - 1)} and not(.{"/*" * scope}) and @*[.="{prompt}"]]')
        else:
            elements = driver.find_elements(By.XPATH, f'//*[.{"/*" * (scope - 1)} and not(.{"/*" * scope}) and @*[contains(., "{prompt}")]]')
        elements = [elem for elem in elements if elem.is_displayed()]
        logger.info(f'Search for attr "{prompt}" in scope {scope}: found {len(elements)} element(s)')
        if elements:
            return elements
    raise NoSuchElementException

# Cell
def find_element(prompt, closest_prompt = None):
    if closest_prompt is None:
        return find_elements(prompt)[-1]
    if isinstance(closest_prompt, str):
        closest_prompt = find_elements(closest_prompt, False)[-1]
    closest_location = closest_prompt.location
    return min(
        find_elements(prompt),
        key = lambda elem: (elem.location['x'] - closest_location['x']) ** 2 + (elem.location['y'] - closest_location['y']) ** 2
    )

# Internal Cell
from selenium.webdriver.common.action_chains import ActionChains

# Cell
def click(prompt = None, closest_prompt = None, xoffset: int = 0, yoffset: int = 0):
    if prompt is None:
        return move_and_click(xoffset, yoffset, True)
    if isinstance(prompt, str):
        prompt = find_element(prompt, closest_prompt)
    ActionChains(driver).move_to_element_with_offset(prompt, xoffset, yoffset).click().perform()
    return prompt

# Cell
def input(text, prompt = None, closest_prompt = None, xoffset: int = 0, yoffset: int = 0):
    prompt = click(prompt, closest_prompt, xoffset, yoffset)
    ActionChains(driver).send_keys(text).perform()
    return prompt

# Internal Cell
import hashlib, time
from selenium.common.exceptions import StaleElementReferenceException

# Cell
def get_html_hash(xpath = '//body'):
    """Get the hash of the element's outerHTML."""
    # driver is the Selenium WebDriver global instance.
    elements = driver.find_elements(By.XPATH, xpath)
    try:
        html = elements[-1].get_attribute('outerHTML') if elements else ''
    except StaleElementReferenceException:
        html = ''
    return hashlib.md5(html.encode('utf-8')).hexdigest(), time.time()

# Cell
def wait(timeout=float('inf'), stability_duration=1.0, check_interval=0.5, xpath='//body'):
    """
    Wait until the HTML of the specified element does not change.

    Args:
        timeout: Maximum wait time for stabilization (seconds).
        stability_duration: Duration for stabilization (seconds).
        check_interval: Interval to check for changes (seconds).
        xpath: XPATH of the element to monitor for HTML changes.
    """
    # Get the initial hash value
    previous_hash, previous_time = get_html_hash(xpath)

    # Wait until the HTML does not change
    start_time = time.time()
    while True:
        time.sleep(check_interval)

        # Get the current hash value
        current_hash, current_time = get_html_hash(xpath)

        # Check if the hash value has stabilized
        if current_hash == previous_hash:
            if current_time - previous_time >= stability_duration:
                logger.info('HTML content has stabilized.')
                break
        else:
            # Update hash and time if the content changes
            previous_hash, previous_time = current_hash, current_time

        # Check for timeout
        if current_time - start_time >= timeout:
            logger.info('Wait for HTML stabilization timed out.')
            break

# Cell
def screen_hash():
    """Calculate the hash value of the screenshot."""
    return hashlib.md5(driver.get_screenshot_as_base64().encode('utf-8')).hexdigest(), time.time()

# Internal Cell
from sys import float_info

# Cell
def watch(timeout = float_info.max, stability_duration = 1.0, check_interval = 0.5):
    """
    Wait until the screenshot does not change.

    Args:
        timeout: how long to wait for stabilization (seconds)
        stability_duration: duration for stabilization (seconds)
        check_interval: check interval (seconds)
    """
    # Get the initial hash value
    previous_hash, previous_time = screen_hash()

    # Wait until the image does not change
    start_time = previous_time
    while True:
        time.sleep(check_interval)

        # Take another screenshot and calculate the new hash value
        current_hash, current_time = screen_hash()

        # Check if the hash value has changed
        if current_hash == previous_hash:
            if current_time - previous_time >= stability_duration:
                logger.info('Screenshot has stabilized.')
                break
        else:
            if current_time - start_time >= timeout:
                logger.warning('Wait for screenshot stabilization timeout.')
                break
            previous_hash, previous_time = current_hash, current_time

# Cell
def try_log_screen(xpath = None):
    screenshot_path = 'screen.png'
    if isinstance(xpath, str):
        driver.find_element(By.XPATH, xpath).screenshot(screenshot_path)
        return
    driver.save_screenshot(screenshot_path)

# Internal Cell
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

import os, time
from airtest.aircv import get_resolution, imread
from airtest.core.api import Template
from airtest.core.error import TargetNotFoundError

# Cell
def find_position(image, timeout=1.0, threshold=0.9, interval=0.5, intervalfunc=None):
    """
    Search for image template in the screen until timeout

    Args:
        image: image file path to be found in screenshot
        timeout: time interval how long to look for the image template
        threshold: default is None
        interval: sleep interval before next attempt to find the image template
        intervalfunc: function that is executed after unsuccessful attempt to find the image template

    Raises:
        TargetNotFoundError: when image template is not found in screenshot

    Returns:
        TargetNotFoundError if image template not found, otherwise returns the position where the image template has
        been found in screenshot
    """
    logger.info(f'Try to find {image}')
    query = Template(image, rgb = True)
    start_time = time.time()
    while True:
        LOG_DIR = '.'
        file_path = os.path.join(LOG_DIR, 'screen.png')
        driver.save_screenshot(file_path)
        screen = imread(file_path)
        query.resolution = get_resolution(screen)
        # query.resolution = (1920, 1080)
        if screen is None:
            logger.warning('Screen is None: may be locked')
        else:
            if threshold:
                query.threshold = threshold
            match_pos = query.match_in(screen)
            if match_pos:
                # logger.info(f'match_pos == {match_pos}')
                # try_log_screen(screen)
                return match_pos[0] / device_pixel_ratio, match_pos[1] / device_pixel_ratio

        if intervalfunc is not None:
            intervalfunc()

        # Raise an exception if timeout occurs, otherwise proceed to the next loop.
        if (time.time() - start_time) > timeout:
            # try_log_screen(screen)
            raise TargetNotFoundError(f'Picture {query} not found in screen')
        else:
            # ActionChains(driver).move_by_offset(0, 0).perform()
            time.sleep(interval)

def inject():
    # Inject JavaScript code to get the mouse coordinates.
    driver.execute_script("""
var mouseMarkRD = document.createElement('div');
mouseMarkRD.id = 'mouse_mark_rd';
mouseMarkRD.style.position = 'absolute';
mouseMarkRD.style.width = '8px';
mouseMarkRD.style.height = '8px';
mouseMarkRD.style.backgroundColor = 'red';
mouseMarkRD.style.zIndex = '9999';
document.body.appendChild(mouseMarkRD);

var mouseMarkLU = document.createElement('div');
mouseMarkLU.id = 'mouse_mark_lu';
mouseMarkLU.style.position = 'absolute';
mouseMarkLU.style.width = '8px';
mouseMarkLU.style.height = '8px';
mouseMarkLU.style.backgroundColor = 'red';
mouseMarkLU.style.zIndex = '9999';
document.body.appendChild(mouseMarkLU);

var mouseMarkRU = document.createElement('div');
mouseMarkRU.id = 'mouse_mark_ru';
mouseMarkRU.style.position = 'absolute';
mouseMarkRU.style.width = '8px';
mouseMarkRU.style.height = '8px';
mouseMarkRU.style.backgroundColor = 'red';
mouseMarkRU.style.zIndex = '9999';
document.body.appendChild(mouseMarkRU);

var mouseMarkLD = document.createElement('div');
mouseMarkLD.id = 'mouse_mark_ld';
mouseMarkLD.style.position = 'absolute';
mouseMarkLD.style.width = '8px';
mouseMarkLD.style.height = '8px';
mouseMarkLD.style.backgroundColor = 'red';
mouseMarkLD.style.zIndex = '9999';
document.body.appendChild(mouseMarkLD);

document.addEventListener('mousemove', function(e) {
    mouseMarkRD.style.left = e.pageX + 2 + 'px';
    mouseMarkRD.style.top = e.pageY + 2 + 'px';

    mouseMarkLU.style.left = e.pageX - 10 + 'px';
    mouseMarkLU.style.top = e.pageY - 10 + 'px';

    mouseMarkRU.style.left = e.pageX + 2 + 'px';
    mouseMarkRU.style.top = e.pageY - 10 + 'px';

    mouseMarkLD.style.left = e.pageX - 10 + 'px';
    mouseMarkLD.style.top = e.pageY + 2 + 'px';

    window.mouseX = e.clientX;
    window.mouseY = e.clientY;
});
""")

def get_mouse_position():
    return driver.execute_script('return window.mouseX'), driver.execute_script('return window.mouseY')

def move_to_center():
    body = driver.find_element(By.TAG_NAME, 'body')
    ActionChains(driver).move_to_element(body).perform()
    return body.rect['x'] + body.rect['width'] >> 1, body.rect['y'] + body.rect['height'] >> 1

def move_and_click(x, y, offset = True):
    if offset:
        xoffset, yoffset = x, y
    else:
        # Retrieve the mouse coordinates.
        center_x, center_y = move_to_center()
        logger.debug(f'Get center position: {center_x}, {center_y}')
        xoffset, yoffset = x - center_x, y - center_y
    # Move to the specified coordinates and click.
    ActionChains(driver).move_by_offset(xoffset, yoffset).click().perform()
    return xoffset, yoffset

def exists(image):
    try:
        find_position(image)
        return True
    except:
        return False

def touch(image, text = None):
    try:
        x, y = find_position(image)
        move_and_click(x, y, False)
        if isinstance(text, str):
            fill(text)
    except:
        pass

def fill(text):
    ActionChains(driver).send_keys(text).perform()

# Internal Cell
import markdown
import os, re

# Cell
def convert_md_with_ruby_to_html(md_file_path):
    ''' Convert Markdown file containing <ruby> tags into HTML file '''
    # Get the filename without extension
    base_name = os.path.splitext(md_file_path)[0]
    html_file_path = f'{base_name}.html'

    # Read the Markdown file and convert it to HTML
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Create an HTML document while preserving the <ruby> tags
    html_document = convert_md_content_to_html(md_content)
    save_file(html_file_path, html_document)
    return html_file_path, html_document

def convert_md_content_to_html(md_content):
    ''' Convert Markdown content containing <ruby> tags into HTML content '''
    html_content = markdown.markdown(md_content)

    # Replace the newline characters within <p> tags with <br />
    html_content = re.sub(r'<p>(.*?)</p>', lambda m: '<p>' + m.group(1).replace('\n', '<br />') + '</p>', html_content, flags=re.DOTALL)

    return f"""
<html>
    <head>
        <meta charset="utf-8">
        <style>
            ruby rt {{
                font-size: 0.6em; /* 调整注音字体大小 */
            }}
        </style>
    </head>
    <body>
{html_content}
    </body>
</html>
"""

def save_file(file_path, content):
    ''' Write the content to a file '''
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Cell
def convert_html_with_ruby_to_png(html_file_path):
    ''' Use Selenium to print the HTML file as PNG '''
    # Get the filename without extension
    base_name = os.path.splitext(html_file_path)[0]
    png_file_path = f'{base_name}.png'

    # Set up ChromeDriver
    global driver
    if driver is None:
        init('--headless', '--disable-gpu')

    # Open the local HTML file
    new('file://' + os.path.abspath(html_file_path))
    # Wait for the page to fully load
    wait(1.0)

    # Get the current window size
    current_window_size = driver.get_window_size()
    # Set the window size to the resolution of iPhone 16 Pro Max
    # driver.set_window_size(642, 1389)
    driver.set_window_size(642/3*2, 1389/3*2)
    # Capture the page and save as PNG
    driver.save_screenshot(png_file_path)
    # Reset the window size
    driver.set_window_size(current_window_size['width'], current_window_size['height'])
    # Wait for the page to fully load
    wait(1.0)

    close()
    return png_file_path

def convert_md_with_ruby_to_png(md_file_path):
    ''' Convert the Markdown file containing <ruby> tags into PNG '''
    html_file_path, _ = convert_md_with_ruby_to_html(md_file_path)
    return convert_html_with_ruby_to_png(html_file_path)

# Cell
def dialog_for_printing(timeout = float('inf')):
    ''' Dialog for printing to PDF '''
    # Print to PDF
    driver.set_script_timeout(60 * 60 * 24)
    driver.execute_script('window.print();')
    # Wait until the <body> element becomes clickable
    WebDriverWait(driver, timeout).until(expected_conditions.element_to_be_clickable((By.TAG_NAME, 'body')))

# Internal Cell
import base64
from selenium.webdriver.common.print_page_options import PrintOptions

# Cell
def convert_html_with_ruby_to_pdf(html_file_path):
    ''' Use Selenium to print the HTML file as PDF '''
    # Get the filename without extension
    base_name = os.path.splitext(html_file_path)[0]
    pdf_file_path = f'{base_name}.pdf'

    # Set up ChromeDriver
    global driver
    if driver is None:
        init('--disable-gpu', f'--print-to-pdf="{os.path.abspath(pdf_file_path)}"')

    # Open the local HTML file
    new('file://' + os.path.abspath(html_file_path))
    # Wait for the page to fully load
    wait(1.0)

    # Write the decoded data to a PDF file
    print_options = PrintOptions()
    print_options.scale = 1.3
    with open(pdf_file_path, 'wb') as pdf_file:
        pdf_file.write(base64.b64decode(driver.print_page(print_options)))

    close()
    return pdf_file_path

def convert_md_with_ruby_to_pdf(md_file_path):
    ''' Convert the Markdown file containing <ruby> tags into PDF '''
    html_file_path, _ = convert_md_with_ruby_to_html(md_file_path)
    return convert_html_with_ruby_to_pdf(html_file_path)