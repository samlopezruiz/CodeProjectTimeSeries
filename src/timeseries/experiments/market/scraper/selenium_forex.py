import random

import pandas as pd
import time

from bs4 import BeautifulSoup

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def create_driver():
    user_agent_list = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
    ]
    user_agent = random.choice(user_agent_list)

    browser_options = webdriver.ChromeOptions()
    browser_options.add_argument("--no-sandbox")
    browser_options.add_argument("--headless")
    browser_options.add_argument("start-maximized")
    browser_options.add_argument("window-size=1900,1080")
    browser_options.add_argument("disable-gpu")
    browser_options.add_argument("--disable-software-rasterizer")
    browser_options.add_argument("--disable-dev-shm-usage")
    browser_options.add_argument(f'user-agent={user_agent}')

    driver = webdriver.Chrome(options=browser_options, service_args=["--verbose", "--log-path=test.log"])

    return driver

def parse_data(driver, url):
    driver.get(url)

    data_table = driver.find_element(By.CLASS_NAME, "calendar__table")
    value_list = []

    for row in data_table.find_elements(By.TAG_NAME, "tr"):
        row_data = list(filter(None, [td.text for td in row.find_elements(By.TAG_NAME, "td")]))
        if row_data:
            value_list.append(row_data)
    return value_list

if __name__ == '__main__':
    #%%
    print('starting scraper...')
    url = 'https://www.forexfactory.com/calendar?month=last'
    driver = create_driver()

    t0 = time.time()
    print('parse data...')
    driver.get(url)


    data_table = driver.find_element(By.CLASS_NAME, "calendar__table")
    # value_list = parse_data(driver=driver, url=url)

    #%%
    #
    # # url = 'https://www.forexfactory.com/calendar?month=sep.2021'
    # import urllib.request as r
    #
    # page = r.urlopen(url)
    # print(page.read())
    #
    # #%%
    # headers = {'Accept-Encoding': 'identity'}
    # response = requests.get(url, headers=headers)
    # data = response.text
    # #%%
    # response = requests.get(url)
    # data = response.text
    # soup = BeautifulSoup(data, 'lxml')
    #
    # table = soup.find('table', class_='calendar__table')
    #
    # list_of_rows = []
    # links = []
    #
    # # Filtering events that have a href link
    # for row in table.find_all('tr', {'data-eventid': True}):
    #     list_of_cells = []
    #
    #     # Filtering high-impact events
    #     for span in row.find_all('span', class_='high'):
    #         links.append(url + "#detail=" + row['data-eventid'])
    #
    #         # Extracting the values from the table data in each table row
    #         for cell in row.find_all('td', class_=[
    #             'calendar__cell calendar__currency currency',
    #             'calendar__cell calendar__event event',
    #             'calendar__cell calendar__actual actual',
    #             'calendar__cell calendar__forecast forecast',
    #             'calendar__cell calendar__previous previous']):
    #             list_of_cells.append(cell.text)
    #         list_of_rows.append(list_of_cells)
    #
    # df = pd.DataFrame(list_of_rows, columns=['Country', 'Event Title', 'Actual', 'Forecast', 'Previous'])
    #
    # df.iloc[:, 0] = df.iloc[:, 0].str.split('\n').str[1]
    # df = df.set_index(df.columns[0])
    # df = df.sort_values('Country')
