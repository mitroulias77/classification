import bs4
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

url ='http://www.nsk.gr/web/nsk/anazitisi-gnomodoteseon'
r = requests.get(url)

soup = bs4.BeautifulSoup(r.content, "html.parser")

g_data = soup.find_all('div', {'class':'anazhthsh'})
