import pandas as pd
import requests
from bs4 import BeautifulSoup as soup  # parse html text
from cffi.backend_ctypes import xrange
from lxml import html
from warnings import warn
import os
###########################################################
#Εδώ γίνεται το request της σελίδας του νομικού συμβουλίου

columns=['Title','Concultatory','Status','Year']

df_all = pd.DataFrame(columns=columns)
for status in ['1', '-1']:
    for year in range(2000, 2019):
        nsk_url = 'http://www.nsk.gr/web/nsk/anazitisi-gnomodoteseon?p_p_id=nskconsulatories_WAR_nskplatformportlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-4&p_p_col_pos=2&p_p_col_count=3'
        post_data = {"_nskconsulatories_WAR_nskplatformportlet_isSearch": "1",
                     "_nskconsulatories_WAR_nskplatformportlet_inputDatefrom": year,
                     "_nskconsulatories_WAR_nskplatformportlet_consulState": status}


        response = requests.post(nsk_url , post_data)
        ##########################################################

        #parsing και scraping της σελίδας
        if response.status_code != 200:
            warn('Request: {}; Status code: {}'.format(requests , response.status_code))

        page_soup = soup(response.text, 'html.parser')

        #Εξαγωγή γνωμοδοτήσεων!
        concul_containers = page_soup.find_all("div", {"class" : "article_text"})
        #λίστα με της γνωμοδοτήσεις
        concultatories= []
        for container in concul_containers:
            article = container.strong.text.strip()
            concultatories.append(article)

        #Εξαγωγή αποφάσεων
        articles = html.fromstring (response.content.decode ('utf-8'))
        decisions = articles.xpath('//*[@id="resdiv"]/div/div[1]/div[2]/text()')

        #Καθαρισμός από " και \n\t
        index = 0
        punctuation = '"'
        for item in decisions:
            decisions[index] = item.strip()
            index += 1
        for char in decisions[:]:
            if char in punctuation:
                decisions.remove(char)
        ###############################################################

        #DATA-FRAME και export σε xlsx
        df = pd.DataFrame({'Title':concultatories,
                           'Concultatory':decisions,
                           'Status':status,
                           'Year':year})
        df_all = pd.concat([df_all, df], ignore_index=True)
    #αφαιρεση κενών
    df_all['Title'] = df_all['Title'].apply(lambda x: x.rstrip())
    df_all['Concultatory'] = df_all['Concultatory'].apply(lambda x: x.rstrip())
    #Αποθήκευση df σε xlsx
    fname_path = os.path.join('data','nsk_scrape.xlsx')
    df_all.to_excel(fname_path, index=False)





