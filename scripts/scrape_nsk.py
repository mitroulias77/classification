import pandas as pd
import requests
from bs4 import BeautifulSoup as soup
from lxml import html
from warnings import warn
import os
###########################################################
#Εδώ γίνεται το request της σελίδας του νομικού συμβουλίου
columns=['Title','Concultatory','Status','Year']

df_all = pd.DataFrame(columns=columns)
for status in ['1', '-1']:
    for year in range(2000, 2019):
        nsk_url = 'http://www.nsk.gr/web/nsk/anazitisi-gnomodoteseon?p_p_' \
                  'id=nskconsulatories_WAR_nskplatformportlet&p_p_lifecycle=0&p_p_state=normal&p_p_' \
                  'mode=view&p_p_col_id=column-4&p_p_col_pos=2&p_p_col_count=3'
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

        #Εξαγωγή αποφάσεων τίτλων
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

        # Εξαγωγή κατηγοριών από τα λήμματα-keywords

        #keywords = articles.xpath('//*[@id="resdiv"]/div/div[1]/div[2]/p[2]/strong[1]')
        keywords = articles.xpath('//div[@class="article_text"]/p/text()')
        #keywords[3].strip().split (',')

        keywords = [x.strip() for x in keywords]
        keywords = [x for x in keywords if len(x)>0]
        keywords = [x for x in keywords if ',' in x]
        keywords = [x for i, x in enumerate(keywords) if i%2==0]

        punctuation = '"'
        for i, word in enumerate(keywords):
            keywords[i] = word.strip()
        for chr in keywords[:]:
            if chr in punctuation:
                keywords.remove(chr)

        keywords_list = [x.split(',') for x in keywords]


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





