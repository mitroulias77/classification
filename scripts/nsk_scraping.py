import requests
from lxml import html

post_url = 'http://www.nsk.gr/web/nsk/anazitisi-gnomodoteseon?p_p_id=nskconsulatories_WAR_nskplatformportlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-4&p_p_col_pos=2&p_p_col_count=3'
post_data = {"_nskconsulatories_WAR_nskplatformportlet_isSearch": "1",
             "_nskconsulatories_WAR_nskplatformportlet_inputDatefrom": 2019,
             "_nskconsulatories_WAR_nskplatformportlet_consulState": "1"}
index = 0
punctuation = '"'

response = requests.post(post_url, post_data)

if response.status_code == 200:
    page = html.fromstring (response.content.decode ('utf-8'))
    concultatories = page.xpath ('//div[@class="article_text"]/p/strong/a/text()')
    decisions = page.xpath('//*[@id="resdiv"]/div/div[1]/div[2]/text()')

    for item in decisions:
        decisions[index] = item.strip()
        index += 1

    for char in decisions[:]:
        if char in punctuation:
            decisions.remove(char)