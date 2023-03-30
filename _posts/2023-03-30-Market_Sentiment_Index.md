```python
# Install Packages
!pip install transformers
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab_light_220429.sh
%cd ..
!git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
%cd mecab-python-0.996
!python setup.py build
%cd ..
!pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
!pip install deepspeed
!pip install mpi4py
!pip install accelerate

# Import Packages
import pytz
import datetime
import requests
import time
from bs4 import BeautifulSoup
import urllib.request
import lxml
from functools import partial
import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import deepspeed
import plotly.graph_objects as go

# Mount Google Drive to Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Set PATH
PATH = '/content/drive/MyDrive/analytics/market_sentiment_index/'
```

# 서론
<br> <i>"시장은 절망에서 태어나 회의 속에서 성장하며 낙관과 함께 만개하고, 환희에서 죽는다" </i> - John Templeton</br>
<br> <i>"주식 시장이 불황일 때, 칵테일 파티의 참석자들은 펀드 매니저에게 관심조차 주지 않는다. 주식 시장이 상승으로 전환할 때, 칵테일 파티의 참석자들은 펀드 매니저에게 '주식은 위험하다'며 한 마디만 말하곤 이내 자리를 떠난다. 주식 시장이 호황일 때, 칵테일 파티의 참석자들은 펀드매니저에게 종목을 물어본다. 주식 시장이 정점일 때, 칵테일 파티의 참석자들은 펀드매니저에게 가르치려 든다." </i>- Peter Lynch </br>
<br>이처럼 선대의 수많은 전설적인 투자자들은 시장의 고점에 대하여 많은 인용구를 만들어내었다. 당장 우리나라만 하더라도 "객장에 주부가 유모차 끌고 나타나면 고점 신호다"는 시쳇말도 있지 않은가. 이 인용구들의 공통점을 요약하면 시장이 고점일 때는 </br>
<br>1) 투자에 문외한인 사람들조차 투자에 관심에 가지고
<br>2) 사람들이 시장에 지나치게 낙관적이다 </br>
<br>고 할 수 있겠다. 지금까지의 약 8년 가까운 투자 인생을 살아오면서 여러 차례의 시장 사이클을 관찰해 왔었고, 나 또한 이러한 인용구들을 적극적으로 활용하여 이때까지 수많은 고점들을 찾아내어 큰 손실을 피할 수 있었다. </br>
<br> 대표적으로 이때까지 있었던 암호화폐 시장의 큰 두 고점을 맞췄던 것인데, 첫 번째 고점은 투자와 아무런 상관이 없는 내 동기가 나에게 코인에 대해서 물어본 날 전부 팔아치운 뒤 두 달 뒤에 시장이 폭락하여 살아남았고, 두 번째 고점은 어느 날 학교 근처 스타벅스에서 과제를 하던 도중 뒷자리의 주부들이 코인에 대해 수다를 떠는 것을 엿들은 날 전부 팔아치운 뒤 세 달 뒤에 시장이 폭락하여 살아남았었다. </br>
<br> 내가 이런 식으로 사람들을 관찰해본 결과, 투자와 아무런 상관이 없는 곳에서 투자에 관련된 이야기가 많이 들리기 시작하면 대략 두 달에서 네 달 사이 뒤에 시장이 폭락한다는 것을 경험적으로 파악할 수 있었다. 그러나 투자 공부를 점점 하면서 단순히 감만으로는 부족할 수도 있겠다는 생각이 들었고, 그래서 이러한 경험법칙을 조금 더 객관화된 수치로 파악할 수 있는 도구를 갖춰야겠다는 생각을 하기 시작하였고, 그에 따라 이번 프로젝트를 고안하게 되었다. </br>
<br> 이렇게 객관적인 지표만으로 시장의 상황을 파악하는 방법을 기술적 분석이라고 한다. 기술적 분석으로는 시장이 효율적으로 변하면 변할수록 장기적으로 시장 수익률보다 지속적으로 좋은 수익률을 내게 해줄 수는 없다는 한계점이 있지만, 그럼에도 시장의 상황을 파악할 수 있는 유용한 숫자들을 제공한다는 점에서 여전히 그 효용성은 유효하다고 할 수 있다. 이번 프로젝트의 목적은 그런 유효한 지표들을 사람들의 감성을 바탕으로 추출해내어 실제 투자에 쓸 수 있게끔 개발하는 것이다.</br>
<br>
# 전략 설정
<br>(1) 어떤 시장을 분석할 것인가? </br>
<br>물론 주식 시장 자체를 분석하는 것이 가장 좋겠지만, 시황에 대한 감성을 분석하기에는 여러 난관이 존재한다. 대표적으로 다음이 있다. </br>
<br>1) 주식 시장에는 여러 종목들이 있고, 시장 상황에 따라 종목의 변동성도 방향성도 크게 차이 난다. 즉 같은 시황이라도 어떤 종목에 투자했느냐에 따라 사람들은 긍정적으로 반응할 수도 있고 부정적으로 반응할 수도 있다는 것이다.
<br>2) 나라마다 시황이 다르기 때문에 시황에 대한 감성도 하나로 통일할 수 없다. </br>
<br> 이런 상황에서, 내가 대안으로서 눈을 돌린 시장이 바로 암호화폐 시장이다. 암호화폐들은 종목들 사이의 가격 변동의 상관성이 매우 높기 때문에 시황이 한 가지로 통일될 수 있고, 거래소가 달라진다고 시황이 달라지는 것도 아니기 때문이다. 마침 투자 경험이 부족한 사람들이 많이 찾는 시장이라는 점도 분석 대상으로서 나에게 큰 메리트로 다가왔다. </br>
<br>(2) 어디로부터 감성을 분석할 것인가? </br>
<br> 처음에는 레딧의 암호화폐 커뮤니티들을 분석하려 하였으나, 이내 방향을 바꿔 디시인사이드 비트코인 갤러리를 분석하기로 하였다. 내가 레딧을 분석하지 않은 이유는 다음과 같다. </br>
<br> 1) 레딧의 암호화폐 커뮤니티는 게시판 관리자가 게시글 관리를 지나치게 엄격하게 하였다. 때문에 해당 서브레딧들에는 코인 정보 관련된 글들이 주류였고, 그 이외의 사람들의 감성이 담겨 있는 소위 '똥글'들은 투자에 관련되었다 할지라도 금방 묻히거나 게시판 관리자가 삭제하였다. 따라서 이번 프로젝트의 목적과는 맞지 않았다.
<br> 2) 규모가 지나치게 컸다. 커뮤니티가 하나로 통일되어 있다면 모를까, 암호화폐라는 하나의 주제 아래에 규모가 비슷비슷한 대규모의 서브레딧들이 존재하였다. 하나로 묶어서 분석하자니 스크랩하기 지나치게 컸고, 하나만 초점을 맞추어 분석하자니 커뮤니티 자체가 흥하고 지는 사이클이 존재하여 분석에 왜곡을 줄 것 같았다.
<br> 3) 레딧 포스트를 스크랩하려면 레딧으로부터 API 키를 받아야만 한다. 일단 이것부터 과정이 복잡한데, 거기에 더해 API를 통해 서버에 요청을 보낼 수 있는 횟수가 제한되어 있었다. </br>
<br> 반면 디시인사이드의 경우 여러 장점이 있었는데, </br>
<br> 1) 디시인사이드는 대한민국 4위의 트래픽을 자랑할 정도로 대한민국 사람들에게 인기가 많은 커뮤니티 사이트이다. 이러한 트래픽은 표본의 대표성을 충분히 보장한다. 그러면서도 레딧만큼 '지나치게' 글이 많지 않아 전체 게시글을 스크랩하기에도 적당한 스케일을 가지고 있었다.
<br> 2) 디시인사이드 비트코인 갤러리는 암호화폐라는 큰 주제 안에서 디시인사이드 갤러리 중 가장 큰 규모의 독보적인 갤러리이고, 갤러리 관리자가 게시글 관리를 느슨하게 하기 때문에 게시판 이용자들이 '날 것 그대로의 감정'을 그대로 '똥글'의 형태로 마치 채팅방을 이용하듯이 게시글을 작성하는 분위기가 있다. 덕분에 이들을 통해 감성을 분석할 수 있는 기회가 있었다.
<br> 3) 사이트 자체의 기술력이 오래된 웹사이트라서 스크래핑에 큰 제한이 없었다. 여러 시행착오 결과 초당 30번의 요청의 속도만 유지한다면 웹사이트가 스크래핑 봇을 차단하지도 않고 제한 없이 스크랩할 수 있게 하였다. </br>
<br> (3) 무슨 언어 모형을 쓸 것인가? </br>
<br> 우선 커뮤니티 게시글은 정형화된 형식의 언어를 사용하는 것이 아니기 때문에 FastText와 같은 카운트 기반이나 BoW 기반의 전통적인 언어 모형으로는 무리라고 판단했기 때문에 BERT 아니면 GPT 중 하나를 선택하려 했었다. 그런데 GPT-2로 학습하려고 하니까 GPT-2가 너무 무거워서 학습과 로드에 너무 많은 시간이 소요되었고, 메모리 부담이 상당하다는 한계에 부딪혔다. 그래서 최종적으로 그보다는 가벼운 BERT를 바탕으로 학습하기로 했다. </br>
<br> 방향성이 정해졌으니, 이제 스크랩을 시작하였다. 정말 다시 하기 싫은 경험이었다. 혹시라도 따라할 사람들이 있을까봐 미리 캐글에다 내가 스크랩한 자료들을 업로드하였으니 다음의 링크를 이용하시길. 파이썬으로는 다음의 함수들을 사용하였다.</br>
<br> https://www.kaggle.com/datasets/soltialuxiant/dcinside-bitcoin-gallery-posts


```python
# # 디시인사이드의 robots.txt는 https://www.dcinside.com/robots.txt에서 확인할 수 있다.
# # 이에 의거해 스크랩 시 서버에 요청을 보낼 user agent를 구글봇으로 설정해준다.
# user_agent = 'Googlebot/2.1 (+http://www.google.com/bot.html)'

# # 주어진 url 주소로부터 beautifulsoup object를 반환하는 함수.
# def download_url(url):
#     request = urllib.request.Request(url, headers = {'User-Agent' : user_agent})
#     try:
#         html = urllib.request.urlopen(request).read()
#         soup = BeautifulSoup(html, 'lxml')
#         return soup
#     except:
#         return None

# # 스크랩 시작 시점의 가장 최근의 게시글 번호를 추출하는 함수.
# def get_recent_post_number(gall_name):
#     req = urllib.request.Request('https://gall.dcinside.com/board/lists/?id=' + gall_name + '&page=1', headers = {'User-Agent' : user_agent})
#     html = urllib.request.urlopen(req).read()
#     soup = BeautifulSoup(html, 'lxml')
#     page_num_candidate = soup.find_all('td', class_ = 'gall_num')
#     for element in page_num_candidate:
#         try:
#             page_num = int(element.text)
#             break
#         except:
#             continue
#     return page_num

# # 게시글 번호의 리스트인 page_num_chunk를 반복문으로 스크랩한다. 너무 많아서 전체 게시글을 10개 덩이로 쪼개서 여덟 개의 인스턴스를 병렬로 짜서 처리하였다.
# # 비트코인 갤러리는 시간이 지남에 따라 이전 게시글들 모음인 '이전 비트코인 갤러리' (갤러리 주소 bitcoins)와 현 갤러리인 '비트코인 갤러리' (갤러리 주소 bitcoins_new1)으로 분리되었다.
# def process_page_num(page_num_chunk, gall_name):
#     result = pd.DataFrame()
#     timeindex = datetime.datetime.now(pytz.timezone('Asia/Seoul')).replace(hour = 0, minute = 0, second = 0, microsecond = 0) # 스크랩 시작 시점의 날짜
#     for i in tqdm.tqdm(page_num_chunk):
#         url = 'https://gall.dcinside.com/board/view/?id=' + gall_name + '&no=' + str(i)
#         soup = download_url(url)
#         if soup is None:
#             continue
#         else:
#             try:
#                 gall_time = soup.find('span', class_ = 'gall_date').text
#             except:
#                 continue # 게시글 주소가 연령제한으로 막힌 경우 웹사이트 양식이 달라 게시글 작성 시간을 추출할 수 없게 된다. 이 경우 무시하고 continue하게 된다.
#             converted_time = datetime.datetime.strptime(gall_time, '%Y.%m.%d %H:%M:%S')
#             if converted_time.date() >= timeindex.date(): # 스크랩 시작 날짜 이전의 게시글들만 스크랩할 것이다.
#                 continue
#             else:
#                 gall_title = soup.find('span', class_ = 'title_subject').text
#                 gall_body = ' '.join([x.text for x in soup.find('div', class_ = 'write_div').find_all('div')])
#                 parsed_text = gall_title + ' ' + gall_body
#                 result = pd.concat([result, pd.DataFrame({'post_num' : [i], 'time' : [np.datetime64(converted_time)], 'text' : [parsed_text]})])
#     return result
```

최종적으로 다음의 자료를 모을 수 있었다. 대략 1340만개 정도의 게시글들이 모였다.


```python
# 이전 비트코인 갤러리의 글들은 글 번호의 혼선을 방지하기 위해 글 번호들을 전부 0으로 바꾸었다.
data
```





  <div id="df-fb744536-ae76-44b4-9e55-7e774e62fb5f">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_num</th>
      <th>time</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2013-11-11 17:03:21</td>
      <td>비트코인 갤러리 이용 안내</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2013-11-11 17:04:57</td>
      <td>여기가 도박갤인가요</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2013-11-11 17:14:48</td>
      <td>가상화폐의 등장 : 비트코인</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2013-11-11 17:17:04</td>
      <td>시발 첫글 뺏기뮤ㅠ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2013-11-11 17:26:42</td>
      <td>비트 코인</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13399971</th>
      <td>4217290</td>
      <td>2023-03-28 23:51:14</td>
      <td>전세계 희귀한 동전</td>
    </tr>
    <tr>
      <th>13399972</th>
      <td>4217294</td>
      <td>2023-03-28 23:54:39</td>
      <td>마스크 씹호재떴다 ㅋㅋㅋㅋㅋㅋ  허재</td>
    </tr>
    <tr>
      <th>13399973</th>
      <td>4217296</td>
      <td>2023-03-28 23:57:16</td>
      <td>스택스 안사고 뭐함?</td>
    </tr>
    <tr>
      <th>13399974</th>
      <td>4217299</td>
      <td>2023-03-28 23:58:53</td>
      <td>노가다 잘하게 생겼다 개추 노가다 못하게 생겼다 비추  ㄱ  ㄱ     - dc o...</td>
    </tr>
    <tr>
      <th>13399975</th>
      <td>4217300</td>
      <td>2023-03-28 23:58:54</td>
      <td>리플 풀매수 조졌다 ㅋㅋㅋ  평딘650  간다!!  평딘650  간다!!   - d...</td>
    </tr>
  </tbody>
</table>
<p>13399976 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fb744536-ae76-44b4-9e55-7e774e62fb5f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fb744536-ae76-44b4-9e55-7e774e62fb5f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fb744536-ae76-44b4-9e55-7e774e62fb5f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




<h1>1) Tokenizer 학습</h1> </br>

> 물론 한국어 BERT 모형 Tokenizer인 KoBERTTokenizer가 따로 개발되어 있긴 하지만, 그 Tokenizer의 경우 XLNet Tokenizer를 기반으로 학습되었기 때문에 호환성에 한계가 있다. 그래서 처음부터 Wordpiece Tokenizer를 바탕으로 따로 학습을 시키기로 결정했다.</br>
<br>Tokenizer를 학습시키기 위해서는 학습시키고자 하는 sample text들의 morphene, 즉 형태소를 먼저 생성해야 한다. 한국어의 경우 morphene을 생성하는 알고리즘으로 okt, kkma, mecab 등이 있는데, okt와 kkma의 경우 자바로 짜여진 모델이라 느리고 메모리 관리도 불안정해서 자바 혐오증이 있는 나로서는 (...) 처음부터 고려 대상이 아니었다. mecab은 원래 일본어 morphene 분리를 위해 만들어진 알고리즘을 한국어에다 적용시켰기 때문에 한국어 사용자의 입장에선 다소 부자연스럽긴 하나, 이번 프로젝트의 경우 대량의 자연어 처리를 할 것이라서 mecab으로 하기로 하였다. mecab은 C 언어로 짜여진 모델이라 굉장히 빠른 morphene 분리가 가능하기 때문이다.</br>
<br> 우선 학습에 필요한 데이터를 들여온다. tokenizer의 vocabulary dictionary pool은 가급적이면 단어가 많을수록 encoding을 더 정확하게 할 수 있기 때문에, 스크랩한 모든 게시글을 전부 훈련시킬 것이다.</br>
<br> 학습 데이터 전처리는 다음과 같이 하기로 했다; 디시인사이드 게시글을 모바일로 업로드할 경우 '- dc official App'이라는 글귀가 뒤에 붙게 된다. 이 글귀는 분석에 필요 없으므로 뺀다. 그리고 'ㅋ'라는 글자가 많이 들어가는데 이것도 분석에 큰 의미가 없다고 판단되어 빼기로 하였다. 그리고 정규 표현식으로 숫자, 영어, 한글, 그리고 특수문자 ?, ., \, %은 살려두기로 하였는데, 어째선지 이렇게 해도 'ㅡ'가 제거되지 않아서 이건 따로 제거하기로 하였다. 마지막으로 쓸데없는 공백을 줄이는 알고리즘도 같이 넣어 텍스트를 압축하도록 하였다. </br>
<br>이제 mecab을 로드한 후, 모든 게시글에 대해 morphene을 추출하여 txt 파일로 저장한다.


```python
## 데이터 처리
# Text Data Preprocess 함수 정의
def text_preprocess(text):
    process_1 = text.replace('- dc official App', '').replace('ㅋ', ' ').replace('\n', ' ').replace('ㅡ', ' ').lower()
    process_2 = re.sub('[^a-zA-Z0-9ㄱ-ㅣ가-힣%?\.]', ' ', process_1) # 정규표현식 적용
    process_3 = ' '.join(list(filter(('').__ne__, process_2.split(' ')))).strip() # 쓸 데 없는 공백 제거
    return process_3

# Data Preprocess
data['text'] = data['text'].apply(text_preprocess)
data = data.dropna(how = 'any').reset_index(drop = True) # Drop NaN
mask = data['text'].isin([''])
data = data[~mask] # Drop Empty Value

# Load Mecab Morphenizer
mecab_morphenizer = Mecab().morphs
total_morph = []

# Save Custom Morphene Dictionary
with open(PATH + 'model/tokenizer/after_mecab.txt', 'w', encoding = 'utf-8') as f:
    for line in tqdm(data['text'].values.tolist()):
        morph_sentence = mecab_morphenizer(line)
        f.write(' '.join(morph_sentence) + '\n')
```

> 다음으로, BERT 용으로 개발된 wordpiece tokenizer 모형을 불러온다. 이 때 주의해야 할 점은 한국어의 경우 대소문자 구분이 없기 때문에 lowercase 설정으로 False로 설정해야 하고, 영어나 일본어와 달리 모아쓰기를 하기 때문에 strip_accent를 반드시 False로 설정해줘야 한다. 만약 strip_accent를 True로 설정하게 되면 '는'을 'ㄴ', 'ㅡ', 'ㄴ'으로 인코딩하는 대참사가 발생한다.</br>
<br> 가급적이면 많은 morphene을 학습시킬 것이기 때문에, 반복 등장 기준을 2로 설정하고, 최대 dictionary 크기를 백만개로 맞추었다. 여기에다 방금 저장한 morphene 파일을 불러와서 학습을 진행하면 된다. </br>
<br> 훈련이 끝나고 나서 모델을 저장하게 되면 vocab.txt라는 파일 하나가 저장될 것이다. 당연히 이걸로 끝나면 안 되고, BERT Tokenizer로 다시 불러온 다음 special token을 따로 추가하는 과정이 있어야 한다.


```python
## Train Tokenizer
# Train WordPiece Tokenizer
tokenizer = BertWordPieceTokenizer(strip_accents = False, lowercase = False)
output_path   = 'hugging_%d'%(1000000)
tokenizer.train(
    files = [PATH + 'model/tokenizer/after_mecab.txt'],
    vocab_size = 1000000,
    min_frequency = 2,
    show_progress = True,
    limit_alphabet = 9999
)

# Save WordPiece Tokenizer
tokenizer.save_model(PATH + 'model/tokenizer')
```

> 특수 토큰의 경우 크게 Begin of Sentence (BOS), End of Sentence (EOS), Unknown (UNK), Unused (unused)가 필요하다. BOS 토큰에다 '[SEP]'을, EOS 토큰에다 '[CLS]'를 추가했고, 현재 tokenizer dictionary에 추가한 단어가 많기 때문에 unknown 토큰이나 unused 토큰도 충분히 많아야 한다. 그래서 unknown 토큰과 unused 토큰을 각각 2000개씩 만들어서 추가하기로 했다. </br>
<br> 결과적으로, 약 34만개의 토큰이 tokenizer dictionary에 추가되었다. 이대로 훈련된 tokenizer를 저장하면 해당 디렉토리에 네 개의 파일이 생성된다. 하나는 configuration이 json 형태로 저장된 것이고, 하나는 vocabulary dictionary가 txt 형태로 저장된 것이다. 나머지 둘 중 하나는 특수 토큰들만 저장해 놓은 것이고 하나는 따로 추가된 토큰들을 저장해 놓은 것인데, 이 파일들은 특수 토큰을 제대로 dictionary에 추가했다면 필요 없는 파일들이다.


```python
# Load WordPiece Tokenizer into BertTokenizerFast
# BertTokenizer는 파이썬으로 짜여진 tokenizer이고, 그걸 Rust로 다시 작성한 것이 BerTokenizerFast이다. 따라서 대량의 텍스트를 처리하려면 BerTokenizerFast를 써야 한다.
tokenizer = BertTokenizerFast.from_pretrained(PATH + 'model/tokenizer', strip_accents = False, lowercase = False)

# Add Special Tokens to Tokenizer
user_defined_symbols = ['[SEP]','[CLS]']
unknown_token_num = 2000
unknown_list = ['[UNK{}]'.format(n) for n in range(unknown_token_num)]
unused_token_num = 2000
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unknown_list + unused_list
special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
tokenizer.add_special_tokens(special_tokens_dict)

# Save the Trained Tokenizer
tokenizer.save_pretrained(PATH + 'model/tokenizer')
```

> 새 문장에다 시범삼아 적용해보았다. 정상적으로 토큰화가 잘 되는 것을 볼 수 있다. 이 과정을 똑바로 하지 않으면 토큰화 결과 [UNK] 토큰이 비정상적으로 많이 등장하게 된다.


```python
# Let's Check the Encoding Performance of Tokenizer
encoding = tokenizer("오늘 비트 안 산 흑우 없제?", return_tensors = "pt")
for e in encoding["input_ids"].tolist()[0]:
    print(tokenizer.convert_ids_to_tokens(e))
```

    [CLS]
    오늘
    비트
    안
    산
    흑우
    없
    ##제
    ?
    [SEP]


<h1>2) 학습 데이터 작성</h1> </br>

> 이제 스크랩한 텍스트 데이터들 중 일부를 가져와 감성 분류에 필요한 학습 데이터를 작성한다. 암호화폐 투자를 하는 사람들은 선물 투기도 많이 하기 때문에 비트코인 갤러리에서는 암호화폐가 올라도 떨어져도 신나하는 사람들이 있다. 그래서 단순히 긍정과 부정으로 텍스트를 분류하기에는 무리가 있었다. 그 대신 나는 크게 세 가지 분류, 'long', 'neutral', 'short'로 글들을 분류하기로 하였고, 다음의 기준에 따라 글들에 라벨링을 하였다. </br>
<br> 1) 'long'으로 라벨링하는 경우 : 암호화폐에 대한 현물 매수나 선물 롱 포지션으로 돈을 번 경우나 해당 포지션에 긍정적인 경우, 숏 포지션으로 돈을 잃었거나 해당 포지션에 부정적인 경우
<br> 2) 'short'로 라벨링하는 경우 : 암호화폐 숏 포지션으로 돈을 번 경우나 해당 포지션에 긍정적인 경우, 롱 포지션이나 현물 매수로 돈을 잃었거나 해당 포지션에 부정적인 경우
<br> 3) 'neutral'로 라벨링하는 경우 : 시장의 등락을 둘 다 표현하거나, 시황과는 상관없는 경우 </br>
<br>처음에는 수작업으로 게시글들을 일일히 라벨링을 하였다. 작업이 많이 부담스러워서 결국에 20000개밖에 못하였다. 이대로는 학습 정확도가 떨어지므로 데이터 강화 작업을 따로 해야 했다.


```python
df
```





  <div id="df-07f9b614-5b34-4f77-b18a-abfc62df7d70">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>한낱 개미가 매수해서 쳐 물려서 존버하고있는데 그 개미를 구해주고 수익까지 줄까??</td>
      <td>short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>너 말야 너 롱충아 정신차려 너 인생 잘풀린적 있어?</td>
      <td>short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>근데 왜 악명높은 주식에서 너가 개나소나하는 무현매수법으로 잘될꺼라 생각하는거지?</td>
      <td>short</td>
    </tr>
    <tr>
      <th>3</th>
      <td>악재해소 랠리 가자</td>
      <td>long</td>
    </tr>
    <tr>
      <th>4</th>
      <td>롱충이 새끼들 또 대가리 깨져서는 푸틴 g20 참석으로 희망회로 돌리네</td>
      <td>short</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19994</th>
      <td>인증할게 고민중.그나저나 개.새.끼들아 ㅃㄹ타라 설마오늘또가는거아니냐? 상품을 선물있어서</td>
      <td>long</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>카바카바카바 상승폭 호재덩어리 영차 고래잡으로가즈아 주움 시발아?</td>
      <td>long</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>얼마했냐 이거 뚫을거 같은데..? 메타다 롱이닷</td>
      <td>long</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>갔을텐데 30k돌파 예상못했다 놓쳤네 세력붙었냐?</td>
      <td>long</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>스팀달러 찐반이냐? 반등각할듯</td>
      <td>long</td>
    </tr>
  </tbody>
</table>
<p>19999 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-07f9b614-5b34-4f77-b18a-abfc62df7d70')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-07f9b614-5b34-4f77-b18a-abfc62df7d70 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-07f9b614-5b34-4f77-b18a-abfc62df7d70');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




> 언어 데이터들에 대해 데이터를 강화하는 방법은 여러 가지가 있지만, 일반적으로 사용되는 방법은 임의의 단어를 문장 사이사이에 집어넣는 방법이다. 이렇게 단어를 임의로 집어 넣어도 전체 문장의 의미 해석에는 큰 변화가 없기 때문에 이런 식으로 단어를 집어넣어 샘플 숫자를 불리면 학습의 정확도를 올리는 데 큰 도움이 된다. 한국어의 경우 찾아보니 python 패키지 중 koEDA라는 패키지가 있었다. 그러나 이번에는 이건 사용하지 않았고, 대신 다른 방식을 사용하기로 했다.</br>
<br> 전략은 이렇다. 한국어의 경우 영어와 달리 어순이 문장의 의미에 영향을 크게 미치지 않는다. 그래서 long으로 분류한 문장들, neutral로 분류한 문장들, 그리고 short로 분류한 문장들에서 공백으로 분리한 단어들끼리 무작위로 섞는다 하더라도 여전히 그 의미는 long으로, neutral로, short로 해석될 수 있을 것이다. 따라서 먼저 공백으로 분리한 vocabulary bag을 만든 후, 교집합에 해당하는 단어들을 전부 제한 뒤에, 그 안에서 무작위로 20 단어씩 샘플링하여 문장을 생성하기로 하였다. 다만 neutral로 분류된 문장의 경우 문장들이 말하는 주제나 어휘의 폭이 너무 넓어 학습에 문제가 생길 것 같아, long과 short에 대해서만 문장을 생성하여 학습을 시키기로 하였다. </br>
<br> 이렇게 각각 4만개씩 long 문장과 short 문장을 생성하였고, 10만개로 샘플을 늘릴 수 있었다. 이제 이 강화된 학습 데이터를 가지고 BERT 학습을 진행할 것이다.


```python
## Data Augmentation
# Data Preprocess
df['Text'] = df['Text'].apply(text_preprocess)
df = df.dropna(how = 'any').reset_index(drop = True) # Drop NaN
mask = df['Text'].isin([''])
df = df[~mask] # Drop Empty Value
df = df[df['Label'].isin(['long', 'short', 'neutral'])] # Check Again for Values in 'Label' Column

# Split Data by the Value of the Column "Label", and Name the Splitted Dataframes as 'long_data', 'short_data', 'neutral_data'
long_data = df[df['Label'] == 'long'].reset_index(drop = True)
short_data = df[df['Label'] == 'short'].reset_index(drop = True)
neutral_data = df[df['Label'] == 'neutral'].reset_index(drop = True)

# Make a Vocabulary Bag
long_vocab = []
for i in range(len(long_data)):
    long_vocab.extend(long_data['Text'][i].split(' '))
short_vocab = []
for i in range(len(short_data)):
    short_vocab.extend(short_data['Text'][i].split(' '))
neutral_vocab = []
for i in range(len(neutral_data)):
    neutral_vocab.extend(neutral_data['Text'][i].split(' '))
long_vocab = list(set(long_vocab))
short_vocab = list(set(short_vocab))
neutral_vocab = list(set(neutral_vocab))
long_vocab = [x for x in long_vocab if x not in short_vocab]
long_vocab = [x for x in long_vocab if x not in neutral_vocab]
short_vocab = [x for x in short_vocab if x not in long_vocab]
short_vocab = [x for x in short_vocab if x not in neutral_vocab]

# Generate 40000 Texts Which Are Consisted With 20 Words Randomly Sampled
sample_data_Text = []
sample_data_Label = []
for i in range(40000):
    text = ' '.join(np.random.choice(long_vocab, 20, replace = False))
    sample_data_Text += text,
    sample_data_Label += 'long',
for i in range(40000):
    text = ' '.join(np.random.choice(short_vocab, 20, replace = False))
    sample_data_Text += text,
    sample_data_Label += 'short',
sampling_data = pd.DataFrame.from_dict({'Text' : sample_data_Text, 'Label' : sample_data_Label})

# Concatenate Dataframe
df = pd.concat([df, sampling_data], axis = 0)
data = df.reset_index(drop = True)
```


```python
data
```





  <div id="df-46439d3b-64e1-4dd7-ad3b-095ffbf82b78">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Text</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>한낱 개미가 매수해서 쳐 물려서 존버하고있는데 그 개미를 구해주고 수익까지 줄까??</td>
      <td>short</td>
    </tr>
    <tr>
      <th>1</th>
      <td>너 말야 너 롱충아 정신차려 너 인생 잘풀린적 있어?</td>
      <td>short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>근데 왜 악명높은 주식에서 너가 개나소나하는 무현매수법으로 잘될꺼라 생각하는거지?</td>
      <td>short</td>
    </tr>
    <tr>
      <th>3</th>
      <td>악재해소 랠리 가자</td>
      <td>long</td>
    </tr>
    <tr>
      <th>4</th>
      <td>롱충이 새끼들 또 대가리 깨져서는 푸틴 g20 참석으로 희망회로 돌리네</td>
      <td>short</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99994</th>
      <td>코인떡락예정 청산후 떨어진다... 갈힘도 원금찾으려면3 접으라네 아아아아아 골렘안가...</td>
      <td>short</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>설거지였노 30%에 쭉정이만 지옥부랄장이다 응징갑니다 축제 안사서 확정되면 뜬금포로...</td>
      <td>short</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>기세등등한데 뺀다 따라가 장투해서 못따라가 떤거는주식시장에서는 함께한다 쏟아 시바아...</td>
      <td>short</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>갖고있다팔면 아닌이상 피말려죽이면서 지지는 버거형들도 왜떨어지는지 임마그때 갈듯하다...</td>
      <td>short</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>떨궜으면 떡락한거? 전부개좃되게 방금전 끝낫어 청산빔도 다른사람들도 센 일어나기 비...</td>
      <td>short</td>
    </tr>
  </tbody>
</table>
<p>99999 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-46439d3b-64e1-4dd7-ad3b-095ffbf82b78')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-46439d3b-64e1-4dd7-ad3b-095ffbf82b78 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-46439d3b-64e1-4dd7-ad3b-095ffbf82b78');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




<h1>3) BERT 학습</h1> </br>

> Colab에서 학습을 진행할 때는 GPU 상으로 학습을 시킬 건데, 이 경우 CUDA를 제대로 사용하기 위해 먼저 추가적인 장치와 변수 설정을 해야 한다.


```python
## Set up Configuration
# Set Up Environment Variable to Properly Run Code in CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# TORCH_CUDA_ARCH_LIST는 각 GPU 라인 별로 값이 다르다. https://developer.nvidia.com/cuda-gpus#compute에서 환경변수 값을 조회할 수 있다.
# Google Colab의 스탠다드 GPU 환경에서는 GPU를 NVIDIA T4를 쓴다. 해당하는 CUDA 이미지 환경변수 값은 7이다.
os.environ['TORCH_CUDA_ARCH_LIST'] = '7'

# Set Device and Architecture
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set Training Hyperparameters
max_encoding_length = 128
learning_rate = 3e-5
epsilon = 1e-8
num_epochs = 5
warmup_step = 0
batch_size = 32
```

> BERT 모형에 문장들을 forward 시키려면 다음의 과정이 필요하다. </br>
<br>1) 인코딩 : tokenizer를 사용하여 문장들을 morphene으로 분리한 후 각각의 morphene에 매치되는 숫자로 변환한다. 이렇게 해서 문장을 벡터로 변환한다.
<br>2) 텐서화 : 이 벡터는 말이 벡터지 사실 그냥 아무것도 아닌 정수 list이다. 따라서 이걸 연산이 가능한 텐서로 변환해줘야 한다. </br>
<br>여기에 학습을 위해선 이렇게 텐서로 만든 샘플들의 벡터를 batch로 묶는 과정이 추가로 더 있어야 한다. pytorch에서는 DataLoader라는 패키지가 이를 담당한다. 여러 차례의 시행착오 끝에 batch size는 32로 하기로 하였다.


```python
## Process Data
# Labels
tag = {"long" : 0, "neutral" : 1, "short" : 2}

# Load Pretrained BERT Tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PATH + 'model/tokenizer', strip_accents = False, lowercase = False)

# Create Tensor Dataset
def make_dataset(data):
    input_ids = []
    labels = []
    for i in range(len(data)):
        text = data.loc[i, 'Text']
        label = tag[data.loc[i, 'Label']]
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = max_encoding_length,           
                            padding = 'max_length',
                            truncation = True,
                            return_tensors = 'pt',      
                    )
        input_ids.append(encoded_dict['input_ids'])
        labels.append(label)
    return input_ids, labels

# 학습 데이터를 다섯 개로 쪼개어 multiprocessing으로 인코딩을 진행한다.
input_ids = []
labels = []
dfs = [df.reset_index(drop = True) for df in np.array_split(data, 5)]
pool = ThreadPool()
result = pool.map(make_dataset, dfs)
for zip in result:
    input_ids += zip[0]
    labels += zip[1]
input_ids = torch.cat(input_ids, dim = 0)
labels = torch.tensor(labels)

# Split Data for Training, Validation, and Testing Into 8:1:1
train_sentences, test_sentences, train_labels, test_labels = train_test_split(input_ids, labels, random_state = 42, test_size = 0.2)
test_sentences, val_sentences, test_labels, val_labels = train_test_split(test_sentences, test_labels, random_state = 42, test_size = 0.5)

# Create Data Loader
train_dataset = TensorDataset(train_sentences, train_labels)
train_loader = DataLoader(
    train_dataset, 
    batch_size = batch_size,
    shuffle = True
)
val_dataset = TensorDataset(val_sentences, val_labels)
val_loader = DataLoader(
    val_dataset, 
    batch_size = batch_size,
    shuffle = True
)
test_dataset = TensorDataset(test_sentences, test_labels)
test_loader = DataLoader(
    test_dataset, 
    batch_size = batch_size, 
    shuffle = True
)
```

> 이제 훈련을 위한 BERT 모형을 불러온다. 여러 가지가 있지만, 이번에 할 것은 라벨링 작업이므로 text classification을 위해 짜여진 BERT Model인 BertForSequenceClassification 패키지를 불러올 것이다. </br>
<br> 한국어 BERT 모형인 KoBERT의 경우 이런 거 없이 그냥 일반적인 BERT Model뿐이기 때문에 만약 그걸 불러와서 훈련시킬 생각이라면 따로 class를 선언한 뒤에 그 안에서 KoBERT를 불러오고, 그 위에다가 torch classifier layer를 따로 올려줘야 하는 수고로운 과정이 필요하다.</br>
<br> 삼중 분류 작업이므로 activation function은 softmax를 사용할 것이고, loss function은 cross entropy를 사용할 것이다. 여러 시행착오 결과 warmup은 굳이 할 필요 없어서 0으로 설정하였고, optimizer는 AdamW로 설정하였다.


```python
## Train Model
# Load BERT Model into GPU
config = BertConfig.from_pretrained('bert-base-multilingual-uncased')
config.num_labels = 3
model = BertForSequenceClassification(config)
model.to(device)
model.resize_token_embeddings(tokenizer.vocab_size) # Customized Tokenizer를 사용하는 경우 이 코드를 반드시 써줘야 한다.

# Setup Optimizer and Scheduler, Loss Criterion
optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
total_step = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = warmup_step, 
    num_training_steps = total_step
)
criterion = nn.CrossEntropyLoss()

# Define Functions for Training
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds = elapsed_rounded))
```

> 이제 훈련을 시작할 것이다. 앞서 분리한 훈련용 데이터로 DataLoader를 만들었으므로, DataLoader 안에 있는 각 batch를 BERT 모형에 forward 시킨 다음 softmax로 각각의 label에 대한 확률 값을 계산하고, 그걸 backward propagation 시켜 cross entropy 값을 계산하는 식이다. </br>
<br> 여러 번의 시행착오 결과 learning_rate는 3e-5, epsilon은 1e-8로 설정하는 것이 적당했었다. 이보다 작으면 국소점 trap에서 탈출하지 못하는 문제가 발생했고, 이보다 크면 정확도가 떨어지는 것을 관찰했다. </br>
<br> 머신러닝을 진행하면서 언더피팅이 발생할 경우 training accuracy가 떨어지므로 바로 파악할 수 있고, 오버피팅이 발생할 경우 training accuracy는 올라가는데 반해 validation loss 값이 증가하고 validation accuracy가 떨어지는 것으로 파악할 수 있다. 따라서 적절한 epoch는 validation loss와 validation accuracy를 봐가면서 해야 하는 것이 맞지만, 여러번 관찰한 결과 대부분 epoch 5에서부터 오버피팅이 관찰되기 시작하여 epoch는 5로 설정하였다. </br>
<br> 이렇게 하여 훈련이 끝난 후 모델의 성능은 test 데이터로 검증한 결과 약 93%의 정확도를 보이게 되었다. 목표치가 90%였으므로 이 정도면 합격이라 생각해 그대로 모델을 쓰기로 하였다.


```python
# for Training and Validation
for epoch_i in range(0, num_epochs):
    print(f'======== Epoch {epoch_i + 1} / {num_epochs} ========')
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in tqdm(enumerate(train_loader)):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        model.zero_grad()        
        logits = model(b_input_ids)[0]
        logits = F.softmax(logits, dim = 1)
        loss = criterion(logits, b_labels)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)            

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in tqdm(val_loader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        with torch.no_grad():        
            logits = model(b_input_ids)[0]
            logits = F.softmax(logits, dim = 1)
            loss = criterion(logits, b_labels)
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    avg_val_loss = total_eval_loss / len(val_loader)

    elapsed_time = format_time(time.time() - t0)  
    print(f'Training Loss : {avg_train_loss:.3f}')
    print(f'Validation Loss : {avg_val_loss:.3f}')
    print(f'Validation Accuracy : {avg_val_accuracy:.3f}')
    print(f'Time elapsed : {elapsed_time}')

# For Evaluation
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

for batch in tqdm(test_loader):
    b_input_ids = batch[0].to(device)
    b_labels = batch[1].to(device)
    with torch.no_grad():        
        logits = model(b_input_ids)[0]
        logits = F.softmax(logits, dim = 1)
        loss = criterion(logits, b_labels)
    total_eval_loss += loss.item()
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    total_eval_accuracy += flat_accuracy(logits, label_ids)
avg_test_accuracy = total_eval_accuracy / len(test_loader)
avg_test_loss = total_eval_loss / len(test_loader)
print(f'Test Loss : {avg_test_loss:.3f}')
print(f'Test Accuracy : {avg_test_accuracy:.3f}')
```

> 한 번 샘플 문장으로 모델의 성능을 시험해보기로 하였다. 마찬가지로 전처리 하고, 인코딩한 후에, 텐서로 변환하고, 모델에다 forward 한 다음에, softmax로 확률 뽑아내서, 각각의 라벨에 대한 확률끼리 비교해 가장 높은 확률의 분류로 분류값을 출력하면 된다.


```python
## Let's Test the Model!
# Define a Function for Predict
def predict(model, sentence, tokenizer, max_encoding_length):
    model.eval()
    text = text_preprocess(sentence)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_encoding_length,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )
    input_ids = encoding['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(input_ids = input_ids)
        logits = outputs[0]
        probability = F.softmax(logits, dim = 1)
    label_probs = probability.squeeze().tolist()
    prob_long = label_probs[0]
    prob_neutral = label_probs[1]
    prob_short = label_probs[2]
    if prob_long > prob_neutral and prob_long > prob_short:
        classification = 'long'
    if prob_neutral > prob_long and prob_neutral > prob_short:
        classification = 'neutral'
    if prob_short > prob_long and prob_short > prob_neutral:
        classification = 'short'
    return classification, prob_long, prob_neutral, prob_short

# Load Models
model = BertForSequenceClassification.from_pretrained(PATH + 'model')
tokenizer = BertTokenizerFast.from_pretrained(PATH + 'model/tokenizer', strip_accents = False, lowercase = False)
model.to(device)
model.resize_token_embeddings(tokenizer.vocab_size)
```


```python
# Predict a Text
sentence = '오늘 저녁은 쌰충이 머리털볶음봐아아아압 엌ㅋㅋㅋㅋㅋ'
classification, prob_long, prob_neutral, prob_short = predict(model, sentence, tokenizer, max_encoding_length)
print('Classification: {}, Probability Long: {:.2f}%, Probability Neutral: {:.2f}%, Probability Short: {:.2f}%'.format(classification, prob_long * 100, prob_neutral * 100, prob_short * 100))
```

    Classification: long, Probability Long: 100.00%, Probability Neutral: 0.00%, Probability Short: 0.00%


> ...뭐, 잘 하는 것 같다. </br>
<br> 이제 이 모델을 저장할 것이다. BERT 모형의 config 파일도 필요하기 때문에 'save_pretrained()'로도 모형을 저장할 것이다. </br>
<br> 이렇게 저장하면 총 세 개의 파일이 디렉토리에 저장된다.
<br> 1) 모형의 파이토치 파일의 두 가지 형태 : .bin (바이너리 파일), .pt (파이토치 스크립트)
<br> 2) 그리고 BERT 모형의 config 파일인 config.json. 이건 tokenizer의 config 파일과는 다른 config 파일이므로 반드시 디렉토리를 구분해서 저장해야 한다.


```python
## Save Model
# Save Model as BERT Config
model.save_pretrained(PATH + 'model')

# Save Model as pytorch model
torch.save(model, PATH + 'model/sentiment_model.pt')
```

<h1>4) BERT 모델 로드 및 라벨링</h1> </br>

> 이제 저장한 BERT 모델과 Tokenizer를 로드하여 스크랩한 텍스트들에 라벨을 붙일 것이다. 대량의 데이터들에 대한 라벨링 작업이기 때문에 단순히 모델을 로드하여 처리하면 매우 많은 컴퓨팅 시간이 필요하게 된다. 그렇기 때문에 모델 추론을 할 때는 다음과 같이 하기로 하였다. </br>
<br> 1) batch화를 통한 텐서 연산 : CUDA는 텐서 연산에 특화된 GPGPU이다. 단순히 병렬 처리를 하게 되면 파이썬의 GIL에 가로막혀 연산 속도에 제한이 걸리지만, 여러 샘플들을 batch로 묶어 하나의 텐서로 만든 후 연산하면 단일 텐서 연산으로 바뀌기 때문에 GIL을 우회하여 굉장히 빠른 연산을 가능하게 한다.
<br> 2) deepspeed pipeline : 2022년에 Microsoft에서 개발한 대규모 언어모델 추론용 pipeline 라이브러리이다. C++로 작성되었고 CUDA를 통한 병렬 연산을 십분 활용할 수 있게끔 최적화한 torch 라이브러리이다. </br>
<br> 데이터를 세 개로 쪼개어 세 개의 인스턴스를 사용해 병렬로 진행하였다.


```python
## Labeling
# Load Models and Construct Deepspeed Inference Pipeline
device = torch.device('cuda')
tokenizer = BertTokenizerFast.from_pretrained(PATH + 'model/tokenizer', strip_accents = False, lowercase = False)
model = BertForSequenceClassification.from_pretrained(PATH + 'model')
model.to(device)
model.resize_token_embeddings(tokenizer.vocab_size)
model = deepspeed.init_inference(model = model, config = None)

# Define a Function to Return Numpy Array before Converting into Tensor
def make_dataset_numpy(data):
    print("Start Making Dataset...")
    input_ids = np.zeros(shape = (len(data), 128))
    for i in tqdm(range(len(data))):
        text = data.loc[i, 'text']
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = 128,           
                            padding = 'max_length',
                            truncation = True,
                            return_tensors = 'pt',      
                    )
        input_ids[i] = encoded_dict['input_ids']
    return input_ids

# Define a Function for Labeling
def sentiment_process(data):
    new_data = data.copy()
    new_data['text'] = new_data['text'].apply(text_preprocess)
    dataset = torch.LongTensor(make_dataset_numpy(new_data)).to(device) # 인코딩 Array를 텐서로 변환한 후, CUDA에 로드한다.
    batch_size = 64
    num_batches = len(dataset) // batch_size + 1
    new_list = []
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch = dataset[i*batch_size:(i+1)*batch_size] # 모든 샘플들을 하나의 텐서 안에 넣으면 Out Of Memory Error가 나타나니까 64개씩 묶어 batch화한다.
            predictions = model(batch).logits
            predictions = F.softmax(predictions, dim = 1)
            predictions = predictions.squeeze().tolist()
            for label_probs in predictions:
                if label_probs[0] > label_probs[1] and label_probs[0] > label_probs[2]:
                    sentiment = 'long'
                if label_probs[1] > label_probs[0] and label_probs[1] > label_probs[2]:
                    sentiment = 'neutral'
                if label_probs[2] > label_probs[0] and label_probs[2] > label_probs[1]:
                    sentiment = 'short'
                row = pd.DataFrame({'long': label_probs[0], 'neutral': label_probs[1], 'short': label_probs[2], 'sentiment': sentiment}, index = [0])
                new_list.append(row)
    prob_df = pd.concat(new_list, axis = 0) # Serial Data를 처리한 거니까 순서가 그대로 보존된다. 이대로 dataframe 병합을 하면 된다.
    prob_df = prob_df.reset_index(drop = True)
    result_df = pd.concat([data, prob_df], axis = 1)
    return result_df

# Load Data and Start Processing
data = pd.read_csv(PATH + 'dcinside_bitcoin_gallery.csv').reset_index(drop = True).drop(columns = ['Unnamed: 0'], axis = 1)
data["text"] = data["text"].astype(str)
labeled_df = sentiment_process(data)
```

> 약 한 시간 30분만에 다음과 같이 라벨링 작업이 모두 완료되었다. </br>
<br> 이거 하면서 솔직하게 말하자면, 속도에 놀랐다. 한 시간에 1000만 샘플의 속도로 진행되었다. deepspeed는 마이크로소프트와 NVIDIA가 개발한 대규모 언어 생성 모델인 Megatron-Turing NLG의 빠른 작동 및 학습을 위해 2022년 초에 개발된 패키지이다. GPT나 Megatron과 같은 대규모 언어 모형을 상대로도 빠른 학습시간을 장담한다고 발표했었는데, 과연 그럴만 하다. </br>
<br> BERT의 경우 GPT-3의 램 용량인 12GB인 것에 비해 20분의 1 정도인 500MB 정도의 램 용량을 가지는데, 몇 년 전까지만 해도 이 정도의 램 용량을 차지하는 언어 모형도 무겁다고 아우성이었던 걸 생각해보면 세상이 정말 빠르게 변하고 있다는 걸 실감하게 된다.


```python
labeled_df
```





  <div id="df-083de3e8-a7cd-4423-ad81-e9fa9fe72bb4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_num</th>
      <th>time</th>
      <th>text</th>
      <th>long</th>
      <th>neutral</th>
      <th>short</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2013-11-11 17:03:21</td>
      <td>비트코인 갤러리 이용 안내</td>
      <td>0.000068</td>
      <td>1.000000</td>
      <td>0.000006</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2013-11-11 17:04:57</td>
      <td>여기가 도박갤인가요</td>
      <td>0.000004</td>
      <td>0.000002</td>
      <td>1.000000</td>
      <td>short</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2013-11-11 17:14:48</td>
      <td>가상화폐의 등장 : 비트코인</td>
      <td>0.000004</td>
      <td>1.000000</td>
      <td>0.000088</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2013-11-11 17:17:04</td>
      <td>시발 첫글 뺏기뮤ㅠ</td>
      <td>0.000006</td>
      <td>1.000000</td>
      <td>0.000079</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2013-11-11 17:26:42</td>
      <td>비트 코인</td>
      <td>0.000006</td>
      <td>1.000000</td>
      <td>0.000022</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13399971</th>
      <td>4217290</td>
      <td>2023-03-28 23:51:14</td>
      <td>전세계 희귀한 동전</td>
      <td>0.000072</td>
      <td>1.000000</td>
      <td>0.000006</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>13399972</th>
      <td>4217294</td>
      <td>2023-03-28 23:54:39</td>
      <td>마스크 씹호재떴다 ㅋㅋㅋㅋㅋㅋ  허재</td>
      <td>0.000072</td>
      <td>1.000000</td>
      <td>0.000006</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>13399973</th>
      <td>4217296</td>
      <td>2023-03-28 23:57:16</td>
      <td>스택스 안사고 뭐함?</td>
      <td>1.000000</td>
      <td>0.000002</td>
      <td>0.000004</td>
      <td>long</td>
    </tr>
    <tr>
      <th>13399974</th>
      <td>4217299</td>
      <td>2023-03-28 23:58:53</td>
      <td>노가다 잘하게 생겼다 개추 노가다 못하게 생겼다 비추  ㄱ  ㄱ     - dc o...</td>
      <td>0.000004</td>
      <td>1.000000</td>
      <td>0.000041</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>13399975</th>
      <td>4217300</td>
      <td>2023-03-28 23:58:54</td>
      <td>리플 풀매수 조졌다 ㅋㅋㅋ  평딘650  간다!!  평딘650  간다!!   - d...</td>
      <td>0.000046</td>
      <td>1.000000</td>
      <td>0.000006</td>
      <td>neutral</td>
    </tr>
  </tbody>
</table>
<p>13399976 rows × 7 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-083de3e8-a7cd-4423-ad81-e9fa9fe72bb4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-083de3e8-a7cd-4423-ad81-e9fa9fe72bb4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-083de3e8-a7cd-4423-ad81-e9fa9fe72bb4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




<h1>5) 라벨링 데이터를 바탕으로 시계열 데이터 작성</h1> </br>

> 라벨링 과정이 끝났으니 이제 이걸 바탕으로 시계열 데이터를 만들 것이다. 각 날짜별로 'long', 'neutral', 'short'로 라벨링된 게시글의 수를 카운트하여 시계열 데이터로 만들 것이다.


```python
## Convert into Time Series Data
# 우선 'time' 열의 값들을 pandas datetime type으로 바꿔줘야 한다. 이후 시간을 index로 설정한다.
labeled_df['time'] = pd.to_datetime(labeled_df['time'])
labeled_df.set_index('time', inplace = True)

# 날짜 단위로 그룹화한다.
df_group = labeled_df.groupby(labeled_df.index.date)
dfs = [df_group.get_group(x) for x in df_group.groups]

# 날짜마다 long, neutral, short로 라벨링된 게시글의 수를 카운트하는 함수.
def convert_into_time_series(df):
    split_group = df.groupby('sentiment')
    group_list = [split_group.get_group(x) for x in split_group.groups]
    date_num = df.index[0].date()
    long_count = 0
    neutral_count = 0
    short_count = 0
    for sentiment_group in group_list:
        if sentiment_group['sentiment'][0] == 'long':
            long_count = len(sentiment_group)
        if sentiment_group['sentiment'][0] == 'neutral':
            neutral_count = len(sentiment_group)
        if sentiment_group['sentiment'][0] == 'short':
            short_count = len(sentiment_group)
    # 어차피 1을 더해도 대세에 지장이 없기 때문에, 인덱스를 만들기 편하게 0이 나오지 않도록 1을 더했다.
    row = pd.DataFrame({'long' : (long_count + 1), 'neutral' : (neutral_count + 1), 'short' : (short_count + 1)}, index = [date_num])
    return row

# Process Data
processed = []
for df in dfs:
    df_count = convert_into_time_series(df)
    processed.append(df_count)
result = pd.concat(processed, axis = 0)
result = result.sort_index()
result = result.reset_index(drop = False)
result.rename(columns = {'index' : 'date'}, inplace = True)
```

> 최종적으로, 다음과 같은 시계열 데이터를 만들 수 있었다. 이제 여기서 인덱스를 만들어보고, 비트코인 가격 차트와 비교해보겠다.


```python
result
```





  <div id="df-b4cb349e-d2bc-4623-98be-079f1624727a">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>long</th>
      <th>neutral</th>
      <th>short</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013-11-11</td>
      <td>4</td>
      <td>31</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013-11-12</td>
      <td>3</td>
      <td>36</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013-11-13</td>
      <td>3</td>
      <td>30</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-11-14</td>
      <td>3</td>
      <td>30</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-11-15</td>
      <td>3</td>
      <td>29</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2584</th>
      <td>2023-03-24</td>
      <td>790</td>
      <td>1416</td>
      <td>760</td>
    </tr>
    <tr>
      <th>2585</th>
      <td>2023-03-25</td>
      <td>598</td>
      <td>953</td>
      <td>672</td>
    </tr>
    <tr>
      <th>2586</th>
      <td>2023-03-26</td>
      <td>668</td>
      <td>1053</td>
      <td>569</td>
    </tr>
    <tr>
      <th>2587</th>
      <td>2023-03-27</td>
      <td>701</td>
      <td>1393</td>
      <td>809</td>
    </tr>
    <tr>
      <th>2588</th>
      <td>2023-03-28</td>
      <td>693</td>
      <td>1290</td>
      <td>864</td>
    </tr>
  </tbody>
</table>
<p>2589 rows × 4 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b4cb349e-d2bc-4623-98be-079f1624727a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b4cb349e-d2bc-4623-98be-079f1624727a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b4cb349e-d2bc-4623-98be-079f1624727a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




> 인덱스는 총 두 가지를 만들어볼 것이다. 앞서 인용한 투자가들의 인용구들의 공통점을 요약하면 시장이 고점일 때는 </br>
<br>1) 투자에 문외한인 사람들조차 투자에 관심에 가지고
<br>2) 사람들이 시장에 지나치게 낙관적이다 </br>
<br>는 특징이 있는 것을 찾았다. 이걸 인덱스화하기 위해 다음의 논리를 바탕으로 인덱스를 만들어보았다. </br>
<br>1) 비트코인 갤러리는 시장에 큰 변동이 찾아오면 화제거리가 생기기 때문에 투자와 관련된 글들이 급증한다. 이걸 바탕으로 전체 글 수 대비 'long'과 'short'로 라벨링 된 글의 수가 이런 상황이면 늘 것이라고 가정하고 인덱스를 짜 보았다. 이를 market interest index라고 해보자.</br>
<br>2) 시장이 좋을 때 가격이 오른다고 주장하는 사람들이 내린다고 주장하는 사람들보다 많을 것이라고 생각했다. 갤러리에 게시글을 작성하는 사람들이 모두 다른 사람이라고 가정하고, 이를 인덱스화하기 위해 'long'으로 라벨링 된 글의 수를 'short'으로 라벨링된 글의 수로 나눈 후, 변동 폭을 줄이기 위해 여기에 로그를 씌웠다. 이를 positivity index라고 해보자. </br>
<br> 비트코인 갤러리는 2013년 11월 11일부터 시작되었다. 어느 곳에서도 2013년 11월 11일부터 현재 글 작성 시점인 2023년 3월 28일까지의 비트코인의 일일 종가를 기록한 곳이 없어 여러 출처를 바탕으로 짜깁기 하였다. 비트코인 가격은 다음의 세 곳을 참조하였다.</br>
<br>https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
<br> https://www.kaggle.com/datasets/varpit94/bitcoin-data-updated-till-26jun2021
<br> https://finance.yahoo.com/quote/BTC-USD/history?period1=1648252800&period2=1680134400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true </br>
<br> 이제 모든 정보들을 한데 모아서 시계열 데이터를 다음과 같이 가공해보았다. 가공하던 도중 2017년 5월 이전의 자료들이 어느 쪽으로나 부실한 점이 많아서 결국 커트하게 되었다.


```python
result
```





  <div id="df-6d10921a-f30c-4f8e-9367-e8da30cf47f7">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>long</th>
      <th>neutral</th>
      <th>short</th>
      <th>positivity_index</th>
      <th>market_interest_index</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-05-01</td>
      <td>114</td>
      <td>349</td>
      <td>61</td>
      <td>0.625325</td>
      <td>0.333969</td>
      <td>1421.600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-05-02</td>
      <td>140</td>
      <td>466</td>
      <td>83</td>
      <td>0.522802</td>
      <td>0.323657</td>
      <td>1452.820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-05-03</td>
      <td>85</td>
      <td>273</td>
      <td>44</td>
      <td>0.658462</td>
      <td>0.320896</td>
      <td>1490.090</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-05-04</td>
      <td>207</td>
      <td>599</td>
      <td>86</td>
      <td>0.878371</td>
      <td>0.328475</td>
      <td>1537.670</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-05-05</td>
      <td>239</td>
      <td>769</td>
      <td>131</td>
      <td>0.601266</td>
      <td>0.324846</td>
      <td>1555.450</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2153</th>
      <td>2023-03-24</td>
      <td>790</td>
      <td>1416</td>
      <td>760</td>
      <td>0.038715</td>
      <td>0.522589</td>
      <td>27493.285</td>
    </tr>
    <tr>
      <th>2154</th>
      <td>2023-03-25</td>
      <td>598</td>
      <td>953</td>
      <td>672</td>
      <td>-0.116668</td>
      <td>0.571300</td>
      <td>27494.707</td>
    </tr>
    <tr>
      <th>2155</th>
      <td>2023-03-26</td>
      <td>668</td>
      <td>1053</td>
      <td>569</td>
      <td>0.160408</td>
      <td>0.540175</td>
      <td>27994.330</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>2023-03-27</td>
      <td>701</td>
      <td>1393</td>
      <td>809</td>
      <td>-0.143291</td>
      <td>0.520152</td>
      <td>27139.889</td>
    </tr>
    <tr>
      <th>2157</th>
      <td>2023-03-28</td>
      <td>693</td>
      <td>1290</td>
      <td>864</td>
      <td>-0.220543</td>
      <td>0.546891</td>
      <td>27268.131</td>
    </tr>
  </tbody>
</table>
<p>2158 rows × 7 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6d10921a-f30c-4f8e-9367-e8da30cf47f7')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6d10921a-f30c-4f8e-9367-e8da30cf47f7 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6d10921a-f30c-4f8e-9367-e8da30cf47f7');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
## Let's Plot!
# Plotting Bitcoin Price
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x = result['date'], y = result['price'], name = 'price'))
fig1.update_layout(title = 'price', xaxis_title = 'date', yaxis_title = 'value')
fig1.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="3e390539-2faa-4709-8c01-cb88906cddc1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("3e390539-2faa-4709-8c01-cb88906cddc1")) {                    Plotly.newPlot(                        "3e390539-2faa-4709-8c01-cb88906cddc1",                        [{"name":"price","x":["2017-05-01","2017-05-02","2017-05-03","2017-05-04","2017-05-05","2017-05-06","2017-05-07","2017-05-08","2017-05-09","2017-05-10","2017-05-11","2017-05-12","2017-05-13","2017-05-14","2017-05-15","2017-05-16","2017-05-17","2017-05-18","2017-05-19","2017-05-20","2017-05-21","2017-05-22","2017-05-23","2017-05-24","2017-05-25","2017-05-26","2017-05-27","2017-05-28","2017-05-29","2017-05-30","2017-05-31","2017-06-01","2017-06-02","2017-06-03","2017-06-04","2017-06-05","2017-06-06","2017-06-07","2017-06-08","2017-06-09","2017-06-10","2017-06-11","2017-06-12","2017-06-13","2017-06-14","2017-06-15","2017-06-16","2017-06-17","2017-06-18","2017-06-19","2017-06-20","2017-06-21","2017-06-22","2017-06-23","2017-06-24","2017-06-25","2017-06-26","2017-06-27","2017-06-28","2017-06-29","2017-06-30","2017-07-01","2017-07-02","2017-07-03","2017-07-04","2017-07-05","2017-07-06","2017-07-07","2017-07-08","2017-07-09","2017-07-10","2017-07-11","2017-07-12","2017-07-13","2017-07-14","2017-07-15","2017-07-16","2017-07-17","2017-07-18","2017-07-19","2017-07-20","2017-07-21","2017-07-22","2017-07-23","2017-07-24","2017-07-25","2017-07-26","2017-07-27","2017-07-28","2017-07-29","2017-07-30","2017-07-31","2017-08-01","2017-08-02","2017-08-03","2017-08-04","2017-08-05","2017-08-06","2017-08-07","2017-08-08","2017-08-09","2017-08-10","2017-08-11","2017-08-12","2017-08-13","2017-08-14","2017-08-15","2017-08-16","2017-08-17","2017-08-18","2017-08-19","2017-08-20","2017-08-21","2017-08-22","2017-08-23","2017-08-24","2017-08-25","2017-08-26","2017-08-27","2017-08-28","2017-08-29","2017-08-30","2017-08-31","2017-09-01","2017-09-02","2017-09-03","2017-09-04","2017-09-05","2017-09-06","2017-09-07","2017-09-08","2017-09-09","2017-09-10","2017-09-11","2017-09-12","2017-09-13","2017-09-14","2017-09-15","2017-09-16","2017-09-17","2017-09-18","2017-09-19","2017-09-20","2017-09-21","2017-09-22","2017-09-23","2017-09-24","2017-09-25","2017-09-26","2017-09-27","2017-09-28","2017-09-29","2017-09-30","2017-10-01","2017-10-02","2017-10-03","2017-10-04","2017-10-05","2017-10-06","2017-10-07","2017-10-08","2017-10-09","2017-10-10","2017-10-11","2017-10-12","2017-10-13","2017-10-14","2017-10-15","2017-10-16","2017-10-17","2017-10-18","2017-10-19","2017-10-20","2017-10-21","2017-10-22","2017-10-23","2017-10-24","2017-10-25","2017-10-26","2017-10-27","2017-10-28","2017-10-29","2017-10-30","2017-10-31","2017-11-01","2017-11-02","2017-11-03","2017-11-04","2017-11-05","2017-11-06","2017-11-07","2017-11-08","2017-11-09","2017-11-10","2017-11-11","2017-11-12","2017-11-13","2017-11-14","2017-11-15","2017-11-16","2017-11-17","2017-11-18","2017-11-19","2017-11-20","2017-11-21","2017-11-22","2017-11-23","2017-11-24","2017-11-25","2017-11-26","2017-11-27","2017-11-28","2017-11-29","2017-11-30","2017-12-01","2017-12-02","2017-12-03","2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08","2017-12-09","2017-12-10","2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15","2017-12-16","2017-12-17","2017-12-18","2017-12-19","2017-12-20","2017-12-21","2017-12-22","2017-12-23","2017-12-24","2017-12-25","2017-12-26","2017-12-27","2017-12-28","2017-12-29","2017-12-30","2017-12-31","2018-01-01","2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-06","2018-01-07","2018-01-08","2018-01-09","2018-01-10","2018-01-11","2018-01-12","2018-01-13","2018-01-14","2018-01-15","2018-01-16","2018-01-17","2018-01-18","2018-01-19","2018-01-20","2018-01-21","2018-01-22","2018-01-23","2018-01-24","2018-01-25","2018-01-26","2018-01-27","2018-01-28","2018-01-29","2018-01-30","2018-01-31","2018-02-01","2018-02-02","2018-02-03","2018-02-04","2018-02-05","2018-02-06","2018-02-07","2018-02-08","2018-02-09","2018-02-10","2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23","2018-02-24","2018-02-25","2018-02-26","2018-02-27","2018-02-28","2018-03-01","2018-03-02","2018-03-03","2018-03-04","2018-03-05","2018-03-06","2018-03-07","2018-03-08","2018-03-09","2018-03-10","2018-03-11","2018-03-12","2018-03-13","2018-03-14","2018-03-15","2018-03-16","2018-03-17","2018-03-18","2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26","2018-03-27","2018-03-28","2018-03-29","2018-03-30","2018-03-31","2018-04-01","2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15","2018-04-16","2018-04-17","2018-04-18","2018-04-19","2018-04-20","2018-04-21","2018-04-22","2018-04-23","2018-04-24","2018-04-25","2018-04-26","2018-04-27","2018-04-28","2018-04-29","2018-04-30","2018-05-01","2018-05-02","2018-05-03","2018-05-04","2018-05-05","2018-05-06","2018-05-07","2018-05-08","2018-05-09","2018-05-10","2018-05-11","2018-05-12","2018-05-13","2018-05-14","2018-05-15","2018-05-16","2018-05-17","2018-05-18","2018-05-19","2018-05-20","2018-05-21","2018-05-22","2018-05-23","2018-05-24","2018-05-25","2018-05-26","2018-05-27","2018-05-28","2018-05-29","2018-05-30","2018-05-31","2018-06-01","2018-06-02","2018-06-03","2018-06-04","2018-06-05","2018-06-06","2018-06-07","2018-06-08","2018-06-09","2018-06-10","2018-06-11","2018-06-12","2018-06-13","2018-06-14","2018-06-15","2018-06-16","2018-06-17","2018-06-18","2018-06-19","2018-06-20","2018-06-21","2018-06-22","2018-06-23","2018-06-24","2018-06-25","2018-06-26","2018-06-27","2018-06-28","2018-06-29","2018-06-30","2018-07-01","2018-07-02","2018-07-03","2018-07-04","2018-07-05","2018-07-06","2018-07-07","2018-07-08","2018-07-09","2018-07-10","2018-07-11","2018-07-12","2018-07-13","2018-07-14","2018-07-15","2018-07-16","2018-07-17","2018-07-18","2018-07-19","2018-07-20","2018-07-21","2018-07-22","2018-07-23","2018-07-24","2018-07-25","2018-07-26","2018-07-27","2018-07-28","2018-07-29","2018-07-30","2018-07-31","2018-08-01","2018-08-02","2018-08-03","2018-08-04","2018-08-05","2018-08-06","2018-08-07","2018-08-08","2018-08-09","2018-08-10","2018-08-11","2018-08-12","2018-08-13","2018-08-14","2018-08-15","2018-08-16","2018-08-17","2018-08-18","2018-08-19","2018-08-20","2018-08-21","2018-08-22","2018-08-23","2018-08-24","2018-08-25","2018-08-26","2018-08-27","2018-08-28","2018-08-29","2018-08-30","2018-08-31","2018-09-01","2018-09-02","2018-09-03","2018-09-04","2018-09-05","2018-09-06","2018-09-07","2018-09-08","2018-09-09","2018-09-10","2018-09-11","2018-09-12","2018-09-13","2018-09-14","2018-09-15","2018-09-16","2018-09-17","2018-09-18","2018-09-19","2018-09-20","2018-09-21","2018-09-22","2018-09-23","2018-09-24","2018-09-25","2018-09-26","2018-09-27","2018-09-28","2018-09-29","2018-09-30","2018-10-01","2018-10-02","2018-10-03","2018-10-04","2018-10-05","2018-10-06","2018-10-07","2018-10-08","2018-10-09","2018-10-10","2018-10-11","2018-10-12","2018-10-13","2018-10-14","2018-10-15","2018-10-16","2018-10-17","2018-10-18","2018-10-19","2018-10-20","2018-10-21","2018-10-22","2018-10-23","2018-10-24","2018-10-25","2018-10-26","2018-10-27","2018-10-28","2018-10-29","2018-10-30","2018-10-31","2018-11-01","2018-11-02","2018-11-03","2018-11-04","2018-11-05","2018-11-06","2018-11-07","2018-11-08","2018-11-09","2018-11-10","2018-11-11","2018-11-12","2018-11-13","2018-11-14","2018-11-15","2018-11-16","2018-11-17","2018-11-18","2018-11-19","2018-11-20","2018-11-21","2018-11-22","2018-11-23","2018-11-24","2018-11-25","2018-11-26","2018-11-27","2018-11-28","2018-11-29","2018-11-30","2018-12-01","2018-12-02","2018-12-03","2018-12-04","2018-12-05","2018-12-06","2018-12-07","2018-12-08","2018-12-09","2018-12-10","2018-12-11","2018-12-12","2018-12-13","2018-12-14","2018-12-15","2018-12-16","2018-12-17","2018-12-18","2018-12-19","2018-12-20","2018-12-21","2018-12-22","2018-12-23","2018-12-24","2018-12-25","2018-12-26","2018-12-27","2018-12-28","2018-12-29","2018-12-30","2018-12-31","2019-01-01","2019-01-02","2019-01-03","2019-01-04","2019-01-05","2019-01-06","2019-01-07","2019-01-08","2019-01-09","2019-01-10","2019-01-11","2019-01-12","2019-01-13","2019-01-14","2019-01-15","2019-01-16","2019-01-17","2019-01-18","2019-01-19","2019-01-20","2019-01-21","2019-01-22","2019-01-23","2019-01-24","2019-01-25","2019-01-26","2019-01-27","2019-01-28","2019-01-29","2019-01-30","2019-01-31","2019-02-01","2019-02-02","2019-02-03","2019-02-04","2019-02-05","2019-02-06","2019-02-07","2019-02-08","2019-02-09","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2019-02-16","2019-02-17","2019-02-18","2019-02-19","2019-02-20","2019-02-21","2019-02-22","2019-02-23","2019-02-24","2019-02-25","2019-02-26","2019-02-27","2019-02-28","2019-03-01","2019-03-02","2019-03-03","2019-03-04","2019-03-05","2019-03-06","2019-03-07","2019-03-08","2019-03-09","2019-03-10","2019-03-11","2019-03-12","2019-03-13","2019-03-14","2019-03-15","2019-03-16","2019-03-17","2019-03-18","2019-03-19","2019-03-20","2019-03-21","2019-03-22","2019-03-23","2019-03-24","2019-03-25","2019-03-26","2019-03-27","2019-03-28","2019-03-29","2019-03-30","2019-03-31","2019-04-01","2019-04-02","2019-04-03","2019-04-04","2019-04-05","2019-04-06","2019-04-07","2019-04-08","2019-04-09","2019-04-10","2019-04-11","2019-04-12","2019-04-13","2019-04-14","2019-04-15","2019-04-16","2019-04-17","2019-04-18","2019-04-19","2019-04-20","2019-04-21","2019-04-22","2019-04-23","2019-04-24","2019-04-25","2019-04-26","2019-04-27","2019-04-28","2019-04-29","2019-04-30","2019-05-01","2019-05-02","2019-05-03","2019-05-04","2019-05-05","2019-05-06","2019-05-07","2019-05-08","2019-05-09","2019-05-10","2019-05-11","2019-05-12","2019-05-13","2019-05-14","2019-05-15","2019-05-16","2019-05-17","2019-05-18","2019-05-19","2019-05-20","2019-05-21","2019-05-22","2019-05-23","2019-05-24","2019-05-25","2019-05-26","2019-05-27","2019-05-28","2019-05-29","2019-05-30","2019-05-31","2019-06-01","2019-06-02","2019-06-03","2019-06-04","2019-06-05","2019-06-06","2019-06-07","2019-06-08","2019-06-09","2019-06-10","2019-06-11","2019-06-12","2019-06-13","2019-06-14","2019-06-15","2019-06-16","2019-06-17","2019-06-18","2019-06-19","2019-06-20","2019-06-21","2019-06-22","2019-06-23","2019-06-24","2019-06-25","2019-06-26","2019-06-27","2019-06-28","2019-06-29","2019-06-30","2019-07-01","2019-07-02","2019-07-03","2019-07-04","2019-07-05","2019-07-06","2019-07-07","2019-07-08","2019-07-09","2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15","2019-07-16","2019-07-17","2019-07-18","2019-07-19","2019-07-20","2019-07-21","2019-07-22","2019-07-23","2019-07-24","2019-07-25","2019-07-26","2019-07-27","2019-07-28","2019-07-29","2019-07-30","2019-07-31","2019-08-01","2019-08-02","2019-08-03","2019-08-04","2019-08-05","2019-08-06","2019-08-07","2019-08-08","2019-08-09","2019-08-10","2019-08-11","2019-08-12","2019-08-13","2019-08-14","2019-08-15","2019-08-16","2019-08-17","2019-08-18","2019-08-19","2019-08-20","2019-08-21","2019-08-22","2019-08-23","2019-08-24","2019-08-25","2019-08-26","2019-08-27","2019-08-28","2019-08-29","2019-08-30","2019-08-31","2019-09-01","2019-09-02","2019-09-03","2019-09-04","2019-09-05","2019-09-06","2019-09-07","2019-09-08","2019-09-09","2019-09-10","2019-09-11","2019-09-12","2019-09-13","2019-09-14","2019-09-15","2019-09-16","2019-09-17","2019-09-18","2019-09-19","2019-09-20","2019-09-21","2019-09-22","2019-09-23","2019-09-24","2019-09-25","2019-09-26","2019-09-27","2019-09-28","2019-09-29","2019-09-30","2019-10-01","2019-10-02","2019-10-03","2019-10-04","2019-10-05","2019-10-06","2019-10-07","2019-10-08","2019-10-09","2019-10-10","2019-10-11","2019-10-12","2019-10-13","2019-10-14","2019-10-15","2019-10-16","2019-10-17","2019-10-18","2019-10-19","2019-10-20","2019-10-21","2019-10-22","2019-10-23","2019-10-24","2019-10-25","2019-10-26","2019-10-27","2019-10-28","2019-10-29","2019-10-30","2019-10-31","2019-11-01","2019-11-02","2019-11-03","2019-11-04","2019-11-05","2019-11-06","2019-11-07","2019-11-08","2019-11-09","2019-11-10","2019-11-11","2019-11-12","2019-11-13","2019-11-14","2019-11-15","2019-11-16","2019-11-17","2019-11-18","2019-11-19","2019-11-20","2019-11-21","2019-11-22","2019-11-23","2019-11-24","2019-11-25","2019-11-26","2019-11-27","2019-11-28","2019-11-29","2019-11-30","2019-12-01","2019-12-02","2019-12-03","2019-12-04","2019-12-05","2019-12-06","2019-12-07","2019-12-08","2019-12-09","2019-12-10","2019-12-11","2019-12-12","2019-12-13","2019-12-14","2019-12-15","2019-12-16","2019-12-17","2019-12-18","2019-12-19","2019-12-20","2019-12-21","2019-12-22","2019-12-23","2019-12-24","2019-12-25","2019-12-26","2019-12-27","2019-12-28","2019-12-29","2019-12-30","2019-12-31","2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05","2020-01-06","2020-01-07","2020-01-08","2020-01-09","2020-01-10","2020-01-11","2020-01-12","2020-01-13","2020-01-14","2020-01-15","2020-01-16","2020-01-17","2020-01-18","2020-01-19","2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-25","2020-01-26","2020-01-27","2020-01-28","2020-01-29","2020-01-30","2020-01-31","2020-02-01","2020-02-02","2020-02-03","2020-02-04","2020-02-05","2020-02-06","2020-02-07","2020-02-08","2020-02-09","2020-02-10","2020-02-11","2020-02-12","2020-02-13","2020-02-14","2020-02-15","2020-02-16","2020-02-17","2020-02-18","2020-02-19","2020-02-20","2020-02-21","2020-02-22","2020-02-23","2020-02-24","2020-02-25","2020-02-26","2020-02-27","2020-02-28","2020-02-29","2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05","2020-03-06","2020-03-07","2020-03-08","2020-03-09","2020-03-10","2020-03-11","2020-03-12","2020-03-13","2020-03-14","2020-03-15","2020-03-16","2020-03-17","2020-03-18","2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01","2020-04-02","2020-04-03","2020-04-04","2020-04-05","2020-04-06","2020-04-07","2020-04-08","2020-04-09","2020-04-10","2020-04-11","2020-04-12","2020-04-13","2020-04-14","2020-04-15","2020-04-16","2020-04-17","2020-04-18","2020-04-19","2020-04-20","2020-04-21","2020-04-22","2020-04-23","2020-04-24","2020-04-25","2020-04-26","2020-04-27","2020-04-28","2020-04-29","2020-04-30","2020-05-01","2020-05-02","2020-05-03","2020-05-04","2020-05-05","2020-05-06","2020-05-07","2020-05-08","2020-05-09","2020-05-10","2020-05-11","2020-05-12","2020-05-13","2020-05-14","2020-05-15","2020-05-16","2020-05-17","2020-05-18","2020-05-19","2020-05-20","2020-05-21","2020-05-22","2020-05-23","2020-05-24","2020-05-25","2020-05-26","2020-05-27","2020-05-28","2020-05-29","2020-05-30","2020-05-31","2020-06-01","2020-06-02","2020-06-03","2020-06-04","2020-06-05","2020-06-06","2020-06-07","2020-06-08","2020-06-09","2020-06-10","2020-06-11","2020-06-12","2020-06-13","2020-06-14","2020-06-15","2020-06-16","2020-06-17","2020-06-18","2020-06-19","2020-06-20","2020-06-21","2020-06-22","2020-06-23","2020-06-24","2020-06-25","2020-06-26","2020-06-27","2020-06-28","2020-06-29","2020-06-30","2020-07-01","2020-07-02","2020-07-03","2020-07-04","2020-07-05","2020-07-06","2020-07-07","2020-07-08","2020-07-09","2020-07-10","2020-07-11","2020-07-12","2020-07-13","2020-07-14","2020-07-15","2020-07-16","2020-07-17","2020-07-18","2020-07-19","2020-07-20","2020-07-21","2020-07-22","2020-07-23","2020-07-24","2020-07-25","2020-07-26","2020-07-27","2020-07-28","2020-07-29","2020-07-30","2020-07-31","2020-08-01","2020-08-02","2020-08-03","2020-08-04","2020-08-05","2020-08-06","2020-08-07","2020-08-08","2020-08-09","2020-08-10","2020-08-11","2020-08-12","2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17","2020-08-18","2020-08-19","2020-08-20","2020-08-21","2020-08-22","2020-08-23","2020-08-24","2020-08-25","2020-08-26","2020-08-27","2020-08-28","2020-08-29","2020-08-30","2020-08-31","2020-09-01","2020-09-02","2020-09-03","2020-09-04","2020-09-05","2020-09-06","2020-09-07","2020-09-08","2020-09-09","2020-09-10","2020-09-11","2020-09-12","2020-09-13","2020-09-14","2020-09-15","2020-09-16","2020-09-17","2020-09-18","2020-09-19","2020-09-20","2020-09-21","2020-09-22","2020-09-23","2020-09-24","2020-09-25","2020-09-26","2020-09-27","2020-09-28","2020-09-29","2020-09-30","2020-10-01","2020-10-02","2020-10-03","2020-10-04","2020-10-05","2020-10-06","2020-10-07","2020-10-08","2020-10-09","2020-10-10","2020-10-11","2020-10-12","2020-10-13","2020-10-14","2020-10-15","2020-10-16","2020-10-17","2020-10-18","2020-10-19","2020-10-20","2020-10-21","2020-10-22","2020-10-23","2020-10-24","2020-10-25","2020-10-26","2020-10-27","2020-10-28","2020-10-29","2020-10-30","2020-10-31","2020-11-01","2020-11-02","2020-11-03","2020-11-04","2020-11-05","2020-11-06","2020-11-07","2020-11-08","2020-11-09","2020-11-10","2020-11-11","2020-11-12","2020-11-13","2020-11-14","2020-11-15","2020-11-16","2020-11-17","2020-11-18","2020-11-19","2020-11-20","2020-11-21","2020-11-22","2020-11-23","2020-11-24","2020-11-25","2020-11-26","2020-11-27","2020-11-28","2020-11-29","2020-11-30","2020-12-01","2020-12-02","2020-12-03","2020-12-04","2020-12-05","2020-12-06","2020-12-07","2020-12-08","2020-12-09","2020-12-10","2020-12-11","2020-12-12","2020-12-13","2020-12-14","2020-12-15","2020-12-16","2020-12-17","2020-12-18","2020-12-19","2020-12-20","2020-12-21","2020-12-22","2020-12-23","2020-12-24","2020-12-25","2020-12-26","2020-12-27","2020-12-28","2020-12-29","2020-12-30","2020-12-31","2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05","2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10","2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15","2021-01-16","2021-01-17","2021-01-18","2021-01-19","2021-01-20","2021-01-21","2021-01-22","2021-01-23","2021-01-24","2021-01-25","2021-01-26","2021-01-27","2021-01-28","2021-01-29","2021-01-30","2021-01-31","2021-02-01","2021-02-02","2021-02-03","2021-02-04","2021-02-05","2021-02-06","2021-02-07","2021-02-08","2021-02-09","2021-02-10","2021-02-11","2021-02-12","2021-02-13","2021-02-14","2021-02-15","2021-02-16","2021-02-17","2021-02-18","2021-02-19","2021-02-20","2021-02-21","2021-02-22","2021-02-23","2021-02-24","2021-02-25","2021-02-26","2021-02-27","2021-02-28","2021-03-01","2021-03-02","2021-03-03","2021-03-04","2021-03-05","2021-03-06","2021-03-07","2021-03-08","2021-03-09","2021-03-10","2021-03-11","2021-03-12","2021-03-13","2021-03-14","2021-03-15","2021-03-16","2021-03-17","2021-03-18","2021-03-19","2021-03-20","2021-03-21","2021-03-22","2021-03-23","2021-03-24","2021-03-25","2021-03-26","2021-03-27","2021-03-28","2021-03-29","2021-03-30","2021-03-31","2021-04-01","2021-04-02","2021-04-03","2021-04-04","2021-04-05","2021-04-06","2021-04-07","2021-04-08","2021-04-09","2021-04-10","2021-04-11","2021-04-12","2021-04-13","2021-04-14","2021-04-15","2021-04-16","2021-04-17","2021-04-18","2021-04-19","2021-04-20","2021-04-21","2021-04-22","2021-04-23","2021-04-24","2021-04-25","2021-04-26","2021-04-27","2021-04-28","2021-04-29","2021-04-30","2021-05-01","2021-05-02","2021-05-03","2021-05-04","2021-05-05","2021-05-06","2021-05-07","2021-05-08","2021-05-09","2021-05-10","2021-05-11","2021-05-12","2021-05-13","2021-05-14","2021-05-15","2021-05-16","2021-05-17","2021-05-18","2021-05-19","2021-05-20","2021-05-21","2021-05-22","2021-05-23","2021-05-24","2021-05-25","2021-05-26","2021-05-27","2021-05-28","2021-05-29","2021-05-30","2021-05-31","2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07","2021-06-08","2021-06-09","2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29","2021-06-30","2021-07-01","2021-07-02","2021-07-03","2021-07-04","2021-07-05","2021-07-06","2021-07-07","2021-07-08","2021-07-09","2021-07-10","2021-07-11","2021-07-12","2021-07-13","2021-07-14","2021-07-15","2021-07-16","2021-07-17","2021-07-18","2021-07-19","2021-07-20","2021-07-21","2021-07-22","2021-07-23","2021-07-24","2021-07-25","2021-07-26","2021-07-27","2021-07-28","2021-07-29","2021-07-30","2021-07-31","2021-08-01","2021-08-02","2021-08-03","2021-08-04","2021-08-05","2021-08-06","2021-08-07","2021-08-08","2021-08-09","2021-08-10","2021-08-11","2021-08-12","2021-08-13","2021-08-14","2021-08-15","2021-08-16","2021-08-17","2021-08-18","2021-08-19","2021-08-20","2021-08-21","2021-08-22","2021-08-23","2021-08-24","2021-08-25","2021-08-26","2021-08-27","2021-08-28","2021-08-29","2021-08-30","2021-08-31","2021-09-01","2021-09-02","2021-09-03","2021-09-04","2021-09-05","2021-09-06","2021-09-07","2021-09-08","2021-09-09","2021-09-10","2021-09-11","2021-09-12","2021-09-13","2021-09-14","2021-09-15","2021-09-16","2021-09-17","2021-09-18","2021-09-19","2021-09-20","2021-09-21","2021-09-22","2021-09-23","2021-09-24","2021-09-25","2021-09-26","2021-09-27","2021-09-28","2021-09-29","2021-09-30","2021-10-01","2021-10-02","2021-10-03","2021-10-04","2021-10-05","2021-10-06","2021-10-07","2021-10-08","2021-10-09","2021-10-10","2021-10-11","2021-10-12","2021-10-13","2021-10-14","2021-10-15","2021-10-16","2021-10-17","2021-10-18","2021-10-19","2021-10-20","2021-10-21","2021-10-22","2021-10-23","2021-10-24","2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29","2021-10-30","2021-10-31","2021-11-01","2021-11-02","2021-11-03","2021-11-04","2021-11-05","2021-11-06","2021-11-07","2021-11-08","2021-11-09","2021-11-10","2021-11-11","2021-11-12","2021-11-13","2021-11-14","2021-11-15","2021-11-16","2021-11-17","2021-11-18","2021-11-19","2021-11-20","2021-11-21","2021-11-22","2021-11-23","2021-11-24","2021-11-25","2021-11-26","2021-11-27","2021-11-28","2021-11-29","2021-11-30","2021-12-01","2021-12-02","2021-12-03","2021-12-04","2021-12-05","2021-12-06","2021-12-07","2021-12-08","2021-12-09","2021-12-10","2021-12-11","2021-12-12","2021-12-13","2021-12-14","2021-12-15","2021-12-16","2021-12-17","2021-12-18","2021-12-19","2021-12-20","2021-12-21","2021-12-22","2021-12-23","2021-12-24","2021-12-25","2021-12-26","2021-12-27","2021-12-28","2021-12-29","2021-12-30","2021-12-31","2022-01-01","2022-01-02","2022-01-03","2022-01-04","2022-01-05","2022-01-06","2022-01-07","2022-01-08","2022-01-09","2022-01-10","2022-01-11","2022-01-12","2022-01-13","2022-01-14","2022-01-15","2022-01-16","2022-01-17","2022-01-18","2022-01-19","2022-01-20","2022-01-21","2022-01-22","2022-01-23","2022-01-24","2022-01-25","2022-01-26","2022-01-27","2022-01-28","2022-01-29","2022-01-30","2022-01-31","2022-02-01","2022-02-02","2022-02-03","2022-02-04","2022-02-05","2022-02-06","2022-02-07","2022-02-08","2022-02-09","2022-02-10","2022-02-11","2022-02-12","2022-02-13","2022-02-14","2022-02-15","2022-02-16","2022-02-17","2022-02-18","2022-02-19","2022-02-20","2022-02-21","2022-02-22","2022-02-23","2022-02-24","2022-02-25","2022-02-26","2022-02-27","2022-02-28","2022-03-01","2022-03-02","2022-03-03","2022-03-04","2022-03-05","2022-03-06","2022-03-07","2022-03-08","2022-03-09","2022-03-10","2022-03-11","2022-03-12","2022-03-13","2022-03-14","2022-03-15","2022-03-16","2022-03-17","2022-03-18","2022-03-19","2022-03-20","2022-03-21","2022-03-22","2022-03-23","2022-03-24","2022-03-25","2022-03-26","2022-03-27","2022-03-28","2022-03-29","2022-03-30","2022-03-31","2022-04-01","2022-04-02","2022-04-03","2022-04-04","2022-04-05","2022-04-06","2022-04-07","2022-04-08","2022-04-09","2022-04-10","2022-04-11","2022-04-12","2022-04-13","2022-04-14","2022-04-15","2022-04-16","2022-04-17","2022-04-18","2022-04-19","2022-04-20","2022-04-21","2022-04-22","2022-04-23","2022-04-24","2022-04-25","2022-04-26","2022-04-27","2022-04-28","2022-04-29","2022-04-30","2022-05-01","2022-05-02","2022-05-03","2022-05-04","2022-05-05","2022-05-06","2022-05-07","2022-05-08","2022-05-09","2022-05-10","2022-05-11","2022-05-12","2022-05-13","2022-05-14","2022-05-15","2022-05-16","2022-05-17","2022-05-18","2022-05-19","2022-05-20","2022-05-21","2022-05-22","2022-05-23","2022-05-24","2022-05-25","2022-05-26","2022-05-27","2022-05-28","2022-05-29","2022-05-30","2022-05-31","2022-06-01","2022-06-02","2022-06-03","2022-06-04","2022-06-05","2022-06-06","2022-06-07","2022-06-08","2022-06-09","2022-06-10","2022-06-11","2022-06-12","2022-06-13","2022-06-14","2022-06-15","2022-06-16","2022-06-17","2022-06-18","2022-06-19","2022-06-20","2022-06-21","2022-06-22","2022-06-23","2022-06-24","2022-06-25","2022-06-26","2022-06-27","2022-06-28","2022-06-29","2022-06-30","2022-07-01","2022-07-02","2022-07-03","2022-07-04","2022-07-05","2022-07-06","2022-07-07","2022-07-08","2022-07-09","2022-07-10","2022-07-11","2022-07-12","2022-07-13","2022-07-14","2022-07-15","2022-07-16","2022-07-17","2022-07-18","2022-07-19","2022-07-20","2022-07-21","2022-07-22","2022-07-23","2022-07-24","2022-07-25","2022-07-26","2022-07-27","2022-07-28","2022-07-29","2022-07-30","2022-07-31","2022-08-01","2022-08-02","2022-08-03","2022-08-04","2022-08-05","2022-08-06","2022-08-07","2022-08-08","2022-08-09","2022-08-10","2022-08-11","2022-08-12","2022-08-13","2022-08-14","2022-08-15","2022-08-16","2022-08-17","2022-08-18","2022-08-19","2022-08-20","2022-08-21","2022-08-22","2022-08-23","2022-08-24","2022-08-25","2022-08-26","2022-08-27","2022-08-28","2022-08-29","2022-08-30","2022-08-31","2022-09-01","2022-09-02","2022-09-03","2022-09-04","2022-09-05","2022-09-06","2022-09-07","2022-09-08","2022-09-09","2022-09-10","2022-09-11","2022-09-12","2022-09-13","2022-09-14","2022-09-15","2022-09-16","2022-09-17","2022-09-18","2022-09-19","2022-09-20","2022-09-21","2022-09-22","2022-09-23","2022-09-24","2022-09-25","2022-09-26","2022-09-27","2022-09-28","2022-09-29","2022-09-30","2022-10-01","2022-10-02","2022-10-03","2022-10-04","2022-10-05","2022-10-06","2022-10-07","2022-10-08","2022-10-09","2022-10-10","2022-10-11","2022-10-12","2022-10-13","2022-10-14","2022-10-15","2022-10-16","2022-10-17","2022-10-18","2022-10-19","2022-10-20","2022-10-21","2022-10-22","2022-10-23","2022-10-24","2022-10-25","2022-10-26","2022-10-27","2022-10-28","2022-10-29","2022-10-30","2022-10-31","2022-11-01","2022-11-02","2022-11-03","2022-11-04","2022-11-05","2022-11-06","2022-11-07","2022-11-08","2022-11-09","2022-11-10","2022-11-11","2022-11-12","2022-11-13","2022-11-14","2022-11-15","2022-11-16","2022-11-17","2022-11-18","2022-11-19","2022-11-20","2022-11-21","2022-11-22","2022-11-23","2022-11-24","2022-11-25","2022-11-26","2022-11-27","2022-11-28","2022-11-29","2022-11-30","2022-12-01","2022-12-02","2022-12-03","2022-12-04","2022-12-05","2022-12-06","2022-12-07","2022-12-08","2022-12-09","2022-12-10","2022-12-11","2022-12-12","2022-12-13","2022-12-14","2022-12-15","2022-12-16","2022-12-17","2022-12-18","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-24","2022-12-25","2022-12-26","2022-12-27","2022-12-28","2022-12-29","2022-12-30","2022-12-31","2023-01-01","2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-07","2023-01-08","2023-01-09","2023-01-10","2023-01-11","2023-01-12","2023-01-13","2023-01-14","2023-01-15","2023-01-16","2023-01-17","2023-01-18","2023-01-19","2023-01-20","2023-01-21","2023-01-22","2023-01-23","2023-01-24","2023-01-25","2023-01-26","2023-01-27","2023-01-28","2023-01-29","2023-01-30","2023-01-31","2023-02-01","2023-02-02","2023-02-03","2023-02-04","2023-02-05","2023-02-06","2023-02-07","2023-02-08","2023-02-09","2023-02-10","2023-02-11","2023-02-12","2023-02-13","2023-02-14","2023-02-15","2023-02-16","2023-02-17","2023-02-18","2023-02-19","2023-02-20","2023-02-21","2023-02-22","2023-02-23","2023-02-24","2023-02-25","2023-02-26","2023-02-27","2023-02-28","2023-03-01","2023-03-02","2023-03-03","2023-03-04","2023-03-05","2023-03-06","2023-03-07","2023-03-08","2023-03-09","2023-03-10","2023-03-11","2023-03-12","2023-03-13","2023-03-14","2023-03-15","2023-03-16","2023-03-17","2023-03-18","2023-03-19","2023-03-20","2023-03-21","2023-03-22","2023-03-23","2023-03-24","2023-03-25","2023-03-26","2023-03-27","2023-03-28"],"y":[1421.6,1452.82,1490.09,1537.67,1555.45,1578.8,1596.71,1723.35,1755.36,1787.13,1848.57,1724.24,1804.91,1808.91,1738.43,1734.45,1839.09,1888.65,1987.71,2084.73,2041.2,2173.4,2320.42,2443.64,2304.98,2202.42,2038.87,2155.8,2255.61,2175.47,2286.41,2407.88,2488.55,2515.35,2511.81,2686.81,2863.2,2732.16,2805.62,2823.81,2947.71,2958.11,2659.63,2717.02,2506.37,2464.58,2518.56,2655.88,2548.29,2589.6,2721.79,2689.1,2705.41,2744.91,2608.72,2589.41,2478.45,2552.45,2574.79,2539.32,2480.84,2434.55,2506.47,2564.06,2601.64,2601.99,2608.56,2518.66,2571.34,2518.44,2372.56,2337.79,2398.84,2357.9,2233.34,1998.86,1929.82,2228.41,2318.88,2273.43,2817.6,2667.76,2810.12,2730.4,2754.86,2576.48,2529.45,2671.78,2809.01,2726.45,2757.18,2875.34,2718.26,2710.67,2804.73,2895.89,3252.91,3213.94,3378.94,3419.94,3342.47,3381.28,3650.62,3884.71,4073.26,4325.13,4181.93,4376.63,4331.69,4160.62,4193.7,4087.66,4001.74,4100.52,4151.52,4334.68,4371.6,4352.4,4382.88,4382.66,4579.02,4565.3,4703.39,4892.01,4578.77,4582.96,4236.31,4376.53,4597.12,4599.88,4228.75,4226.06,4122.94,4161.27,4130.81,3882.59,3154.95,3637.52,3625.04,3582.88,4065.2,3924.97,3905.95,3631.04,3630.7,3792.4,3682.84,3926.07,3892.35,4200.67,4174.73,4163.07,4338.71,4403.74,4409.32,4317.48,4229.36,4328.41,4370.81,4426.89,4610.48,4772.02,4781.99,4826.48,5446.91,5647.21,5831.79,5678.19,5725.59,5605.51,5590.69,5708.52,6011.45,6031.6,6008.42,5930.32,5526.64,5750.8,5904.83,5780.9,5753.09,6153.85,6130.53,6468.4,6767.31,7078.5,7207.76,7379.95,7407.41,7022.76,7144.38,7459.69,7143.58,6618.14,6357.6,5950.07,6559.49,6635.75,7315.54,7871.69,7708.99,7790.15,8036.49,8200.64,8071.26,8253.55,8038.77,8253.69,8790.92,9330.55,9818.35,10058.8,9888.61,10233.6,10975.6,11074.6,11323.2,11657.2,11916.7,14291.5,17899.699,16569.4,15178.2,15455.4,16936.801,17415.4,16408.199,16564.0,17706.9,19497.4,19140.801,19114.199,17776.699,16624.6,15802.9,13831.8,14699.2,13925.8,14026.6,16099.8,15838.5,14606.5,14656.2,12952.2,14156.4,13657.2,14982.1,15201.0,15599.2,17429.5,17527.0,16477.6,15170.1,14595.4,14973.3,13405.8,13980.6,14360.2,13772.0,13819.8,11490.5,11188.6,11474.9,11607.4,12899.2,11600.1,10931.4,10868.4,11359.4,11259.4,11171.4,11440.7,11786.3,11296.4,10106.3,10221.1,9170.54,8830.75,9174.91,8277.01,6955.27,7754.0,7621.3,8265.59,8736.98,8621.9,8129.97,8926.57,8598.31,9494.63,10166.4,10233.9,11112.7,10551.8,11225.3,11403.7,10690.4,10005.0,10301.1,9813.07,9664.73,10366.7,10725.6,10397.9,10951.0,11086.4,11489.7,11512.6,11573.3,10779.9,9965.57,9395.01,9337.55,8866.0,9578.63,9205.12,9194.85,8269.81,8300.86,8338.35,7916.88,8223.68,8630.65,8913.47,8929.28,8728.47,8879.62,8668.12,8495.78,8209.4,7833.04,7954.48,7165.7,6890.52,6973.53,6844.23,7083.8,7456.11,6853.84,6811.47,6636.32,6911.09,7023.52,6770.73,6834.76,6968.32,7889.25,7895.96,7986.24,8329.11,8058.67,7902.09,8163.42,8294.31,8845.83,8895.58,8802.46,8930.88,9697.5,8845.74,9281.51,8987.05,9348.48,9419.08,9240.55,9119.01,9235.92,9743.86,9700.76,9858.15,9654.8,9373.01,9234.82,9325.18,9043.94,8441.49,8504.89,8723.94,8716.79,8510.38,8368.83,8094.32,8250.97,8247.18,8513.25,8418.99,8041.78,7557.82,7587.34,7480.14,7355.88,7368.22,7135.99,7472.59,7406.52,7494.17,7541.45,7643.45,7720.25,7514.47,7633.76,7653.98,7678.24,7624.92,7531.98,6786.02,6906.92,6582.36,6349.9,6675.35,6456.58,6550.16,6499.27,6734.82,6769.94,6776.55,6729.74,6083.69,6162.48,6173.23,6249.18,6093.67,6157.13,5903.44,6218.3,6404.0,6385.82,6614.18,6529.59,6597.55,6639.14,6673.5,6856.93,6773.88,6741.75,6329.95,6394.71,6228.81,6238.05,6276.12,6359.64,6741.75,7321.04,7370.78,7466.86,7354.13,7419.29,7418.49,7711.11,8424.27,8181.39,7951.58,8165.01,8192.15,8218.46,8180.48,7780.44,7624.91,7567.15,7434.39,7032.85,7068.48,6951.8,6753.12,6305.8,6568.23,6184.71,6295.73,6322.69,6297.57,6199.71,6308.52,6334.73,6580.63,6423.76,6506.07,6308.53,6488.76,6376.71,6534.88,6719.96,6763.19,6707.26,6884.64,7096.28,7047.16,6978.23,7037.58,7193.25,7272.72,7260.06,7361.66,6792.83,6529.17,6467.07,6225.98,6300.86,6329.7,6321.2,6351.8,6517.31,6512.71,6543.2,6517.18,6281.2,6371.3,6398.54,6519.67,6734.95,6721.98,6710.63,6595.41,6446.47,6495.0,6676.75,6644.13,6601.96,6625.56,6589.62,6556.1,6502.59,6576.69,6622.48,6588.31,6602.95,6652.23,6642.64,6585.53,6256.24,6274.58,6285.99,6290.93,6596.54,6596.11,6544.43,6476.71,6465.41,6489.19,6482.35,6487.16,6475.74,6495.84,6476.29,6474.75,6480.38,6486.39,6332.63,6334.27,6317.61,6377.78,6388.44,6361.26,6376.13,6419.66,6461.01,6530.14,6453.72,6385.62,6409.22,6411.27,6371.27,6359.49,5738.35,5648.03,5575.55,5554.33,5623.54,4871.49,4451.87,4602.17,4365.94,4347.11,3880.76,4009.97,3779.13,3820.72,4257.42,4278.847,4017.269,4214.6720000000005,4139.878,3894.131,3956.894,3753.995,3521.1020000000003,3419.937,3476.115,3614.234,3502.656,3424.588,3486.95,3313.677,3242.485,3236.762,3252.8390000000004,3545.865,3696.059,3745.951,4134.441,3896.544,4014.183,3998.98,4078.599,3815.491,3857.298,3654.833,3923.919,3820.409,3865.953,3742.7,3843.52,3943.409,3836.741,3857.718,3845.195,4076.633,4025.248,4030.848,4035.296,3678.925,3687.365,3661.301,3552.953,3706.052,3630.675,3655.007,3678.564,3657.839,3728.568,3601.014,3576.032,3604.577,3585.123,3600.865,3599.766,3602.46,3583.966,3470.45,3448.117,3486.182,3457.793,3487.945,3521.061,3464.013,3459.1540000000005,3466.357,3413.768,3399.472,3666.78,3671.204,3690.188,3648.431,3653.529000000001,3632.071,3616.881,3620.811,3629.788,3673.836,3915.714,3947.094,3999.821,3954.118,4005.527,4142.527,3810.427,3882.696,3854.358,3851.047,3854.785,3859.584,3864.415,3847.176,3761.557,3896.375,3903.943,3911.484,3901.132,3963.314,3951.6,3905.227,3909.156,3906.717,3924.369,3960.911,4048.726,4025.229,4032.507,4071.19,4087.476,4029.327,4023.968,4035.826,4022.168,3963.071,3985.081,4087.066,4069.107,4098.375,4106.66,4105.404,4158.183,4879.878,4973.022,4922.799,5036.681,5059.817,5198.897,5289.771,5204.958,5324.552,5064.488,5089.539000000001,5096.585999999999,5167.722,5067.108,5235.56,5251.938,5298.386,5303.812,5337.886,5314.531,5399.365,5572.361999999999,5464.866999999999,5210.516,5279.348000000001,5268.291,5285.139,5247.353,5350.727,5402.696999999999,5505.284000000001,5768.29,5831.167,5795.7080000000005,5746.807,5829.501,5982.4580000000005,6174.529,6378.849,7204.771,6972.371999999999,7814.915,7994.416,8205.168,7884.909000000001,7343.896,7271.208,8197.689,7978.309,7963.328,7680.066,7881.847,7987.371999999999,8052.544,8673.216,8805.778,8719.962,8659.487,8319.473,8574.502,8564.017,8742.958,8208.995,7707.771,7824.231,7822.023,8043.951,7954.128000000001,7688.076999999999,8000.33,7927.714,8145.857,8230.924,8693.833,8838.375,8994.488000000001,9320.353,9081.763,9273.521,9527.16,10144.557,10701.691,10855.371,11011.103,11790.917,13016.231000000002,11182.807,12407.332,11959.371,10817.155,10583.135,10801.678,11961.27,11215.438,10978.46,11208.551,11450.847,12285.958,12573.812,12156.513,11358.662,11815.986,11392.379,10256.059,10895.09,9477.642,9693.803,10666.482,10530.732,10767.14,10599.105,10343.106,9900.768,9811.926,9911.842,9870.304,9477.678,9552.86,9519.146,9607.424,10085.628,10399.669,10518.175,10821.727,10970.185,11805.653,11478.169,11941.969,11966.407,11862.937,11354.024,11523.579,11382.616000000002,10895.83,10051.704,10311.546,10374.339,10231.744,10345.811,10916.053999999998,10763.232,10138.05,10131.056,10407.965,10159.961,10138.518,10370.82,10185.5,9754.423,9510.2,9598.174,9630.664,9757.971,10346.761,10623.54,10594.493,10575.533,10353.303,10517.255,10441.276,10334.975,10115.976,10178.372,10410.127,10360.547,10358.049,10347.713,10276.794,10241.272,10198.248,10266.415,10181.642,10019.717,10070.393,9729.324,8620.565999999999,8486.993,8118.968000000001,8251.846,8245.915,8104.186,8293.868,8343.276,8393.042,8259.992,8205.939,8151.5,7988.156,8245.623,8228.783000000001,8595.74,8586.474,8321.757,8336.556,8321.006,8374.687,8205.369,8047.526999999999,8103.911,7973.208,7988.561,8222.078000000001,8243.721,8078.203,7514.672,7493.489,8660.7,9244.973,9551.715,9256.148,9427.688,9205.727,9199.585,9261.104,9324.718,9235.354,9412.612,9342.527,9360.88,9267.562,8804.881,8813.582,9055.526,8757.788,8815.662,8808.263,8708.095,8491.992,8550.761,8577.976,8309.286,8206.146,8027.268,7642.75,7296.578,7397.797,7047.917,7146.134,7218.371,7531.664000000001,7463.106,7761.244000000001,7569.63,7424.291999999999,7321.988,7320.146,7252.035,7448.308000000001,7546.996999999999,7556.238,7564.345,7400.899,7278.12,7217.427,7243.134,7269.685,7124.674,7152.302,6932.48,6640.515,7276.803000000001,7202.844,7218.816,7191.159000000001,7511.589,7355.628000000001,7322.531999999999,7275.156,7238.967,7290.088000000001,7317.99,7422.653,7292.995,7193.599,7200.174,6985.47,7344.884,7410.656999999999,7411.316999999999,7769.219,8163.691999999999,8079.863,7879.071,8166.554,8037.538,8192.494,8144.194,8827.765,8807.011,8723.786,8929.038,8942.809000000001,8706.245,8657.643,8745.895,8680.876,8406.516,8445.435,8367.848,8596.83,8909.819,9358.59,9316.63,9508.993,9350.529,9392.875,9344.365,9293.521,9180.963,9613.424,9729.802,9795.943,9865.119,10116.674,9856.611,10208.236,10326.055,10214.38,10312.116,9889.425,9934.434,9690.143,10141.996,9633.387,9608.476,9686.441,9663.182,9924.516,9650.175,9341.705,8820.521999999999,8784.494,8672.455,8599.509,8562.454,8869.67,8787.786,8755.246,9078.763,9122.546,8909.954,8108.116,7923.645,7909.729,7911.43,4970.788,5563.706999999999,5200.366,5392.315,5014.48,5225.629,5238.438,6191.193,6198.778,6185.066,5830.255,6416.315,6734.804,6681.063,6716.44,6469.798000000001,6242.194,5922.043000000001,6429.842,6438.645,6606.776,6793.625,6733.387,6867.526999999999,6791.129,7271.781,7176.415,7334.099,7302.089,6865.493,6859.0830000000005,6971.092,6845.0380000000005,6842.428000000001,6642.11,7116.804,7096.185,7257.665,7189.425,6881.9580000000005,6880.323,7117.2080000000005,7429.725,7550.901,7569.936,7679.866999999999,7795.601,7807.059,8801.038,8658.554,8864.767,8988.597,8897.469000000001,8912.654,9003.07,9268.762,9951.519,9842.666,9593.896,8756.431,8601.796,8804.478000000001,9269.987,9733.722,9328.197,9377.014,9670.739,9726.575,9729.038,9522.981,9081.762,9182.577,9209.287,8790.368,8906.935,8835.053,9181.018,9525.751,9439.124,9700.414,9461.059,10167.269,9529.804,9656.718,9800.637,9665.533,9653.68,9758.853,9771.489,9795.7,9870.095,9321.781,9480.844,9475.277,9386.788,9450.702,9538.024,9480.255,9411.841,9288.019,9332.341,9303.63,9648.718,9629.658,9313.61,9264.813,9162.918,9045.391,9143.582,9190.854,9137.993,9228.325,9123.41,9087.304,9132.488,9073.942,9375.475,9252.277,9428.333,9277.968,9278.808,9240.347,9276.5,9243.614,9243.214,9192.837,9132.228,9151.393,9159.04,9185.817,9164.231,9374.888,9525.363,9581.072,9536.893,9677.113,9905.167,10990.873,10912.823,11100.468,11111.214,11323.467,11759.593,11053.614,11246.348999999998,11205.893,11747.022,11779.773,11601.473,11754.046,11675.739,11878.111,11410.525,11584.935,11784.138,11768.871,11865.698,11892.803999999998,12254.402,11991.233,11758.283,11878.372,11592.489,11681.825,11664.848,11774.596,11366.135,11488.363,11323.397,11542.5,11506.865,11711.506,11680.82,11970.479,11414.034,10245.297,10511.813,10169.567,10280.352,10369.563,10131.517,10242.348,10363.139,10400.915,10442.171,10323.756,10680.838,10796.951,10974.905,10948.99,10944.586,11094.347,10938.271,10462.26,10538.46,10246.187,10760.066,10692.717,10750.723999999998,10775.27,10709.652,10844.641,10784.491000000002,10619.452,10575.975,10549.329,10669.583,10793.34,10604.406,10668.969,10915.686000000002,11064.458,11296.361,11384.182,11555.363,11425.899,11429.507,11495.35,11322.123,11358.102,11483.359,11742.037,11916.335,12823.688999999998,12965.892,12931.539,13108.062,13031.174,13075.248,13654.219,13271.285,13437.883,13546.522,13780.995,13737.109,13550.489,13950.301,14133.707,15579.848999999998,15565.881,14833.754,15479.567,15332.315,15290.902,15701.34,16276.344,16317.809,16068.139,15955.588,16716.111,17645.406000000003,17804.006,17817.09,18621.314,18642.232,18370.002,18364.121,19107.465,18732.121,17150.623,17108.402,17717.414,18177.484,19625.836,18802.998,19201.092,19445.398,18699.766,19154.23,19345.121,19191.631,18321.145,18553.916,18264.992,18058.904,18803.656000000003,19142.383,19246.645,19417.076,21310.598,22805.162,23137.961,23869.832,23477.295,22803.082,23783.029,23241.346,23735.949,24664.791,26437.037,26272.295,27084.809,27362.438,28840.953,29001.721,29374.152,32127.268,32782.023,31971.914,33992.43,36824.363,39371.043,40797.609,40254.547,38356.441,35566.656,33922.961,37316.359,39187.328,36825.367,36178.141,35791.277,36630.074,36069.805,35547.75,30825.699,33005.762,32067.643,32289.379,32366.393,32569.85,30432.547,33466.098,34316.387,34269.523,33114.359,33537.176,35510.289,37472.09,36926.066,38144.309,39266.012,38903.441,46196.465,46481.105,44918.184,47909.332,47504.852,47105.516,48717.289,47945.059,49199.871,52149.008,51679.797,55888.133,56099.52,57539.945,54207.32,48824.426,49705.332,47093.852,46339.762,46188.453,45137.77,49631.242,48378.988,50538.242,48561.168,48927.305,48912.383,51206.691,52246.523,54824.117,56008.551,57805.121,57332.09,61243.086,59302.316,55907.199,56804.902,58870.895,57858.922,58346.652,58313.645,57523.422,54529.145,54738.945,52774.266,51704.16,55137.312000000005,55973.512,55950.746,57750.199,58917.691,58918.832,59095.809,59384.312000000005,57603.891,58758.555,59057.879,58192.359,56048.938,58323.953,58245.004,59793.234,60204.965,59893.453,63503.457,63109.695,63314.012,61572.789,60683.82,56216.184,55724.266,56473.031,53906.09,51762.273,51093.652,50050.867,49004.254,54021.754,55033.117,54824.703,53555.109,57750.176,57828.051,56631.078,57200.293,53333.539,57424.008,56396.516,57356.402,58803.777,58232.316,55859.797,56704.574,49150.535,49716.191,49880.535,46760.188,46456.059,43537.512,42909.402,37002.441,40782.738,37304.691,37536.633,34770.582,38705.98,38402.223,39294.199,38436.969,35697.605,34616.066,35678.129,37332.855,36684.926,37575.18,39208.766,36894.406,35551.957,35862.379,33560.707,33472.633,37345.121,36702.598,37334.398,35552.516,39097.859,40218.477,40406.27,38347.062000000005,38053.504,35787.246,35615.871,35698.297,31676.693,32505.66,33723.027,34662.437999999995,31637.779,32186.277,34649.645,34434.336,35867.777,35040.836,33572.117,33897.047,34668.547,35287.781,33746.004,34235.195,33855.328,32877.371,33798.012,33520.52,34240.187999999995,33155.848,32702.025,32822.348,31780.73,31421.539,31533.068,31796.811,30817.832,29807.348,32110.693,32313.105,33581.551,34292.445,35350.187999999995,37337.535,39406.941,39995.906,40008.422,42235.547,41626.195,39974.895,39201.945,38152.98,39747.504,40869.555,42816.5,44555.801,43798.117,46365.402,45585.031,45593.637,44428.289,47793.32,47096.945,47047.004,46004.484,44695.359,44801.188,46717.578,49339.176,48905.492,49321.652,49546.148,47706.117,48960.789,46942.219000000005,49058.668,48902.402,48829.832,47054.984,47166.688,48847.027,49327.723,50025.375,49944.625,51753.41,52633.535,46811.129,46091.391,46391.422,44883.91,45201.457,46063.27,44963.074,47092.492,48176.348,47783.359,47267.52,48278.363,47260.219000000005,42843.801,40693.676,43574.508,44895.098,42839.75,42716.594000000005,43208.539,42235.73,41034.543,41564.363,43790.895,48116.941,47711.488,48199.953,49112.902,51514.812000000005,55361.449,53805.984,53967.848,54968.223,54771.578,57484.789,56041.059,57401.098,57321.523,61593.949,60892.18,61553.617,62026.078,64261.992,65992.836,62210.172,60692.266,61393.617,60930.836,63039.824,60363.793,58482.387,60622.137,62227.965,61888.832,61318.957,61004.406,63226.402,62970.047,61452.23,61125.676,61527.48,63326.988,67566.828,66971.828,64995.23,64949.961,64155.941,64469.527,65466.84,63557.871,60161.246,60368.012,56942.137,58119.578,59697.195,58730.477,56289.289,57569.074,56280.426,57274.68,53569.766,54815.078,57248.457,57806.566,57005.426,57229.828,56477.816,53598.246,49200.703,49368.848,50582.625,50700.086,50504.797,47672.121,47243.305,49362.508,50098.336,46737.48,46612.633,48896.723,47665.426,46202.145,46848.777,46707.016,46880.277,48936.613,48628.512,50784.539,50822.195,50429.859,50809.516,50640.418,47588.855,46444.711,47178.125,46306.445,47686.812000000005,47345.219000000005,46458.117,45897.574,43569.004,43160.93,41557.902,41733.941,41911.602,41821.262,42735.855,43949.102,42591.57,43099.699,43177.398,43113.879,42250.551,42375.633,41744.328,40680.418,36457.316,35030.25,36276.805,36654.328,36954.004,36852.121,37138.234,37784.332,38138.18,37917.602,38483.125,38743.273,36952.984,37154.602,41500.875,41441.164,42412.434,43840.285,44118.445,44338.797,43565.113,42407.938,42244.469000000005,42197.516,42586.918,44575.203,43961.859,40538.012,40030.977,40122.156,38431.379,37075.281,38286.027,37296.57,38332.609,39214.219,39105.148,37709.785,43193.234,44354.637,43924.117,42451.789,39137.605,39400.586,38419.984,38062.039,38737.27,41982.926,39437.461,38794.973,38904.012,37849.664,39666.754,39338.785,41143.93,40951.379,41801.156,42190.652,41247.824,41077.996,42358.809,42892.957,43960.934,44395.965,44500.828,46820.492,47128.004,47465.73,47062.664,45538.676,46281.645,45868.949,46453.566,46622.676,45555.992,43206.738,43503.848,42287.664,42782.137,42207.672,39521.902,40127.184,41166.73,39935.516,40553.465,40424.484,39716.953,40826.215,41502.75,41374.379,40527.363,39740.32,39486.73,39469.293,40458.309,38117.461,39241.121,39773.828,38609.824,37714.875,38469.094,38529.328,37750.453,39698.371,36575.141,36040.922,35501.953,34059.266,30296.953,31022.906000000003,28936.355,29047.752,29283.104,30101.266,31305.113,29862.918,30425.857000000004,28720.271,30314.334,29200.74,29432.227000000003,30323.723,29098.91,29655.586,29562.361,29267.225,28627.574,28814.9,29445.957,31726.391,31792.311,29799.08,30467.488,29704.391,29832.914,29906.662,31370.672,31155.479,30214.355,30111.998,29083.805,28360.811,26762.648,22487.389,22206.793,22572.84,20381.65,20471.482,19017.643,20553.271,20599.537,20710.598,19987.029,21085.877,21231.656000000003,21502.338,21027.295,20735.479,20280.635,20104.023,19784.727,19269.367,19242.256,19297.076,20231.262,20190.115,20548.246,21637.588,21731.117,21592.207,20860.449,19970.557,19323.914,20212.074,20569.92,20836.328,21190.316000000003,20779.344,22485.689,23389.434,23231.732000000004,23164.629,22714.979,22465.479,22609.164,21361.701,21239.754,22930.549,23843.887,23804.633,23656.207,23336.896,23314.199,22978.117,22846.508,22630.957,23289.314,22961.279,23175.891,23809.486,23164.318,23947.643,23957.529,24402.818,24424.068,24319.334,24136.973,23883.291,23335.998,23212.738,20877.553,21166.061,21534.121,21398.908,21528.088,21395.02,21600.904,20260.02,20041.738,19616.814,20297.994,19796.809,20049.764,20127.141,19969.771,19832.088,19986.713,19812.371,18837.668,19290.324,19329.834,21381.152,21680.539,21769.256,22370.449,20296.707,20241.09,19701.211,19772.584,20127.576,19419.506,19544.129,18890.789,18547.4,19413.551,19297.639,18937.012,18802.098,19222.672,19110.547,19426.721,19573.051,19431.789,19312.096,19044.107,19623.58,20336.844,20160.717,19955.443,19546.85,19416.568,19446.426,19141.484,19051.418,19157.445,19382.904,19185.656000000003,19067.635,19268.094,19550.758,19334.416,19139.535,19053.74,19172.469,19208.189,19567.008,19345.572,20095.857,20770.441000000003,20285.836,20595.352,20818.477,20635.604,20495.773,20485.273,20159.504,20209.988,21147.23,21282.691000000003,20926.486,20602.816000000003,18541.271,15880.78,17586.771,17034.293,16799.186,16353.365,16618.199,16884.613,16669.439,16687.518,16697.777,16711.547,16291.832,15787.284,16189.77,16610.707,16604.465,16521.842,16464.281000000003,16444.627,16217.322,16444.982,17168.566000000003,16967.133,17088.66,16908.236,17130.486,16974.826,17089.504,16848.127,17233.475,17133.152,17128.725,17104.193,17206.438000000002,17781.318,17815.65,17364.865,16647.484,16795.092,16757.977,16439.68,16906.305,16817.535,16830.342,16796.953,16847.756,16841.986,16919.805,16717.174,16552.572,16642.342,16602.586,16547.496,16625.08,16688.471,16679.857,16863.238,16836.736,16951.968999999997,16955.078,17091.145,17196.555,17446.293,17934.896,18869.588,19909.574,20976.299,20880.799,21169.633,21161.52,20688.781000000003,21086.793,22676.553,22777.625,22720.416,22934.432,22636.469,23117.859,23032.777,23078.729,23031.09,23774.566000000003,22840.139,23139.283,23723.77,23471.871,23449.322,23331.848,22955.666,22760.109,23264.291,22939.398,21819.039,21651.184,21870.875,21788.203,21808.102000000003,22220.805,24307.842,23623.475,24565.602000000003,24641.277,24327.643,24829.148,24436.354,24188.844,23947.492,23198.127,23175.375,23561.213,23522.871,23147.354,23646.551,23475.467,22362.68,22353.35,22435.514,22429.758,22219.77,21718.08,20363.021,20187.244,20632.41,22163.949,24197.533,24746.074,24375.961,25052.789,27423.93,26965.879,28038.676,27767.236,28175.816000000003,27307.438,28333.973,27493.285,27494.707,27994.33,27139.889,27268.131],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"price"},"xaxis":{"title":{"text":"date"}},"yaxis":{"title":{"text":"value"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('3e390539-2faa-4709-8c01-cb88906cddc1');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
# Plotting Positivity Index
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x = result['date'], y = result['positivity_index'], name = 'positivity_index'))
fig2.update_layout(title = 'positivity_index', xaxis_title = 'date', yaxis_title = 'value')
fig2.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="4cd688b1-60a9-475b-8334-56811194644b" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4cd688b1-60a9-475b-8334-56811194644b")) {                    Plotly.newPlot(                        "4cd688b1-60a9-475b-8334-56811194644b",                        [{"name":"positivity_index","x":["2017-05-01","2017-05-02","2017-05-03","2017-05-04","2017-05-05","2017-05-06","2017-05-07","2017-05-08","2017-05-09","2017-05-10","2017-05-11","2017-05-12","2017-05-13","2017-05-14","2017-05-15","2017-05-16","2017-05-17","2017-05-18","2017-05-19","2017-05-20","2017-05-21","2017-05-22","2017-05-23","2017-05-24","2017-05-25","2017-05-26","2017-05-27","2017-05-28","2017-05-29","2017-05-30","2017-05-31","2017-06-01","2017-06-02","2017-06-03","2017-06-04","2017-06-05","2017-06-06","2017-06-07","2017-06-08","2017-06-09","2017-06-10","2017-06-11","2017-06-12","2017-06-13","2017-06-14","2017-06-15","2017-06-16","2017-06-17","2017-06-18","2017-06-19","2017-06-20","2017-06-21","2017-06-22","2017-06-23","2017-06-24","2017-06-25","2017-06-26","2017-06-27","2017-06-28","2017-06-29","2017-06-30","2017-07-01","2017-07-02","2017-07-03","2017-07-04","2017-07-05","2017-07-06","2017-07-07","2017-07-08","2017-07-09","2017-07-10","2017-07-11","2017-07-12","2017-07-13","2017-07-14","2017-07-15","2017-07-16","2017-07-17","2017-07-18","2017-07-19","2017-07-20","2017-07-21","2017-07-22","2017-07-23","2017-07-24","2017-07-25","2017-07-26","2017-07-27","2017-07-28","2017-07-29","2017-07-30","2017-07-31","2017-08-01","2017-08-02","2017-08-03","2017-08-04","2017-08-05","2017-08-06","2017-08-07","2017-08-08","2017-08-09","2017-08-10","2017-08-11","2017-08-12","2017-08-13","2017-08-14","2017-08-15","2017-08-16","2017-08-17","2017-08-18","2017-08-19","2017-08-20","2017-08-21","2017-08-22","2017-08-23","2017-08-24","2017-08-25","2017-08-26","2017-08-27","2017-08-28","2017-08-29","2017-08-30","2017-08-31","2017-09-01","2017-09-02","2017-09-03","2017-09-04","2017-09-05","2017-09-06","2017-09-07","2017-09-08","2017-09-09","2017-09-10","2017-09-11","2017-09-12","2017-09-13","2017-09-14","2017-09-15","2017-09-16","2017-09-17","2017-09-18","2017-09-19","2017-09-20","2017-09-21","2017-09-22","2017-09-23","2017-09-24","2017-09-25","2017-09-26","2017-09-27","2017-09-28","2017-09-29","2017-09-30","2017-10-01","2017-10-02","2017-10-03","2017-10-04","2017-10-05","2017-10-06","2017-10-07","2017-10-08","2017-10-09","2017-10-10","2017-10-11","2017-10-12","2017-10-13","2017-10-14","2017-10-15","2017-10-16","2017-10-17","2017-10-18","2017-10-19","2017-10-20","2017-10-21","2017-10-22","2017-10-23","2017-10-24","2017-10-25","2017-10-26","2017-10-27","2017-10-28","2017-10-29","2017-10-30","2017-10-31","2017-11-01","2017-11-02","2017-11-03","2017-11-04","2017-11-05","2017-11-06","2017-11-07","2017-11-08","2017-11-09","2017-11-10","2017-11-11","2017-11-12","2017-11-13","2017-11-14","2017-11-15","2017-11-16","2017-11-17","2017-11-18","2017-11-19","2017-11-20","2017-11-21","2017-11-22","2017-11-23","2017-11-24","2017-11-25","2017-11-26","2017-11-27","2017-11-28","2017-11-29","2017-11-30","2017-12-01","2017-12-02","2017-12-03","2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08","2017-12-09","2017-12-10","2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15","2017-12-16","2017-12-17","2017-12-18","2017-12-19","2017-12-20","2017-12-21","2017-12-22","2017-12-23","2017-12-24","2017-12-25","2017-12-26","2017-12-27","2017-12-28","2017-12-29","2017-12-30","2017-12-31","2018-01-01","2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-06","2018-01-07","2018-01-08","2018-01-09","2018-01-10","2018-01-11","2018-01-12","2018-01-13","2018-01-14","2018-01-15","2018-01-16","2018-01-17","2018-01-18","2018-01-19","2018-01-20","2018-01-21","2018-01-22","2018-01-23","2018-01-24","2018-01-25","2018-01-26","2018-01-27","2018-01-28","2018-01-29","2018-01-30","2018-01-31","2018-02-01","2018-02-02","2018-02-03","2018-02-04","2018-02-05","2018-02-06","2018-02-07","2018-02-08","2018-02-09","2018-02-10","2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23","2018-02-24","2018-02-25","2018-02-26","2018-02-27","2018-02-28","2018-03-01","2018-03-02","2018-03-03","2018-03-04","2018-03-05","2018-03-06","2018-03-07","2018-03-08","2018-03-09","2018-03-10","2018-03-11","2018-03-12","2018-03-13","2018-03-14","2018-03-15","2018-03-16","2018-03-17","2018-03-18","2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26","2018-03-27","2018-03-28","2018-03-29","2018-03-30","2018-03-31","2018-04-01","2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15","2018-04-16","2018-04-17","2018-04-18","2018-04-19","2018-04-20","2018-04-21","2018-04-22","2018-04-23","2018-04-24","2018-04-25","2018-04-26","2018-04-27","2018-04-28","2018-04-29","2018-04-30","2018-05-01","2018-05-02","2018-05-03","2018-05-04","2018-05-05","2018-05-06","2018-05-07","2018-05-08","2018-05-09","2018-05-10","2018-05-11","2018-05-12","2018-05-13","2018-05-14","2018-05-15","2018-05-16","2018-05-17","2018-05-18","2018-05-19","2018-05-20","2018-05-21","2018-05-22","2018-05-23","2018-05-24","2018-05-25","2018-05-26","2018-05-27","2018-05-28","2018-05-29","2018-05-30","2018-05-31","2018-06-01","2018-06-02","2018-06-03","2018-06-04","2018-06-05","2018-06-06","2018-06-07","2018-06-08","2018-06-09","2018-06-10","2018-06-11","2018-06-12","2018-06-13","2018-06-14","2018-06-15","2018-06-16","2018-06-17","2018-06-18","2018-06-19","2018-06-20","2018-06-21","2018-06-22","2018-06-23","2018-06-24","2018-06-25","2018-06-26","2018-06-27","2018-06-28","2018-06-29","2018-06-30","2018-07-01","2018-07-02","2018-07-03","2018-07-04","2018-07-05","2018-07-06","2018-07-07","2018-07-08","2018-07-09","2018-07-10","2018-07-11","2018-07-12","2018-07-13","2018-07-14","2018-07-15","2018-07-16","2018-07-17","2018-07-18","2018-07-19","2018-07-20","2018-07-21","2018-07-22","2018-07-23","2018-07-24","2018-07-25","2018-07-26","2018-07-27","2018-07-28","2018-07-29","2018-07-30","2018-07-31","2018-08-01","2018-08-02","2018-08-03","2018-08-04","2018-08-05","2018-08-06","2018-08-07","2018-08-08","2018-08-09","2018-08-10","2018-08-11","2018-08-12","2018-08-13","2018-08-14","2018-08-15","2018-08-16","2018-08-17","2018-08-18","2018-08-19","2018-08-20","2018-08-21","2018-08-22","2018-08-23","2018-08-24","2018-08-25","2018-08-26","2018-08-27","2018-08-28","2018-08-29","2018-08-30","2018-08-31","2018-09-01","2018-09-02","2018-09-03","2018-09-04","2018-09-05","2018-09-06","2018-09-07","2018-09-08","2018-09-09","2018-09-10","2018-09-11","2018-09-12","2018-09-13","2018-09-14","2018-09-15","2018-09-16","2018-09-17","2018-09-18","2018-09-19","2018-09-20","2018-09-21","2018-09-22","2018-09-23","2018-09-24","2018-09-25","2018-09-26","2018-09-27","2018-09-28","2018-09-29","2018-09-30","2018-10-01","2018-10-02","2018-10-03","2018-10-04","2018-10-05","2018-10-06","2018-10-07","2018-10-08","2018-10-09","2018-10-10","2018-10-11","2018-10-12","2018-10-13","2018-10-14","2018-10-15","2018-10-16","2018-10-17","2018-10-18","2018-10-19","2018-10-20","2018-10-21","2018-10-22","2018-10-23","2018-10-24","2018-10-25","2018-10-26","2018-10-27","2018-10-28","2018-10-29","2018-10-30","2018-10-31","2018-11-01","2018-11-02","2018-11-03","2018-11-04","2018-11-05","2018-11-06","2018-11-07","2018-11-08","2018-11-09","2018-11-10","2018-11-11","2018-11-12","2018-11-13","2018-11-14","2018-11-15","2018-11-16","2018-11-17","2018-11-18","2018-11-19","2018-11-20","2018-11-21","2018-11-22","2018-11-23","2018-11-24","2018-11-25","2018-11-26","2018-11-27","2018-11-28","2018-11-29","2018-11-30","2018-12-01","2018-12-02","2018-12-03","2018-12-04","2018-12-05","2018-12-06","2018-12-07","2018-12-08","2018-12-09","2018-12-10","2018-12-11","2018-12-12","2018-12-13","2018-12-14","2018-12-15","2018-12-16","2018-12-17","2018-12-18","2018-12-19","2018-12-20","2018-12-21","2018-12-22","2018-12-23","2018-12-24","2018-12-25","2018-12-26","2018-12-27","2018-12-28","2018-12-29","2018-12-30","2018-12-31","2019-01-01","2019-01-02","2019-01-03","2019-01-04","2019-01-05","2019-01-06","2019-01-07","2019-01-08","2019-01-09","2019-01-10","2019-01-11","2019-01-12","2019-01-13","2019-01-14","2019-01-15","2019-01-16","2019-01-17","2019-01-18","2019-01-19","2019-01-20","2019-01-21","2019-01-22","2019-01-23","2019-01-24","2019-01-25","2019-01-26","2019-01-27","2019-01-28","2019-01-29","2019-01-30","2019-01-31","2019-02-01","2019-02-02","2019-02-03","2019-02-04","2019-02-05","2019-02-06","2019-02-07","2019-02-08","2019-02-09","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2019-02-16","2019-02-17","2019-02-18","2019-02-19","2019-02-20","2019-02-21","2019-02-22","2019-02-23","2019-02-24","2019-02-25","2019-02-26","2019-02-27","2019-02-28","2019-03-01","2019-03-02","2019-03-03","2019-03-04","2019-03-05","2019-03-06","2019-03-07","2019-03-08","2019-03-09","2019-03-10","2019-03-11","2019-03-12","2019-03-13","2019-03-14","2019-03-15","2019-03-16","2019-03-17","2019-03-18","2019-03-19","2019-03-20","2019-03-21","2019-03-22","2019-03-23","2019-03-24","2019-03-25","2019-03-26","2019-03-27","2019-03-28","2019-03-29","2019-03-30","2019-03-31","2019-04-01","2019-04-02","2019-04-03","2019-04-04","2019-04-05","2019-04-06","2019-04-07","2019-04-08","2019-04-09","2019-04-10","2019-04-11","2019-04-12","2019-04-13","2019-04-14","2019-04-15","2019-04-16","2019-04-17","2019-04-18","2019-04-19","2019-04-20","2019-04-21","2019-04-22","2019-04-23","2019-04-24","2019-04-25","2019-04-26","2019-04-27","2019-04-28","2019-04-29","2019-04-30","2019-05-01","2019-05-02","2019-05-03","2019-05-04","2019-05-05","2019-05-06","2019-05-07","2019-05-08","2019-05-09","2019-05-10","2019-05-11","2019-05-12","2019-05-13","2019-05-14","2019-05-15","2019-05-16","2019-05-17","2019-05-18","2019-05-19","2019-05-20","2019-05-21","2019-05-22","2019-05-23","2019-05-24","2019-05-25","2019-05-26","2019-05-27","2019-05-28","2019-05-29","2019-05-30","2019-05-31","2019-06-01","2019-06-02","2019-06-03","2019-06-04","2019-06-05","2019-06-06","2019-06-07","2019-06-08","2019-06-09","2019-06-10","2019-06-11","2019-06-12","2019-06-13","2019-06-14","2019-06-15","2019-06-16","2019-06-17","2019-06-18","2019-06-19","2019-06-20","2019-06-21","2019-06-22","2019-06-23","2019-06-24","2019-06-25","2019-06-26","2019-06-27","2019-06-28","2019-06-29","2019-06-30","2019-07-01","2019-07-02","2019-07-03","2019-07-04","2019-07-05","2019-07-06","2019-07-07","2019-07-08","2019-07-09","2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15","2019-07-16","2019-07-17","2019-07-18","2019-07-19","2019-07-20","2019-07-21","2019-07-22","2019-07-23","2019-07-24","2019-07-25","2019-07-26","2019-07-27","2019-07-28","2019-07-29","2019-07-30","2019-07-31","2019-08-01","2019-08-02","2019-08-03","2019-08-04","2019-08-05","2019-08-06","2019-08-07","2019-08-08","2019-08-09","2019-08-10","2019-08-11","2019-08-12","2019-08-13","2019-08-14","2019-08-15","2019-08-16","2019-08-17","2019-08-18","2019-08-19","2019-08-20","2019-08-21","2019-08-22","2019-08-23","2019-08-24","2019-08-25","2019-08-26","2019-08-27","2019-08-28","2019-08-29","2019-08-30","2019-08-31","2019-09-01","2019-09-02","2019-09-03","2019-09-04","2019-09-05","2019-09-06","2019-09-07","2019-09-08","2019-09-09","2019-09-10","2019-09-11","2019-09-12","2019-09-13","2019-09-14","2019-09-15","2019-09-16","2019-09-17","2019-09-18","2019-09-19","2019-09-20","2019-09-21","2019-09-22","2019-09-23","2019-09-24","2019-09-25","2019-09-26","2019-09-27","2019-09-28","2019-09-29","2019-09-30","2019-10-01","2019-10-02","2019-10-03","2019-10-04","2019-10-05","2019-10-06","2019-10-07","2019-10-08","2019-10-09","2019-10-10","2019-10-11","2019-10-12","2019-10-13","2019-10-14","2019-10-15","2019-10-16","2019-10-17","2019-10-18","2019-10-19","2019-10-20","2019-10-21","2019-10-22","2019-10-23","2019-10-24","2019-10-25","2019-10-26","2019-10-27","2019-10-28","2019-10-29","2019-10-30","2019-10-31","2019-11-01","2019-11-02","2019-11-03","2019-11-04","2019-11-05","2019-11-06","2019-11-07","2019-11-08","2019-11-09","2019-11-10","2019-11-11","2019-11-12","2019-11-13","2019-11-14","2019-11-15","2019-11-16","2019-11-17","2019-11-18","2019-11-19","2019-11-20","2019-11-21","2019-11-22","2019-11-23","2019-11-24","2019-11-25","2019-11-26","2019-11-27","2019-11-28","2019-11-29","2019-11-30","2019-12-01","2019-12-02","2019-12-03","2019-12-04","2019-12-05","2019-12-06","2019-12-07","2019-12-08","2019-12-09","2019-12-10","2019-12-11","2019-12-12","2019-12-13","2019-12-14","2019-12-15","2019-12-16","2019-12-17","2019-12-18","2019-12-19","2019-12-20","2019-12-21","2019-12-22","2019-12-23","2019-12-24","2019-12-25","2019-12-26","2019-12-27","2019-12-28","2019-12-29","2019-12-30","2019-12-31","2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05","2020-01-06","2020-01-07","2020-01-08","2020-01-09","2020-01-10","2020-01-11","2020-01-12","2020-01-13","2020-01-14","2020-01-15","2020-01-16","2020-01-17","2020-01-18","2020-01-19","2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-25","2020-01-26","2020-01-27","2020-01-28","2020-01-29","2020-01-30","2020-01-31","2020-02-01","2020-02-02","2020-02-03","2020-02-04","2020-02-05","2020-02-06","2020-02-07","2020-02-08","2020-02-09","2020-02-10","2020-02-11","2020-02-12","2020-02-13","2020-02-14","2020-02-15","2020-02-16","2020-02-17","2020-02-18","2020-02-19","2020-02-20","2020-02-21","2020-02-22","2020-02-23","2020-02-24","2020-02-25","2020-02-26","2020-02-27","2020-02-28","2020-02-29","2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05","2020-03-06","2020-03-07","2020-03-08","2020-03-09","2020-03-10","2020-03-11","2020-03-12","2020-03-13","2020-03-14","2020-03-15","2020-03-16","2020-03-17","2020-03-18","2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01","2020-04-02","2020-04-03","2020-04-04","2020-04-05","2020-04-06","2020-04-07","2020-04-08","2020-04-09","2020-04-10","2020-04-11","2020-04-12","2020-04-13","2020-04-14","2020-04-15","2020-04-16","2020-04-17","2020-04-18","2020-04-19","2020-04-20","2020-04-21","2020-04-22","2020-04-23","2020-04-24","2020-04-25","2020-04-26","2020-04-27","2020-04-28","2020-04-29","2020-04-30","2020-05-01","2020-05-02","2020-05-03","2020-05-04","2020-05-05","2020-05-06","2020-05-07","2020-05-08","2020-05-09","2020-05-10","2020-05-11","2020-05-12","2020-05-13","2020-05-14","2020-05-15","2020-05-16","2020-05-17","2020-05-18","2020-05-19","2020-05-20","2020-05-21","2020-05-22","2020-05-23","2020-05-24","2020-05-25","2020-05-26","2020-05-27","2020-05-28","2020-05-29","2020-05-30","2020-05-31","2020-06-01","2020-06-02","2020-06-03","2020-06-04","2020-06-05","2020-06-06","2020-06-07","2020-06-08","2020-06-09","2020-06-10","2020-06-11","2020-06-12","2020-06-13","2020-06-14","2020-06-15","2020-06-16","2020-06-17","2020-06-18","2020-06-19","2020-06-20","2020-06-21","2020-06-22","2020-06-23","2020-06-24","2020-06-25","2020-06-26","2020-06-27","2020-06-28","2020-06-29","2020-06-30","2020-07-01","2020-07-02","2020-07-03","2020-07-04","2020-07-05","2020-07-06","2020-07-07","2020-07-08","2020-07-09","2020-07-10","2020-07-11","2020-07-12","2020-07-13","2020-07-14","2020-07-15","2020-07-16","2020-07-17","2020-07-18","2020-07-19","2020-07-20","2020-07-21","2020-07-22","2020-07-23","2020-07-24","2020-07-25","2020-07-26","2020-07-27","2020-07-28","2020-07-29","2020-07-30","2020-07-31","2020-08-01","2020-08-02","2020-08-03","2020-08-04","2020-08-05","2020-08-06","2020-08-07","2020-08-08","2020-08-09","2020-08-10","2020-08-11","2020-08-12","2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17","2020-08-18","2020-08-19","2020-08-20","2020-08-21","2020-08-22","2020-08-23","2020-08-24","2020-08-25","2020-08-26","2020-08-27","2020-08-28","2020-08-29","2020-08-30","2020-08-31","2020-09-01","2020-09-02","2020-09-03","2020-09-04","2020-09-05","2020-09-06","2020-09-07","2020-09-08","2020-09-09","2020-09-10","2020-09-11","2020-09-12","2020-09-13","2020-09-14","2020-09-15","2020-09-16","2020-09-17","2020-09-18","2020-09-19","2020-09-20","2020-09-21","2020-09-22","2020-09-23","2020-09-24","2020-09-25","2020-09-26","2020-09-27","2020-09-28","2020-09-29","2020-09-30","2020-10-01","2020-10-02","2020-10-03","2020-10-04","2020-10-05","2020-10-06","2020-10-07","2020-10-08","2020-10-09","2020-10-10","2020-10-11","2020-10-12","2020-10-13","2020-10-14","2020-10-15","2020-10-16","2020-10-17","2020-10-18","2020-10-19","2020-10-20","2020-10-21","2020-10-22","2020-10-23","2020-10-24","2020-10-25","2020-10-26","2020-10-27","2020-10-28","2020-10-29","2020-10-30","2020-10-31","2020-11-01","2020-11-02","2020-11-03","2020-11-04","2020-11-05","2020-11-06","2020-11-07","2020-11-08","2020-11-09","2020-11-10","2020-11-11","2020-11-12","2020-11-13","2020-11-14","2020-11-15","2020-11-16","2020-11-17","2020-11-18","2020-11-19","2020-11-20","2020-11-21","2020-11-22","2020-11-23","2020-11-24","2020-11-25","2020-11-26","2020-11-27","2020-11-28","2020-11-29","2020-11-30","2020-12-01","2020-12-02","2020-12-03","2020-12-04","2020-12-05","2020-12-06","2020-12-07","2020-12-08","2020-12-09","2020-12-10","2020-12-11","2020-12-12","2020-12-13","2020-12-14","2020-12-15","2020-12-16","2020-12-17","2020-12-18","2020-12-19","2020-12-20","2020-12-21","2020-12-22","2020-12-23","2020-12-24","2020-12-25","2020-12-26","2020-12-27","2020-12-28","2020-12-29","2020-12-30","2020-12-31","2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05","2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10","2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15","2021-01-16","2021-01-17","2021-01-18","2021-01-19","2021-01-20","2021-01-21","2021-01-22","2021-01-23","2021-01-24","2021-01-25","2021-01-26","2021-01-27","2021-01-28","2021-01-29","2021-01-30","2021-01-31","2021-02-01","2021-02-02","2021-02-03","2021-02-04","2021-02-05","2021-02-06","2021-02-07","2021-02-08","2021-02-09","2021-02-10","2021-02-11","2021-02-12","2021-02-13","2021-02-14","2021-02-15","2021-02-16","2021-02-17","2021-02-18","2021-02-19","2021-02-20","2021-02-21","2021-02-22","2021-02-23","2021-02-24","2021-02-25","2021-02-26","2021-02-27","2021-02-28","2021-03-01","2021-03-02","2021-03-03","2021-03-04","2021-03-05","2021-03-06","2021-03-07","2021-03-08","2021-03-09","2021-03-10","2021-03-11","2021-03-12","2021-03-13","2021-03-14","2021-03-15","2021-03-16","2021-03-17","2021-03-18","2021-03-19","2021-03-20","2021-03-21","2021-03-22","2021-03-23","2021-03-24","2021-03-25","2021-03-26","2021-03-27","2021-03-28","2021-03-29","2021-03-30","2021-03-31","2021-04-01","2021-04-02","2021-04-03","2021-04-04","2021-04-05","2021-04-06","2021-04-07","2021-04-08","2021-04-09","2021-04-10","2021-04-11","2021-04-12","2021-04-13","2021-04-14","2021-04-15","2021-04-16","2021-04-17","2021-04-18","2021-04-19","2021-04-20","2021-04-21","2021-04-22","2021-04-23","2021-04-24","2021-04-25","2021-04-26","2021-04-27","2021-04-28","2021-04-29","2021-04-30","2021-05-01","2021-05-02","2021-05-03","2021-05-04","2021-05-05","2021-05-06","2021-05-07","2021-05-08","2021-05-09","2021-05-10","2021-05-11","2021-05-12","2021-05-13","2021-05-14","2021-05-15","2021-05-16","2021-05-17","2021-05-18","2021-05-19","2021-05-20","2021-05-21","2021-05-22","2021-05-23","2021-05-24","2021-05-25","2021-05-26","2021-05-27","2021-05-28","2021-05-29","2021-05-30","2021-05-31","2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07","2021-06-08","2021-06-09","2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29","2021-06-30","2021-07-01","2021-07-02","2021-07-03","2021-07-04","2021-07-05","2021-07-06","2021-07-07","2021-07-08","2021-07-09","2021-07-10","2021-07-11","2021-07-12","2021-07-13","2021-07-14","2021-07-15","2021-07-16","2021-07-17","2021-07-18","2021-07-19","2021-07-20","2021-07-21","2021-07-22","2021-07-23","2021-07-24","2021-07-25","2021-07-26","2021-07-27","2021-07-28","2021-07-29","2021-07-30","2021-07-31","2021-08-01","2021-08-02","2021-08-03","2021-08-04","2021-08-05","2021-08-06","2021-08-07","2021-08-08","2021-08-09","2021-08-10","2021-08-11","2021-08-12","2021-08-13","2021-08-14","2021-08-15","2021-08-16","2021-08-17","2021-08-18","2021-08-19","2021-08-20","2021-08-21","2021-08-22","2021-08-23","2021-08-24","2021-08-25","2021-08-26","2021-08-27","2021-08-28","2021-08-29","2021-08-30","2021-08-31","2021-09-01","2021-09-02","2021-09-03","2021-09-04","2021-09-05","2021-09-06","2021-09-07","2021-09-08","2021-09-09","2021-09-10","2021-09-11","2021-09-12","2021-09-13","2021-09-14","2021-09-15","2021-09-16","2021-09-17","2021-09-18","2021-09-19","2021-09-20","2021-09-21","2021-09-22","2021-09-23","2021-09-24","2021-09-25","2021-09-26","2021-09-27","2021-09-28","2021-09-29","2021-09-30","2021-10-01","2021-10-02","2021-10-03","2021-10-04","2021-10-05","2021-10-06","2021-10-07","2021-10-08","2021-10-09","2021-10-10","2021-10-11","2021-10-12","2021-10-13","2021-10-14","2021-10-15","2021-10-16","2021-10-17","2021-10-18","2021-10-19","2021-10-20","2021-10-21","2021-10-22","2021-10-23","2021-10-24","2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29","2021-10-30","2021-10-31","2021-11-01","2021-11-02","2021-11-03","2021-11-04","2021-11-05","2021-11-06","2021-11-07","2021-11-08","2021-11-09","2021-11-10","2021-11-11","2021-11-12","2021-11-13","2021-11-14","2021-11-15","2021-11-16","2021-11-17","2021-11-18","2021-11-19","2021-11-20","2021-11-21","2021-11-22","2021-11-23","2021-11-24","2021-11-25","2021-11-26","2021-11-27","2021-11-28","2021-11-29","2021-11-30","2021-12-01","2021-12-02","2021-12-03","2021-12-04","2021-12-05","2021-12-06","2021-12-07","2021-12-08","2021-12-09","2021-12-10","2021-12-11","2021-12-12","2021-12-13","2021-12-14","2021-12-15","2021-12-16","2021-12-17","2021-12-18","2021-12-19","2021-12-20","2021-12-21","2021-12-22","2021-12-23","2021-12-24","2021-12-25","2021-12-26","2021-12-27","2021-12-28","2021-12-29","2021-12-30","2021-12-31","2022-01-01","2022-01-02","2022-01-03","2022-01-04","2022-01-05","2022-01-06","2022-01-07","2022-01-08","2022-01-09","2022-01-10","2022-01-11","2022-01-12","2022-01-13","2022-01-14","2022-01-15","2022-01-16","2022-01-17","2022-01-18","2022-01-19","2022-01-20","2022-01-21","2022-01-22","2022-01-23","2022-01-24","2022-01-25","2022-01-26","2022-01-27","2022-01-28","2022-01-29","2022-01-30","2022-01-31","2022-02-01","2022-02-02","2022-02-03","2022-02-04","2022-02-05","2022-02-06","2022-02-07","2022-02-08","2022-02-09","2022-02-10","2022-02-11","2022-02-12","2022-02-13","2022-02-14","2022-02-15","2022-02-16","2022-02-17","2022-02-18","2022-02-19","2022-02-20","2022-02-21","2022-02-22","2022-02-23","2022-02-24","2022-02-25","2022-02-26","2022-02-27","2022-02-28","2022-03-01","2022-03-02","2022-03-03","2022-03-04","2022-03-05","2022-03-06","2022-03-07","2022-03-08","2022-03-09","2022-03-10","2022-03-11","2022-03-12","2022-03-13","2022-03-14","2022-03-15","2022-03-16","2022-03-17","2022-03-18","2022-03-19","2022-03-20","2022-03-21","2022-03-22","2022-03-23","2022-03-24","2022-03-25","2022-03-26","2022-03-27","2022-03-28","2022-03-29","2022-03-30","2022-03-31","2022-04-01","2022-04-02","2022-04-03","2022-04-04","2022-04-05","2022-04-06","2022-04-07","2022-04-08","2022-04-09","2022-04-10","2022-04-11","2022-04-12","2022-04-13","2022-04-14","2022-04-15","2022-04-16","2022-04-17","2022-04-18","2022-04-19","2022-04-20","2022-04-21","2022-04-22","2022-04-23","2022-04-24","2022-04-25","2022-04-26","2022-04-27","2022-04-28","2022-04-29","2022-04-30","2022-05-01","2022-05-02","2022-05-03","2022-05-04","2022-05-05","2022-05-06","2022-05-07","2022-05-08","2022-05-09","2022-05-10","2022-05-11","2022-05-12","2022-05-13","2022-05-14","2022-05-15","2022-05-16","2022-05-17","2022-05-18","2022-05-19","2022-05-20","2022-05-21","2022-05-22","2022-05-23","2022-05-24","2022-05-25","2022-05-26","2022-05-27","2022-05-28","2022-05-29","2022-05-30","2022-05-31","2022-06-01","2022-06-02","2022-06-03","2022-06-04","2022-06-05","2022-06-06","2022-06-07","2022-06-08","2022-06-09","2022-06-10","2022-06-11","2022-06-12","2022-06-13","2022-06-14","2022-06-15","2022-06-16","2022-06-17","2022-06-18","2022-06-19","2022-06-20","2022-06-21","2022-06-22","2022-06-23","2022-06-24","2022-06-25","2022-06-26","2022-06-27","2022-06-28","2022-06-29","2022-06-30","2022-07-01","2022-07-02","2022-07-03","2022-07-04","2022-07-05","2022-07-06","2022-07-07","2022-07-08","2022-07-09","2022-07-10","2022-07-11","2022-07-12","2022-07-13","2022-07-14","2022-07-15","2022-07-16","2022-07-17","2022-07-18","2022-07-19","2022-07-20","2022-07-21","2022-07-22","2022-07-23","2022-07-24","2022-07-25","2022-07-26","2022-07-27","2022-07-28","2022-07-29","2022-07-30","2022-07-31","2022-08-01","2022-08-02","2022-08-03","2022-08-04","2022-08-05","2022-08-06","2022-08-07","2022-08-08","2022-08-09","2022-08-10","2022-08-11","2022-08-12","2022-08-13","2022-08-14","2022-08-15","2022-08-16","2022-08-17","2022-08-18","2022-08-19","2022-08-20","2022-08-21","2022-08-22","2022-08-23","2022-08-24","2022-08-25","2022-08-26","2022-08-27","2022-08-28","2022-08-29","2022-08-30","2022-08-31","2022-09-01","2022-09-02","2022-09-03","2022-09-04","2022-09-05","2022-09-06","2022-09-07","2022-09-08","2022-09-09","2022-09-10","2022-09-11","2022-09-12","2022-09-13","2022-09-14","2022-09-15","2022-09-16","2022-09-17","2022-09-18","2022-09-19","2022-09-20","2022-09-21","2022-09-22","2022-09-23","2022-09-24","2022-09-25","2022-09-26","2022-09-27","2022-09-28","2022-09-29","2022-09-30","2022-10-01","2022-10-02","2022-10-03","2022-10-04","2022-10-05","2022-10-06","2022-10-07","2022-10-08","2022-10-09","2022-10-10","2022-10-11","2022-10-12","2022-10-13","2022-10-14","2022-10-15","2022-10-16","2022-10-17","2022-10-18","2022-10-19","2022-10-20","2022-10-21","2022-10-22","2022-10-23","2022-10-24","2022-10-25","2022-10-26","2022-10-27","2022-10-28","2022-10-29","2022-10-30","2022-10-31","2022-11-01","2022-11-02","2022-11-03","2022-11-04","2022-11-05","2022-11-06","2022-11-07","2022-11-08","2022-11-09","2022-11-10","2022-11-11","2022-11-12","2022-11-13","2022-11-14","2022-11-15","2022-11-16","2022-11-17","2022-11-18","2022-11-19","2022-11-20","2022-11-21","2022-11-22","2022-11-23","2022-11-24","2022-11-25","2022-11-26","2022-11-27","2022-11-28","2022-11-29","2022-11-30","2022-12-01","2022-12-02","2022-12-03","2022-12-04","2022-12-05","2022-12-06","2022-12-07","2022-12-08","2022-12-09","2022-12-10","2022-12-11","2022-12-12","2022-12-13","2022-12-14","2022-12-15","2022-12-16","2022-12-17","2022-12-18","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-24","2022-12-25","2022-12-26","2022-12-27","2022-12-28","2022-12-29","2022-12-30","2022-12-31","2023-01-01","2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-07","2023-01-08","2023-01-09","2023-01-10","2023-01-11","2023-01-12","2023-01-13","2023-01-14","2023-01-15","2023-01-16","2023-01-17","2023-01-18","2023-01-19","2023-01-20","2023-01-21","2023-01-22","2023-01-23","2023-01-24","2023-01-25","2023-01-26","2023-01-27","2023-01-28","2023-01-29","2023-01-30","2023-01-31","2023-02-01","2023-02-02","2023-02-03","2023-02-04","2023-02-05","2023-02-06","2023-02-07","2023-02-08","2023-02-09","2023-02-10","2023-02-11","2023-02-12","2023-02-13","2023-02-14","2023-02-15","2023-02-16","2023-02-17","2023-02-18","2023-02-19","2023-02-20","2023-02-21","2023-02-22","2023-02-23","2023-02-24","2023-02-25","2023-02-26","2023-02-27","2023-02-28","2023-03-01","2023-03-02","2023-03-03","2023-03-04","2023-03-05","2023-03-06","2023-03-07","2023-03-08","2023-03-09","2023-03-10","2023-03-11","2023-03-12","2023-03-13","2023-03-14","2023-03-15","2023-03-16","2023-03-17","2023-03-18","2023-03-19","2023-03-20","2023-03-21","2023-03-22","2023-03-23","2023-03-24","2023-03-25","2023-03-26","2023-03-27","2023-03-28"],"y":[0.6253245842211843,0.5228018148127064,0.6584616225720553,0.8783714970118612,0.6012662287303592,0.3566749439387324,0.9244208579574053,0.4688869345456606,0.2413739885756553,0.7594482603811854,0.4900224961362274,0.4415611238650373,0.0461893824693747,0.4426878185987727,0.4054651081081644,0.5108256237659907,0.40140830741255,0.4488987015398678,0.4173243332503078,0.2943710606025776,0.4264082819534076,0.246064521220886,0.2833018639592199,0.4937183356144239,0.207341386064963,-0.0887719030256473,-0.1870751050649905,0.0662090369291873,-0.1206279877886147,0.1593865855114587,0.072467839444341,0.1885355137618977,0.0508837939723277,-0.0339015516756813,0.1501745314476188,0.0094118341823465,0.3189150921417124,0.0437228110138315,-0.067365531827576,0.3098559299461029,0.1294320928735819,0.1345901539727647,0.2914513946680728,0.0821856749715148,0.1605268969665343,-0.2512015041093672,0.2617813440196238,0.6517945668911846,0.5679386851036071,0.7796672777558571,0.452107966320449,0.1503119035200925,0.1625991830991983,0.3723410606831318,-0.1233130036658446,0.0979545180574152,-0.0736118173432459,-0.2223400157145264,0.1320717514370374,0.0748550771567207,0.3880300462860629,-0.1482299994175306,-0.0867509264086548,0.4954133447711038,0.5215495871289663,0.3043689912367957,0.6110728020351923,0.019531870917246,0.1209984725103578,0.329780892760588,0.162016290633209,-0.1012166031541513,0.1668815965212982,0.3024802935507308,0.0686768657962193,-0.180381691976128,-0.2449926637567304,-0.1118414197419094,0.2221982227421562,0.1611250670722034,0.2540088573451929,0.2662515385558928,-0.0870113769896298,-0.0813701681955064,0.3656265838037292,-0.0155041865359652,-0.1793738759566122,0.0507723253734231,0.0289875368732521,-0.2157995770584512,0.252565601418789,0.1785141004673732,0.5067356385144657,0.2771065462046945,0.2026612410310773,0.1369223758670809,0.3660401436083025,0.3301212557269376,0.2142044186296748,0.3087758555115805,0.1955555947953807,0.4078432305131318,0.3788342421074421,0.3392157225669637,0.1568424714929696,0.4035585298375828,0.1085663805514595,0.180546933435586,0.4598263173270518,0.2434348973624805,0.2702903297399117,-0.0292703823001132,0.108400955823093,0.0466502494046933,0.2716919652104953,0.1310791866820171,-0.0470470035060994,0.3750595933400517,0.3818335035748872,0.4141780345150883,0.386818438940897,0.2381814286787503,0.5489331937654514,0.469707376836518,0.0245444554627324,-0.2506969537321114,-0.2304339845768891,0.010216198874721,0.2399354139840946,0.1138875924936465,0.0934657280056771,-0.351166667617917,-0.2867659019800013,0.0254725947658078,0.1385497393847785,-0.2544248506952985,-0.0672350928659598,-0.0754397076015754,-0.0301997398636324,-0.1416155110244816,0.3273214083423899,-0.0918934045869381,0.0846369541276422,0.0442248937378571,-0.1138016103571985,0.2630762616495806,0.2047944126460132,0.2585739882937104,0.3703737882968943,0.2526824175503309,0.1819279334863729,-0.012248622076199,0.2562188029959958,0.1934948573920799,0.2185992365007321,0.0,0.1157502195069621,-0.0535971860089779,0.3156381497183709,0.2433917121441572,0.4294945328349997,0.1193794368371572,-0.0104712998672953,0.0789630438068013,0.2120587188897168,0.32335276655504,0.3905764956144137,0.1725935978652853,0.2295744416445002,0.1069340801052568,-0.259906254217222,0.0612019652280756,0.4453916932632253,0.083900113066294,-0.0394973765939725,-0.1407276594951176,0.1979265548628446,-0.0881927120354608,0.2111673602674941,0.2366381857223382,0.2344953654586989,0.175632568643158,0.1587742683233206,0.4398853319946437,0.3323337807522972,0.0776417461359983,0.1605079803227246,0.1772535892547115,0.1541506798272583,0.4452143624999656,0.3912902813168446,0.4794581905558775,0.2910043346437587,0.0903428625806332,0.0510201574076578,-0.0365988458954085,-0.0184247636306981,0.1228027028163772,0.2336706656542124,0.1389630051049739,0.2020439094272448,0.2607444167199724,0.3350598070078705,0.4454059248317746,0.2953189360421259,0.4061311087987825,0.3474828255698172,0.2291901640815806,0.3995562120477897,0.519300250756963,0.4806646642584982,0.5724592620341219,0.341398981641679,0.0393870136909226,0.4709462575343942,0.5917480201532834,0.8406772426736782,0.6925108458264145,0.8391151782852128,0.5337355046637648,0.5247975130498492,0.2978291382849328,0.3494680200629384,0.0980014570774605,0.4537172882987058,0.8262595846167261,0.6056235084771442,0.6342867703320356,0.513690955239277,0.6701754895538443,0.3798624538939608,0.5409452180331027,0.4426380688895494,0.3843381087144753,0.4998108476796897,-0.265540266218548,0.1288704366284172,-0.0348459323895634,0.2669183177625151,0.3634025394839317,0.4164982108336835,0.0277671220404483,0.2908559191629068,0.3791270456905583,0.3060713667944655,0.4863806585270583,0.4268018137452364,0.3480233427849989,0.3196646060754841,0.3004097597308133,0.2367052560189078,0.5355352613712171,0.0370059773769389,-0.0656901843211558,-0.1636074368944425,-0.114602223184945,0.2660164424400479,0.1704321356701683,-0.0557872030846647,0.1464064137696973,-0.2427910940448208,-0.1747219313435405,0.0220698612870311,0.0479271279546488,0.2705117846998979,-0.0924202932976024,-0.0738066072858259,-0.2633339387642121,-0.0319914617300483,0.0061013001961771,-0.1077273806680925,-0.1785793755122202,0.0958298747245399,-0.1262527515133805,-0.1741060838405636,-0.2036461271187226,-0.3192619985835874,-0.2849535577985769,-0.1501890399503271,-0.1983635223157227,-0.4192584302405002,-0.2935406812967718,-0.1723513844741047,-0.0891759818534543,0.0072975269118534,0.0615033650749261,-0.1245304317655464,0.1157519182501469,0.1035118422143835,0.173695964218876,0.2425237666317463,0.1293224011422447,0.2118570764194632,-0.1879729183547609,0.1046759357919668,0.2464532187590173,-0.046277400239756,-0.3387332600690637,-0.1324737510663871,-0.2001975914177492,-0.2862706396133731,-0.1400514797378415,-0.0215680538773212,-0.235905802928061,0.1993643595480957,0.5098916988790567,0.5433161618772034,0.2622062990615863,0.3835823968586566,0.1301543274589303,0.0434851119397388,-0.1589848766147189,-0.276812400214877,-0.2675953138850436,-0.2313740504507252,0.0446242804699937,0.0695932917991951,0.1137383255120897,-0.3078593017377318,-0.3425168337939281,-0.2983304662088161,-0.4033233210396894,-0.1433435326359185,0.2206865476211576,0.1894346221952253,0.0,0.1015489128944999,0.9082192682179852,0.6146104305612878,0.0773332034031703,0.3305952311556807,0.304950150083861,0.2031497794712116,-0.0065919816821528,-0.0678831987265137,0.0318295488623867,0.3871159694399678,0.6179991667108095,0.3327867287640369,0.2642562421479508,0.2040702996527512,0.3726137259881612,0.8605870925058864,0.3328169534848384,0.2889038197021033,0.5389965007326869,0.4560506048962706,0.6618538839382218,0.5162843818097418,0.6070963261684171,0.2885957317204556,0.4524270475920407,0.6635883783184009,0.513787335260319,0.5795687761855937,0.6174089209334622,0.627621562113148,0.6475366693078929,0.5312344953971978,0.1941560144409573,0.4818027355891554,0.4604494164409239,0.4598718303153287,0.3453238434361506,0.4226628361713191,0.4224146664219376,0.5600204005243941,0.6775523226020389,0.3679513320066669,0.4272232886368811,0.0509328690817924,0.0439824626013288,0.2816832542225274,0.2981169497443604,0.3398678256223511,-0.0240049932672579,0.0860746087712429,0.2852340908131408,0.3257583654974217,0.5270783715799667,0.2390743733864041,0.0992549580953412,0.1531849627072994,0.2072280960083103,0.2305236586118323,0.2876820724517809,-0.0920746789461764,-0.0738888420991582,0.1146448315117806,0.0709517359722844,-0.0544352065550171,-0.1576289442035831,-0.2537182755923005,0.0064725145056175,0.3014753945841168,0.1476359988060646,0.1317692776311233,0.4776275543563949,0.8012214020893679,0.1120050583764948,0.0253178079842897,0.3607649291902574,0.0716559889064351,0.0531098253139483,0.3109389346160482,-0.3396185423105779,-0.2505425255023241,0.2206086957110217,-0.3256132947751904,-0.1195451506497826,0.2000951558779188,-0.3364722366212129,-0.3414432493432334,-0.3829922522561059,0.4141231708512789,-0.0773494660778928,0.3144933299024378,-0.2797392189378442,-0.393769068344973,-0.1836923589277333,-0.0253178079842898,-0.2694761079552085,0.1074859149701397,0.0969922659873097,-0.4074832722644016,0.3872375943488992,0.2951170509392991,0.3091045080391877,0.6498971967661287,0.597499105837724,0.4298565612323235,0.4084636111044209,0.4776275543563949,0.4233227255081708,0.5915673877420251,-0.0098522964430115,0.3329386640399018,-0.0257524961024147,0.3083013596545166,0.1388364448542158,0.4399512841793337,0.8178506590609026,0.3941655528542311,0.7679663397146976,0.457936159845618,0.476476143751353,0.5655016201126839,0.960895038110909,0.7766681526098065,0.7300398969403109,0.6248279365824682,0.938853069181242,0.6662177686813139,1.0833448165373212,0.6718697821126605,0.7404000654104909,0.2481347753635264,0.7998313231482799,0.2431664671893174,0.4004775665971254,0.254701391721922,0.9670796974286238,0.4023449807719207,0.5714502455824255,0.0070922283094918,0.4324209180966926,0.3948267099031087,-0.2287353645799876,-0.125163142954006,0.3458730109059187,-0.1700082965605976,0.4527618228112295,0.3566749439387324,0.6148949836601971,0.1289921201911033,0.5705448584676129,0.3805970415301509,0.4393666597838457,0.1773340152829154,-0.2779258975064162,0.4975803970159699,0.3769499378001431,0.6474771437267571,1.0385083645984043,0.4558189942537791,0.2118439960602763,0.093090423066012,0.2170645052378276,0.3117796240308415,0.5533852381847866,0.5855165163675798,0.7259370033829362,0.3669988272803684,-0.2661095871976531,0.2498117983963711,0.0266682470821612,0.0,0.2837681731306446,0.3197703870032813,-0.325422400434628,0.0545589842504344,0.0953101798043249,-0.2578291093020998,-0.1736634940508401,0.3094220590881869,0.3260915205666519,0.4787373092144901,0.8443781502838688,0.7904104859851522,0.3501150130249995,0.5207759546191587,-0.0061919702479211,0.2355660713127669,0.5272194335416671,0.4626235219481129,0.6190392084062235,0.6332490389788763,0.3764775712349121,0.2894661942452823,0.1384023228591191,-0.0118344576470027,0.7884573603642703,0.4700036292457356,0.4362367667749181,0.0382212128201976,0.3617900446055029,0.1880522315029396,0.4971735345976634,0.068137805167218,0.3112125698619751,0.410914712875729,0.6151856390902335,0.639344474566969,0.5648928450362665,0.6725278933572095,0.5568107370078141,0.2776317365982795,0.7621400520468968,0.7635279773217449,0.921156921525328,0.4605248852911918,0.8472978603872037,0.7777045685880083,0.7525706010307461,0.664159643686693,0.2231435513142097,0.0387145121806904,0.0763729787845739,0.4924764850977942,0.4519851237430572,0.8472978603872037,0.7363193524251541,0.6931471805599453,0.5845133395571499,0.7354495602506347,0.7521807121441383,0.7320625968096189,0.3622929362429557,0.1582240052148941,0.6211736809348561,0.3383258052705358,0.5306282510621704,0.3215836241274623,-0.2369515242805445,-0.1414116540498285,-0.0096619109117368,0.372239460479844,-0.1950605825713843,-0.2847103020626233,0.0209431738452431,0.0857668217574251,-0.1974351946040028,0.0993724738132037,-0.1857171457950928,0.2507950826447199,0.1306201824170642,0.7997569156182036,0.6887898751909896,0.3611845344447076,0.4151270190199013,0.2231435513142097,0.4026120391257579,0.2006706954621512,0.1900436028878648,0.2349935912317056,-0.3292377427202801,-0.4645540244781709,0.2411620568168881,0.5483162326814893,-0.0074906717291576,0.4168938039317871,0.3150810466398954,-0.0088106296821549,0.2336148511815051,0.5793120379111468,0.651598177647073,0.3398678256223511,0.3390900391632918,0.3750059006234559,0.2353591390823337,0.2145964907357513,0.3033073903548618,0.4090817485783528,-0.0761236954728006,-0.0396652563924314,-0.0398459085471996,0.2793832696370859,0.1933713929805396,0.0287101058824313,0.4228568508200337,0.2528353411216128,0.4014409578084389,0.4165149442947493,0.3993860620317822,0.0224728558520585,0.1612681475961223,0.3715635564324829,0.0702042586732485,0.9555114450274363,0.1264139248556587,0.092373320131015,0.6151856390902335,0.2392296890658341,-0.0224728558520586,0.3413031638908784,0.6931471805599453,0.4982468415591305,0.5218754599525756,0.5527898228650229,0.4679854650894984,0.6330432564902398,0.8183103235139513,0.8472978603872037,0.7041970167465302,0.5639354490799391,0.6931471805599453,0.3184537311185346,0.3393540829961017,0.4663738611951568,1.1160040313799788,0.5668534552356531,0.2113090936672069,0.6931471805599453,0.1541506798272583,0.4809726606163096,0.7630168605204314,0.4353180712578456,0.8194409058842373,0.5313919013474677,1.1831696766961728,0.4539174914941112,0.3548213752894094,1.0252810155825602,0.9444616088408516,0.6931471805599453,0.7519876805828787,1.421385680931161,0.9382696385929302,0.7621400520468968,0.6561059088795963,1.01160091167848,0.9631743177730056,0.7453329337305156,1.0169342576538425,0.5642134971840521,0.5187937934151675,0.9219887529887928,0.7602864833975739,0.3920420877760237,1.2144441041932317,0.4989911661189878,0.2436220826577505,0.4128452154057869,0.6931471805599453,1.0185695809945732,0.9520088144762344,0.6268977950187448,0.8842024173226545,0.5505330733611034,0.6931471805599453,0.9162907318741552,0.6208265189803193,1.1118575154181305,0.9985288301111273,0.3403258059372029,0.0,0.98794672078059,0.623188591953035,1.3862943611198906,0.7985076962177717,0.9279867716373464,0.3409265869705932,0.4367176516122688,0.3074846997479607,0.0971637484536476,0.8109302162163288,0.7389567165912395,0.9903987040278772,0.339507140316367,1.2606681480024813,0.5474353693785518,0.7860409274526415,0.5937747067467416,0.6308432972237904,0.4312176042105791,0.5766133643039939,0.9309968792638508,0.1407725538810822,0.5292593254548287,0.8979415932059585,-0.0839653801173342,0.4554755286828258,0.6678293725756554,0.7308875085427924,1.2527629684953678,0.5035263212843791,1.0537617225027578,1.0345904299031787,0.9911920400472728,0.9852836033611064,-0.1512309697239235,0.9262410627273232,0.7529656757708555,0.4112960284189576,0.7550225842780328,0.2564295289476765,0.3429447511268304,0.3448404862917296,0.4574248470388755,0.962036753662359,0.5108256237659907,0.9343092373768334,0.5596157879354227,0.7884573603642703,0.9985288301111273,0.4198538455602641,1.1927995037278107,0.3031862589877462,0.5780778507751579,0.8329091229351041,0.3016683144265208,0.1019991679551215,0.377294231141468,0.3416370665741918,0.365045609065269,0.2498427576694499,-0.1476359988060645,0.180261823830944,0.2341933875007947,-0.0130720815673527,0.3684238364278154,0.1364914342529995,-0.0222506089348198,0.6115973135958376,0.5864083049319977,0.7638929008743112,0.7808527609790508,0.3024098791620244,0.3244960455744972,0.1561191843999305,0.0368336274039108,0.2414058106267483,0.1865261880759864,0.2285343999490862,-0.0962469480227116,0.116072171252754,0.1210456812911998,0.4409046779664854,0.3333275799417412,0.4324937804960837,0.44873308835668,0.2600305411212708,0.6475696720636259,0.0686285063890963,0.3471054929225817,0.2876820724517809,0.4958491695764334,0.3895665220403661,0.3095647837012885,0.4172995657551672,0.1670540846631662,0.2542341383842409,0.3341275696619588,0.3302416868705768,0.2562188029959958,0.367365261875894,0.3179135312314448,-0.1506061479815329,0.0752624845005467,0.0165841280155352,0.3378718169756361,-0.1433397637230426,-0.1870948355466122,0.0521857531705702,0.1792014294577109,0.0888312137066158,0.1862508349338442,0.4530060528771645,0.4721564826068366,0.4146817632130884,0.0823709974331275,-0.0549588842807574,0.1571855835224123,0.1431008436406732,-0.0121213605323448,-0.2939912416450457,0.1275966571047081,-0.0592556962090931,0.1419702612703872,0.208544751893057,0.3728889376735516,0.3530648317146326,0.5888343339345627,0.2185528776056108,0.1722459048052131,0.4263426931297196,0.3462762367178337,0.3063742054639334,-0.0689928714869514,0.5537276453102002,0.7156200364120039,0.9118363815247748,0.3654597734944652,0.4740936144972607,0.2270574506353461,0.3237870770938971,0.3325326386171327,0.2588616339162889,0.386772975096012,0.1738469298029823,-0.1218898176090368,0.0655972824858132,0.6931471805599453,0.3471961999841885,0.3022808718729337,0.0270286723879194,-0.2305236586118324,0.125163142954006,0.4228568508200337,0.661398482245365,0.7318616927406357,0.4883527679139321,0.4192584302405002,0.3837251214717585,0.4480247225269604,0.4353180712578456,0.048790164169432,0.5268259651124318,0.8517522107365839,1.0986122886681098,-0.1226023220923324,0.5491078103370076,0.2876820724517809,-0.0741079721537218,0.3364722366212128,0.544727175441672,0.1603426500751794,1.01160091167848,0.5679840376059392,0.2231435513142097,-0.1000834585569825,0.0571584138399486,0.2135741002980591,0.7503055943998941,0.8754687373538999,1.2527629684953678,-0.4519851237430572,-0.4307829160924542,-0.03509131981127,0.1670540846631662,0.2478361639045813,-0.1769307081590782,0.4177352006999788,0.2876820724517809,0.0645385211375711,0.666478933477784,-0.0425596144187958,-0.2429461786103895,0.2006706954621512,-0.1206279877886147,0.0,-0.1625189294977749,0.3364722366212128,0.6097655716208943,0.6632942174102642,0.8823891801984736,0.4054651081081644,0.6109090823229731,-0.1967102942460542,0.1978257433299198,0.5108256237659907,0.1091992919649919,0.6085897925318823,0.4054651081081644,0.6418538861723947,0.6359887667199966,0.6931471805599453,0.5039051809214169,0.3254224004346279,0.4212134650763035,0.6190392084062235,0.7282385003712155,0.0408219945202552,-0.1144103511777442,0.8586616190375187,-0.2542341383842409,0.2461330695389085,0.8064758658669486,0.3184537311185346,0.5725191927713306,0.4877032063451365,0.7949298748698878,0.7184649885442351,0.3364722366212128,0.7922380832041763,0.8909729238898652,-0.1112256351102243,0.4382549309311553,0.3894647667617233,0.5260930958967791,0.2984929885559966,0.0377403279828471,0.6931471805599453,0.4274440148269397,0.6505875661411494,0.0540672212702757,0.7672551527136672,0.4382549309311553,0.1910552367627092,0.4307829160924544,0.0465200156348929,0.7537718023763802,0.2876820724517809,0.4626235219481129,0.6931471805599453,-0.037740327982847,0.0606246218164348,0.0645385211375711,0.1988508587451651,0.2468600779315258,0.4260843953109001,0.9864949905474036,0.5839478885949534,0.6931471805599453,0.0,0.0392207131532813,0.6701576623352465,0.4769240720903094,0.2468600779315258,0.3513978868378886,-0.0870113769896298,-0.0741079721537218,0.9162907318741552,0.4700036292457356,0.3794896217049036,0.4418327522790392,0.9382696385929302,0.2006706954621512,-0.4855078157817008,0.3483066942682158,0.0689928714869514,0.5179430915348546,0.7537718023763802,0.9808292530117262,0.2682639865946793,1.3217558399823195,1.029619417181158,0.3677247801253173,0.2513144282809061,0.4700036292457356,0.5947071077466928,1.871802176901592,0.0,0.9382696385929302,0.6931471805599453,0.4855078157817008,0.0,0.4054651081081644,0.3254224004346279,0.2513144282809061,0.7827593392496325,1.1676051601550612,0.5877866649021191,0.3894647667617233,0.0689928714869514,1.0360919316867758,1.8325814637483104,0.1625189294977749,0.8690378470236095,0.1728428128394108,0.494696241836107,0.5639354490799391,0.0870113769896297,0.4489502200479032,0.0377403279828471,0.3746934494414107,0.1484200051182732,0.4212134650763035,-0.1335313926245226,0.0,0.9162907318741552,0.1823215567939546,1.01160091167848,0.0,0.1670540846631662,0.965080896043587,-0.1177830356563835,0.5108256237659907,-0.4054651081081645,0.3184537311185346,0.2076393647782445,0.2719337154836417,0.1941560144409573,0.4054651081081644,0.5306282510621704,-0.1718502569266592,-0.1823215567939546,-0.294799540220645,0.0342890734786321,0.0,0.8426789145309092,0.2231435513142097,0.0240975515790605,0.9279867716373464,0.3746934494414107,0.2076393647782445,-0.0307716586667536,-0.3364722366212129,0.0953101798043249,-0.0800427076735363,0.2006706954621512,0.2682639865946793,0.3829922522561057,1.029619417181158,1.0986122886681098,0.0465200156348929,0.5198754592859086,1.01160091167848,1.3581234841531942,0.0465200156348929,0.9382696385929302,0.6567795363890705,-0.1625189294977749,0.1984509387238382,0.6931471805599453,0.2135741002980591,-0.5072478024181066,-0.1579030294458087,-0.3654597734944652,-0.037740327982847,-0.3934889170614488,0.0555698511548107,-0.1872115420881464,0.0594234204708008,0.4855078157817008,-0.2876820724517809,0.262364264467491,-0.1910552367627092,0.093090423066012,-0.4480247225269604,0.1213608570042675,0.3136575588550416,-0.4700036292457356,-0.0953101798043249,-0.0408219945202551,-0.4700036292457356,0.1541506798272583,0.2363887780642303,0.6486954179891115,0.0689928714869514,0.125163142954006,0.5596157879354227,0.1823215567939546,0.9343092373768334,0.0,0.0,0.2113090936672069,0.3136575588550416,-0.0339015516756813,0.4418327522790392,0.5108256237659907,0.5920510636885766,0.5108256237659907,0.3513978868378886,-0.9343092373768336,0.9808292530117262,0.9444616088408516,0.1177830356563834,0.1226023220923322,0.6061358035703155,0.0540672212702757,-0.0870113769896298,0.2876820724517809,0.6061358035703155,0.6701576623352465,0.4760826753221178,0.5679840376059392,0.0984400728132525,0.4818380868927384,-0.4149438520627082,0.2035989552412395,-0.2113090936672069,0.4144337780909249,-0.1313360020610869,-0.7191226669632059,-0.1000834585569825,0.1961148789262903,0.1910552367627092,0.5280674302004967,-0.038466280827796,0.2468600779315258,0.0,0.1758906664636642,0.2947995402206449,0.4626235219481129,0.9985288301111273,0.5925036547802579,1.067840630001356,0.3939042857070885,0.659245628884264,0.3690974639372896,0.627189212768148,0.4187103348581849,0.2429461786103894,0.0635134057223259,0.3819346106979702,0.2231435513142097,0.3101549283038394,0.5798184952529422,0.3583975972501787,0.3942918075100392,0.5355182363563621,0.1844290391335192,0.2231435513142097,0.0347861160854156,0.3417492937220567,0.3483066942682158,0.3976829676661094,-0.4519851237430572,-0.0896121586896871,0.0,0.0560894666510435,0.5240708505160113,0.3014753945841168,-0.0615578929994333,0.8150369981689821,0.2336148511815051,0.7731898882334817,1.2809338454620642,0.0984400728132525,0.1823215567939546,0.4653632496892333,0.6931471805599453,0.2113090936672069,0.5705448584676129,0.4274440148269397,0.1484200051182732,0.0896121586896871,0.1941560144409573,0.5930637220029628,0.5819215454497211,0.328504066972036,0.3101549283038394,0.7221347174331976,0.723918839226699,0.2231435513142097,0.5877866649021191,1.01160091167848,-0.0870113769896298,0.6286086594223741,0.4054651081081644,0.7472144018302211,0.1625189294977749,0.4228568508200337,1.6739764335716716,0.5306282510621704,1.1260112628562242,2.0149030205422647,0.8023464725249374,0.6678293725756554,1.01160091167848,0.7985076962177717,0.5306282510621704,0.6021754023542185,1.0459685551826876,1.181993897607161,0.7435780341868372,0.8056251639866356,0.5798184952529422,0.041672696400568,0.6931471805599453,0.2425616371713113,0.9308188324370648,0.3254224004346279,0.5920510636885766,0.6061358035703155,0.2876820724517809,0.1582240052148941,0.4054651081081644,-0.0400053346136991,1.0169342576538425,0.3144933299024378,1.0459685551826876,0.0,1.1260112628562242,0.2363887780642303,-0.209720530982069,0.0,0.8023464725249374,0.2436220826577505,0.1823215567939546,0.6931471805599453,0.6286086594223741,-0.3930425881096072,-0.2776317365982794,-0.0281708769666963,0.4274440148269397,0.0870113769896297,0.9162907318741552,0.4446858212614457,0.093090423066012,0.0953101798043249,-0.213574100298059,-0.2787134024690204,-0.6649763035932491,0.0317486983145802,0.3158529494184772,0.6466271649250525,0.7939518796819108,0.9675840262617056,0.6931471805599453,-0.125163142954006,1.157452788691043,0.3429447511268304,0.6931471805599453,0.4653632496892333,0.8909729238898652,0.2384110234449981,0.624154309072994,0.1177830356563834,0.5819215454497211,0.3074846997479607,0.1698990367953974,0.7221347174331976,0.1823215567939546,0.7731898882334817,1.1856236656577397,0.3877655310087634,0.7375989431307791,0.6061358035703155,-0.5753641449035618,0.4054651081081644,0.4418327522790392,-0.1335313926245226,1.1394342831883648,0.9694005571881034,0.6061358035703155,0.4248831939652659,0.9162907318741552,0.8979415932059585,1.0704414117014132,1.3545456628053103,1.3633048428951922,0.7282385003712155,0.8708283577973976,0.7419373447293773,0.780158557549575,0.4895482253187058,0.8472978603872037,0.9067212808580044,1.0233888674305225,1.0809127115687087,0.7282385003712155,0.3364722366212128,0.4519851237430572,1.0473189942805592,1.0414538748281608,0.3958956570920137,0.6931471805599453,0.7472144018302211,0.7308875085427924,0.9007865453381898,0.5232481437645479,1.0986122886681098,1.4403615823901663,0.9249487946172698,0.3364722366212128,0.2719337154836417,0.4769240720903094,1.1676051601550612,1.0340737675305385,0.7166776779701394,0.7654678421395714,0.4989911661189878,0.6007738604289301,0.7731898882334817,0.3946541920039488,0.3651138125845969,0.6443570163905132,0.8649974374866045,0.2662679779479644,0.2829539312558349,0.4416166061634242,0.0947918809646533,0.2302076928598909,-0.2221592300647989,-0.1233790211605046,0.1188287327086928,0.2231435513142097,0.3435258086678738,0.1730548344664109,0.3831427526702661,0.2184670034303079,0.1080046746274606,-0.0912582495963559,0.2881673915586133,0.3002252269881525,-0.1254251288082684,-0.2642452269997617,-0.0048543784647981,-0.345269947324742,-0.0784219617758031,0.1997961873172186,0.0800427076735363,0.1414995622736994,0.049117014964955,0.3197875586597457,0.4567114530367987,0.4544301127361238,0.2881483816232057,0.116682320080198,0.1150693297847872,-0.2555562161892692,-0.2074210638859822,0.011799546931155,-0.0302636576398269,0.2264453181507664,0.118050737438199,-0.0849095461741856,-0.0784897400846668,0.1341343471197268,0.177295114111337,0.256357303314768,0.1179846688862228,0.0013175232472406,0.1184738819705613,0.0817361881798695,0.1103774953413456,0.0257594065783507,0.1311532702195552,0.0538198207931317,-0.4858954019552635,-0.2676228217209581,-0.191323254239053,0.3282736460984303,0.0392573774367735,-0.0021299262578248,-0.2211872415934488,0.0774665384298444,0.150664236923263,-0.171123438759097,-0.2616098321420058,-0.2916498685314057,-0.0779001647907847,-0.0300641563922823,0.0317739199497001,-0.1892702869813588,-0.4004775665971253,0.0173801166299167,0.0977314884933352,0.0417259336702184,0.3645444748030591,0.0140064081730232,-0.0919544719573525,0.2516884698038755,0.1263548014862851,0.2304907957649209,0.1470424108018461,-0.1254030877678503,0.1714715407698888,0.1711696170989691,0.111149243834071,0.216470134757711,0.1680873967661301,0.0709091265726444,0.0846386220351626,-0.1705241224138489,0.107073333661501,0.5909578647341225,0.6913202789328509,0.6811016877286808,0.4292485068600835,0.2542105205832406,-0.1526926578188893,-0.3242629215660397,-0.1099788520317329,0.2688857804900432,0.1652058738790555,0.1672513038729564,-0.0687780717868451,0.1109967759720244,0.2421239550367284,0.3008185503571509,0.2805480728881423,0.153477052317784,0.3953042579363013,0.3415699536928816,0.2589591208426731,0.3530777842359883,0.3823356534143941,0.3763663324050298,0.3363668014373944,0.1967495140850226,0.1050735892524803,0.1662898387278809,0.2202226554752923,0.0031371871739043,0.1816109854000827,0.3147443723724327,0.2274179004094418,-0.2681046145203137,0.0887541549584831,-0.2264376626016468,0.1532589445324647,-0.189990306033255,0.0467512729530495,0.1002191161574075,0.3176654562081028,0.3103360385098946,0.3590340819224004,0.2098829173119951,0.2347949532021565,0.2310775848627501,0.1746574183241348,-0.1348607335632076,0.0439901713536826,-0.12032790255597,-0.3760929048963703,-0.101727277126291,0.1611199331107031,0.0622068404995561,-0.2023798284933996,0.0085913352194629,0.0424164673534127,-0.0361835804106688,0.0800248308981486,0.1600457717026755,0.1409109677172281,-0.2429364924891482,0.0464437467106291,-0.1201022379856093,0.0323313438190287,-0.3223021668925604,-0.4461592989347187,-0.2779312419761993,-0.1006729943146613,-0.022044708195743,0.0778849185692786,-0.0435307321175513,-0.0157925724745244,0.1554343945819012,0.1066935629976514,0.1056141296586784,0.2017266060190513,0.1377326381696191,0.1003456289292331,0.1391093432505763,0.122718714267584,0.0736793815202202,-0.120237572200551,0.1261886871895171,-0.2059778383900977,0.0794051964954997,-0.2308322419489957,0.009528244326458,-0.1134463707553333,-0.2796589012900485,-0.3555589735248929,-0.1931912290308584,-0.4635296955730191,-0.2217837047434412,-0.0992581305382332,-0.2644275329647908,-0.3942032581122338,-0.2651643079803767,-0.0746560074609113,-0.0473017967760019,-0.1519643243810646,-0.290321187639719,-0.332796899065739,-0.2373314218872037,-0.1134452648629671,-0.0712080184539321,0.0688977647295447,0.0912397130991509,-0.1968940373089196,-0.1920440332769021,-0.0614782878409771,-0.0447517854695685,-0.3027466019131768,-0.1822767046937475,-0.0308873754912825,-0.2299769938307627,-0.2797213654613254,-0.2911308922991425,0.1683623221381999,-0.0334027077697704,-0.1043559034905911,-0.1012662145159264,-0.2137509193242468,-0.1414116540498285,-0.3348313130553531,-0.3189546355397494,-0.3954322098593128,-0.0909081184036872,-0.0035524016043677,-0.1552609275265153,-0.1807392043439052,-0.1783819587898743,-0.0730828567384765,0.1301887711278328,0.1176895820793798,0.0667050257235286,-0.0885072635812124,0.0704442505515984,0.0558214287573452,-0.0467424438100453,0.0122967191411482,0.3550620403632676,-0.011618387953865,0.1504101498236544,0.3619023473738841,0.3454008673655143,0.3162339743802218,0.2594139263169024,0.1990103929057556,0.4388779772178665,0.0785810752574134,0.0437387500629083,0.0690532014920977,-0.1146754868573818,-0.1603426500751793,0.2411620568168881,0.1566608473182928,0.6054159475104359,0.698645675096074,0.5791056521670359,0.401767057426997,0.3978241922319857,0.3995653859809761,0.3222876016292564,0.1675964799411171,0.2643532353790429,0.3031192422383207,0.2287910941471175,-0.0390271551160592,0.0619090165985464,0.21005910516479,0.4386335173660577,0.4692196229335268,0.3469391229486508,0.237442200101959,0.3990156860959777,0.6901391273318567,0.4658904122038274,0.2785195509334042,0.2501374782559506,0.2367529441279782,0.3023463243953336,0.1340696837702092,0.1711482561958294,0.3228780019282748,0.328514294483018,0.3829305543291365,0.2157460730610726,0.4160978936439577,0.2878707338993739,0.2541152488126977,0.1951276803743512,0.1941360402077176,0.4244148568878265,0.354133273867189,0.4852577636862117,0.3560967421301982,0.3932715913739045,0.4729013671469358,0.6082202526504462,0.5298202152417245,0.5342995704855444,0.3447608906376102,0.2223540393162484,-0.189612485667504,0.4403975085706694,0.3311525511374497,0.1824959399141105,0.2490777912491935,-0.1412960604767024,0.1242422940337181,0.3126499320781033,0.1487452612603504,0.3736875817958307,0.4082939643086421,0.0930348659671893,-0.2695375152046947,-0.229246651979016,-0.2101844066717046,0.0939900070691044,-0.0949201750636096,-0.2974192477296392,-0.1247151477251289,0.0922113803869678,0.1115676908686476,0.2430737377661208,0.3318034612843217,0.2709770522517782,0.687308088879181,0.3941065823065337,0.4967754676612935,0.3298517117161786,0.2835476205015783,0.4479737528284351,0.4409954904402085,0.3344152667951917,0.1305044279261175,0.1466034741918754,-0.1463023186689603,0.2227019527201488,0.3609017259368129,0.1965786835503017,0.2395660995689722,0.2362832983953387,0.1250008019970554,0.10716883441461,0.2342765081837621,0.2162005856934776,0.1394982347837095,0.1236347756376019,0.1332283898648792,0.2137717092306257,0.2474478212402596,0.0642879259727463,0.1734081847770846,0.3579963036631301,0.3389937760955059,-0.0440696596178947,0.2095438681426377,0.4506672098097549,0.221241807476392,0.4368288595155585,0.2340293589594609,0.0720240953902344,0.3506315394451143,0.1504039538256495,0.326016401046094,0.4894363090319714,0.1502505188771634,0.183471643177192,0.1558104312456227,0.2729913570417773,0.3825705070682932,0.231749195895168,0.5416553696216312,0.1167665406282306,0.1012268705795123,0.3760399328433321,0.1049337254187472,0.1284693015790108,0.0971445680330204,0.2667513838583342,0.3092474002328093,0.0836869688328147,0.2299057731861521,-0.0322006355400433,0.1946360912624679,0.0864151502354568,0.1861393961736136,0.0877913473902974,0.2700971676763403,-0.3979392556400763,-0.1324653947182783,-0.3850034611244114,-0.043341551708173,-0.0574409196685102,0.1102031401336142,-0.1404446139228911,-0.2769353173257474,-0.2169256517049392,-0.2761636204955102,-0.0683891227378564,-0.1652783910899646,0.1574649502475642,0.2374295085616864,0.1253844675828075,0.1694120202050704,-0.1120812367442394,-0.0523146357587109,0.0195445960729701,0.2936040278472447,0.2137800862505455,0.2330236599139169,0.0845044270691042,0.2769652920551189,-0.0326707822895487,-0.0453279564409611,0.0241947285870569,0.1775989978397812,-0.0879150315361726,0.2782797197733905,0.4022517377338747,0.1737400912949087,0.1658109476544237,-0.2693280051380501,-0.2141479884056318,-0.2575308936236856,-0.2585604064119403,-0.2021991634708397,-0.0846849205635247,0.2365746442676516,0.2098883866266092,0.0530315709088376,0.1099398581953575,0.0320260267194084,-0.1490676968396671,-0.2292500351774295,-0.1492532622872771,0.0272950985894378,-0.4193181434974581,-0.2978142739637826,-0.3792093290775677,-0.5196583926817867,-0.1673904263895277,0.0446842396147727,-0.1344372844672477,-0.044509120109064,0.0461366829357533,-0.1289336833758931,-0.2161445631292978,0.0057400731610599,0.2673390241053314,0.1092153377918738,0.3413226907549623,0.3457111053097441,0.1289563362545086,0.261739459698649,0.1933552265961107,0.1589933043030463,0.00126422267154,-0.0084507545177232,-0.3162695293036936,-0.1744683230681271,-0.0809833716535849,-0.0508700736716333,0.1318380808000279,-0.106468549012188,-0.2064919177217676,-0.4209949097262913,-0.3826024646853941,-0.0826917158451134,-0.3144590956373585,-0.0143887374520995,-0.1600105117853734,0.0899310870929645,0.1953087523207657,-0.0561096460458641,0.0498203486635496,0.153039512482261,-0.0628411847196436,-0.2197247445654241,-0.1302084680543353,-0.1487109565314401,-0.0861418117174454,0.0119881555599696,0.0823362866606358,0.1631671343531982,0.1844815642799452,0.0958295254405111,-0.0333629036691214,-0.4028575447010835,0.0442686794505861,-0.0865680163084589,-0.0953101798043249,0.0462753433721665,0.0677037962198213,0.2160701354384829,0.1577732755886933,0.1739231870951228,0.2807563040988208,0.3205244314590808,0.3645215404311125,0.2308947161300043,0.0927420516986147,0.3033189641833494,0.2694920256501142,0.1214975530607989,0.1147191837929228,0.192910942688498,0.120752105005073,0.2633430041465133,0.0011037528714377,0.0931506849037149,-0.1131906050821733,-0.3329501271316124,-0.2157974859903441,0.0009315324381114,-0.0862031095245769,-0.1376644062102124,-0.2396307098513761,-0.1155876450929477,0.1023620126615696,0.1297592267030991,-0.0171127170676023,0.016682499959936,0.0015515906914188,-0.2128291658178416,0.2809500113177232,0.2534669920072669,0.2978207196231145,0.0699010336563054,0.0116415750154857,0.0338060361304536,-0.3175224878415508,-0.0185190477672375,-0.1552102825754047,0.1987137476751902,-0.1242547598134216,-0.4210108513586618,-0.4901701739036079,-0.2259204408482709,-0.1971819701191451,-0.1177830356563835,-0.0739167980253716,-0.3320558580569708,-0.1137432790111176,-0.1994351740694599,-0.3169524547518942,-0.4004257273647113,-1.280879351034106,-1.2130960303014582,-1.2924341835534523,-1.2815097579392882,-1.158376906745809,-1.1284652518177911,-0.9988994063614308,-0.8618915052887447,-0.8083134766225816,-0.7042672316040229,-0.4307046657696318,-0.3564471275577493,-0.0617532322065866,-0.5462122183151787,-0.2361338711038901,-0.3920174726943619,-0.4671640627376578,-0.6761845533403661,-0.3913693383077709,-0.0789066666538839,-0.0499212383753692,0.2664214983055168,-0.0230376680670173,0.0212397365009109,-0.0013395849290564,-0.1125052258158495,0.2495005744789841,-0.2320468881523538,-0.1771177289189276,-0.2858896360514661,-0.3072521686458782,-0.3382563584147143,-0.4013413909243023,-0.5427185650920264,-0.3305348417072881,-0.2589436112086249,-0.194775887593438,-0.2215702619415273,-0.4454157012929891,-0.3701294329051985,-0.1531224021016922,0.0317486983145802,-0.1403450720034529,0.1399008409100025,0.3088974619325493,0.0334396482052925,-0.0750846301081006,0.0040568006956144,-0.0509664435920275,0.0,-0.1588115492976012,-0.1004842869409982,-0.0505405027498087,-0.2316674308971096,0.041700729198944,0.1395015596110264,-0.0648876222058748,-0.0957593824771474,0.1232500804086013,0.2131319158999631,-0.1048078767932499,-0.2407176123651276,-0.200103239070133,-0.1999764920326754,-0.0351449463027698,0.1148048285004139,0.1388364448542158,-0.0165488410244728,0.0775988283717926,0.1622519899579952,0.1299254224821088,-0.0778705455189689,0.2806521665235228,-0.0998595194674788,0.1337176643955743,-0.0257083567102069,-0.0793017708838838,0.0885276775075936,0.1799594838212388,0.2478862452868378,0.1823215567939546,0.2915369922422264,0.2680536927081991,0.2412336977525494,0.2555575856480112,0.112926756041142,0.0182154398913411,0.0,0.0928937468926962,0.2454325259403403,0.0265894541892394,0.1617316947703468,0.3150435604629188,0.3967819855347036,0.2173655740343211,0.1219189352004318,-0.0877542008153713,-0.0995115397538612,-0.100968414366949,0.2776951300497571,-0.4050917648146735,-0.2545710971474209,-0.276472987917061,-0.3215836241274623,-0.0430173850836908,0.1553921449153232,0.2039121893863222,-0.065622318475049,-0.3767617025054698,-0.0039138993211363,-0.1214100320921778,0.1340586797445697,0.1233364017729387,0.0454132086597016,-0.0673201640994479,-0.0830927585615372,-0.2530422904818431,-0.1854444680984388,0.0647585673481631,-0.2776919793766946,-0.0095511709843429,0.0223057575142981,-0.1648200341906572,0.1527024536508934,-0.0574870909176815,-0.1162748431445454,-0.1674329443002039,0.0193959303475794,0.0276356101720861,-0.0948213017696552,-0.2988333851212781,-0.3770271727876309,-0.1882798663775852,-0.1995275582428684,-0.1277210056465159,-0.1715959657619068,0.048893903990055,-0.1130233884033953,-0.0840494443424517,-0.1687050068027898,-0.2578291093020998,-0.2601321206835503,-0.1693823744227573,-0.1831000698733682,0.0915151088357733,0.0,-0.1271894860992384,-0.0508330657992322,-0.262364264467491,0.3162339743802218,0.3457059125681584,-0.0500939453189154,-0.0195810451991065,0.0836500292869249,0.0074074412778618,-0.4207732273408201,0.2821965951751097,0.1431008436406732,0.2301775779715897,0.2428163169129145,0.3085988105029574,0.2088039406337542,0.054635435388638,-0.157264976956617,0.3286326760648762,0.0985577683350737,0.1666313410508674,0.4807312799175473,0.0245626383684091,0.0488724043931718,0.0413852161628542,0.4790769254514104,0.1458518770125631,0.3623865253866833,0.1766852964797565,0.2975761460969795,0.3818276439538289,0.4180727249627392,0.5368011101692514,0.4054651081081644,0.2556340894254225,-0.3645022784352224,-0.4261365887397592,-0.2234482281052732,-0.1490048708738542,-0.3366418862950441,-0.4054651081081645,-0.389813206408969,0.1456399901593496,0.0224728558520585,-0.2865450626647058,0.2926399408125678,0.3435258086678738,0.1232058759778368,-0.1596301455918839,0.0294327067871184,0.5550999082992658,-0.1852292814117122,-0.1028667487794238,0.067558151011291,0.2121011101825516,-0.1649190044983083,0.0663226429310725,0.0631789016215316,0.0393018523476136,-0.0414997309067527,0.4155917769260936,0.13095606284679,0.3624204884256375,0.2744368457017604,-0.2596994573579049,-0.1559066065295232,0.2667388986065377,0.0927420516986147,0.2555483589243438,-0.1397619423751587,0.0121634861931974,0.0242436116099928,0.011376686982108,-0.197890985541654,-0.3218150788212395,-0.048790164169432,-0.2125021739439459,0.0297128973185115,0.189692064905588,0.3814370419892993,0.2768711563475653,0.4788476540421883,0.161891117402235,0.2027826286652946,-0.0477037979572111,-0.3569481306635566,-0.363843433417345,-0.2554517945468676,0.4079495814358263,-0.0427249446663412,0.1829187500037007,0.2260379093405743,0.3612100991041237,-0.0610876919798381,-0.0610141057551462,0.2110709700799406,0.1441697908295783,0.431011200535861,0.0457360093226842,0.4834991107206201,0.2622154660147203,0.3470931018862572,0.1898378203992284,0.4175134466243388,0.3123746850421525,0.3017253383480715,0.4224146664219376,0.3161058258896693,0.2617420514431651,0.4676886557872947,0.3967694011406106,0.4030201191150979,0.3006831287696277,-0.0287831911195271,0.2728971995161675,0.0980068106519242,0.2395224433982495,0.2170347968258885,0.0036805341511252,0.0842297340277529,-0.1650528451331067,0.1360679255533155,0.2668696667058463,0.3539072984433144,0.2990395860005975,0.4077393870052049,0.6941506929371853,0.4525044693792435,0.319022937091592,0.1655296941240698,0.4606626168525968,0.3070493357456865,-0.0019267828697,0.1067059993174759,0.5795994132058396,0.215767332411866,0.1795603953526267,0.2423799126046787,0.3528594938214814,0.2840168270534849,0.372239460479844,0.1337268387308811,0.399123932769717,0.5534895080352539,0.3173859253577045,0.1781008691897245,0.5136665347675944,0.2440235923395275,0.3061490765766953,0.3041443918016972,-0.0862843685970124,-0.0306369694618898,-0.0526951169861912,-0.1393902642441376,0.090439051844967,0.0437534733509786,0.1611633312866085,-0.2087912576748658,-0.1872517229018133,-0.2609425114098817,-0.0209384071851703,-0.0995984924951012,0.1944479000238821,0.1572317878379036,0.1813109452880311,0.4355199623046633,0.2720374553042649,0.3335755496907026,0.1205340690282732,0.0640714964101313,-0.074733558661456,0.0387145121806904,-0.1166675865725178,0.1604077394103147,-0.1432910300239015,-0.2205427696141523],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"positivity_index"},"xaxis":{"title":{"text":"date"}},"yaxis":{"title":{"text":"value"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4cd688b1-60a9-475b-8334-56811194644b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
# Plotting Market Interest Index
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x = result['date'], y = result['market_interest_index'], name = 'market_interest_index'))
fig3.update_layout(title = 'market_interest_index', xaxis_title = 'date', yaxis_title = 'value')
fig3.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="990c2328-22a7-4798-b63d-2f2cb79f4b95" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("990c2328-22a7-4798-b63d-2f2cb79f4b95")) {                    Plotly.newPlot(                        "990c2328-22a7-4798-b63d-2f2cb79f4b95",                        [{"name":"market_interest_index","x":["2017-05-01","2017-05-02","2017-05-03","2017-05-04","2017-05-05","2017-05-06","2017-05-07","2017-05-08","2017-05-09","2017-05-10","2017-05-11","2017-05-12","2017-05-13","2017-05-14","2017-05-15","2017-05-16","2017-05-17","2017-05-18","2017-05-19","2017-05-20","2017-05-21","2017-05-22","2017-05-23","2017-05-24","2017-05-25","2017-05-26","2017-05-27","2017-05-28","2017-05-29","2017-05-30","2017-05-31","2017-06-01","2017-06-02","2017-06-03","2017-06-04","2017-06-05","2017-06-06","2017-06-07","2017-06-08","2017-06-09","2017-06-10","2017-06-11","2017-06-12","2017-06-13","2017-06-14","2017-06-15","2017-06-16","2017-06-17","2017-06-18","2017-06-19","2017-06-20","2017-06-21","2017-06-22","2017-06-23","2017-06-24","2017-06-25","2017-06-26","2017-06-27","2017-06-28","2017-06-29","2017-06-30","2017-07-01","2017-07-02","2017-07-03","2017-07-04","2017-07-05","2017-07-06","2017-07-07","2017-07-08","2017-07-09","2017-07-10","2017-07-11","2017-07-12","2017-07-13","2017-07-14","2017-07-15","2017-07-16","2017-07-17","2017-07-18","2017-07-19","2017-07-20","2017-07-21","2017-07-22","2017-07-23","2017-07-24","2017-07-25","2017-07-26","2017-07-27","2017-07-28","2017-07-29","2017-07-30","2017-07-31","2017-08-01","2017-08-02","2017-08-03","2017-08-04","2017-08-05","2017-08-06","2017-08-07","2017-08-08","2017-08-09","2017-08-10","2017-08-11","2017-08-12","2017-08-13","2017-08-14","2017-08-15","2017-08-16","2017-08-17","2017-08-18","2017-08-19","2017-08-20","2017-08-21","2017-08-22","2017-08-23","2017-08-24","2017-08-25","2017-08-26","2017-08-27","2017-08-28","2017-08-29","2017-08-30","2017-08-31","2017-09-01","2017-09-02","2017-09-03","2017-09-04","2017-09-05","2017-09-06","2017-09-07","2017-09-08","2017-09-09","2017-09-10","2017-09-11","2017-09-12","2017-09-13","2017-09-14","2017-09-15","2017-09-16","2017-09-17","2017-09-18","2017-09-19","2017-09-20","2017-09-21","2017-09-22","2017-09-23","2017-09-24","2017-09-25","2017-09-26","2017-09-27","2017-09-28","2017-09-29","2017-09-30","2017-10-01","2017-10-02","2017-10-03","2017-10-04","2017-10-05","2017-10-06","2017-10-07","2017-10-08","2017-10-09","2017-10-10","2017-10-11","2017-10-12","2017-10-13","2017-10-14","2017-10-15","2017-10-16","2017-10-17","2017-10-18","2017-10-19","2017-10-20","2017-10-21","2017-10-22","2017-10-23","2017-10-24","2017-10-25","2017-10-26","2017-10-27","2017-10-28","2017-10-29","2017-10-30","2017-10-31","2017-11-01","2017-11-02","2017-11-03","2017-11-04","2017-11-05","2017-11-06","2017-11-07","2017-11-08","2017-11-09","2017-11-10","2017-11-11","2017-11-12","2017-11-13","2017-11-14","2017-11-15","2017-11-16","2017-11-17","2017-11-18","2017-11-19","2017-11-20","2017-11-21","2017-11-22","2017-11-23","2017-11-24","2017-11-25","2017-11-26","2017-11-27","2017-11-28","2017-11-29","2017-11-30","2017-12-01","2017-12-02","2017-12-03","2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08","2017-12-09","2017-12-10","2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15","2017-12-16","2017-12-17","2017-12-18","2017-12-19","2017-12-20","2017-12-21","2017-12-22","2017-12-23","2017-12-24","2017-12-25","2017-12-26","2017-12-27","2017-12-28","2017-12-29","2017-12-30","2017-12-31","2018-01-01","2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-06","2018-01-07","2018-01-08","2018-01-09","2018-01-10","2018-01-11","2018-01-12","2018-01-13","2018-01-14","2018-01-15","2018-01-16","2018-01-17","2018-01-18","2018-01-19","2018-01-20","2018-01-21","2018-01-22","2018-01-23","2018-01-24","2018-01-25","2018-01-26","2018-01-27","2018-01-28","2018-01-29","2018-01-30","2018-01-31","2018-02-01","2018-02-02","2018-02-03","2018-02-04","2018-02-05","2018-02-06","2018-02-07","2018-02-08","2018-02-09","2018-02-10","2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23","2018-02-24","2018-02-25","2018-02-26","2018-02-27","2018-02-28","2018-03-01","2018-03-02","2018-03-03","2018-03-04","2018-03-05","2018-03-06","2018-03-07","2018-03-08","2018-03-09","2018-03-10","2018-03-11","2018-03-12","2018-03-13","2018-03-14","2018-03-15","2018-03-16","2018-03-17","2018-03-18","2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26","2018-03-27","2018-03-28","2018-03-29","2018-03-30","2018-03-31","2018-04-01","2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15","2018-04-16","2018-04-17","2018-04-18","2018-04-19","2018-04-20","2018-04-21","2018-04-22","2018-04-23","2018-04-24","2018-04-25","2018-04-26","2018-04-27","2018-04-28","2018-04-29","2018-04-30","2018-05-01","2018-05-02","2018-05-03","2018-05-04","2018-05-05","2018-05-06","2018-05-07","2018-05-08","2018-05-09","2018-05-10","2018-05-11","2018-05-12","2018-05-13","2018-05-14","2018-05-15","2018-05-16","2018-05-17","2018-05-18","2018-05-19","2018-05-20","2018-05-21","2018-05-22","2018-05-23","2018-05-24","2018-05-25","2018-05-26","2018-05-27","2018-05-28","2018-05-29","2018-05-30","2018-05-31","2018-06-01","2018-06-02","2018-06-03","2018-06-04","2018-06-05","2018-06-06","2018-06-07","2018-06-08","2018-06-09","2018-06-10","2018-06-11","2018-06-12","2018-06-13","2018-06-14","2018-06-15","2018-06-16","2018-06-17","2018-06-18","2018-06-19","2018-06-20","2018-06-21","2018-06-22","2018-06-23","2018-06-24","2018-06-25","2018-06-26","2018-06-27","2018-06-28","2018-06-29","2018-06-30","2018-07-01","2018-07-02","2018-07-03","2018-07-04","2018-07-05","2018-07-06","2018-07-07","2018-07-08","2018-07-09","2018-07-10","2018-07-11","2018-07-12","2018-07-13","2018-07-14","2018-07-15","2018-07-16","2018-07-17","2018-07-18","2018-07-19","2018-07-20","2018-07-21","2018-07-22","2018-07-23","2018-07-24","2018-07-25","2018-07-26","2018-07-27","2018-07-28","2018-07-29","2018-07-30","2018-07-31","2018-08-01","2018-08-02","2018-08-03","2018-08-04","2018-08-05","2018-08-06","2018-08-07","2018-08-08","2018-08-09","2018-08-10","2018-08-11","2018-08-12","2018-08-13","2018-08-14","2018-08-15","2018-08-16","2018-08-17","2018-08-18","2018-08-19","2018-08-20","2018-08-21","2018-08-22","2018-08-23","2018-08-24","2018-08-25","2018-08-26","2018-08-27","2018-08-28","2018-08-29","2018-08-30","2018-08-31","2018-09-01","2018-09-02","2018-09-03","2018-09-04","2018-09-05","2018-09-06","2018-09-07","2018-09-08","2018-09-09","2018-09-10","2018-09-11","2018-09-12","2018-09-13","2018-09-14","2018-09-15","2018-09-16","2018-09-17","2018-09-18","2018-09-19","2018-09-20","2018-09-21","2018-09-22","2018-09-23","2018-09-24","2018-09-25","2018-09-26","2018-09-27","2018-09-28","2018-09-29","2018-09-30","2018-10-01","2018-10-02","2018-10-03","2018-10-04","2018-10-05","2018-10-06","2018-10-07","2018-10-08","2018-10-09","2018-10-10","2018-10-11","2018-10-12","2018-10-13","2018-10-14","2018-10-15","2018-10-16","2018-10-17","2018-10-18","2018-10-19","2018-10-20","2018-10-21","2018-10-22","2018-10-23","2018-10-24","2018-10-25","2018-10-26","2018-10-27","2018-10-28","2018-10-29","2018-10-30","2018-10-31","2018-11-01","2018-11-02","2018-11-03","2018-11-04","2018-11-05","2018-11-06","2018-11-07","2018-11-08","2018-11-09","2018-11-10","2018-11-11","2018-11-12","2018-11-13","2018-11-14","2018-11-15","2018-11-16","2018-11-17","2018-11-18","2018-11-19","2018-11-20","2018-11-21","2018-11-22","2018-11-23","2018-11-24","2018-11-25","2018-11-26","2018-11-27","2018-11-28","2018-11-29","2018-11-30","2018-12-01","2018-12-02","2018-12-03","2018-12-04","2018-12-05","2018-12-06","2018-12-07","2018-12-08","2018-12-09","2018-12-10","2018-12-11","2018-12-12","2018-12-13","2018-12-14","2018-12-15","2018-12-16","2018-12-17","2018-12-18","2018-12-19","2018-12-20","2018-12-21","2018-12-22","2018-12-23","2018-12-24","2018-12-25","2018-12-26","2018-12-27","2018-12-28","2018-12-29","2018-12-30","2018-12-31","2019-01-01","2019-01-02","2019-01-03","2019-01-04","2019-01-05","2019-01-06","2019-01-07","2019-01-08","2019-01-09","2019-01-10","2019-01-11","2019-01-12","2019-01-13","2019-01-14","2019-01-15","2019-01-16","2019-01-17","2019-01-18","2019-01-19","2019-01-20","2019-01-21","2019-01-22","2019-01-23","2019-01-24","2019-01-25","2019-01-26","2019-01-27","2019-01-28","2019-01-29","2019-01-30","2019-01-31","2019-02-01","2019-02-02","2019-02-03","2019-02-04","2019-02-05","2019-02-06","2019-02-07","2019-02-08","2019-02-09","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2019-02-16","2019-02-17","2019-02-18","2019-02-19","2019-02-20","2019-02-21","2019-02-22","2019-02-23","2019-02-24","2019-02-25","2019-02-26","2019-02-27","2019-02-28","2019-03-01","2019-03-02","2019-03-03","2019-03-04","2019-03-05","2019-03-06","2019-03-07","2019-03-08","2019-03-09","2019-03-10","2019-03-11","2019-03-12","2019-03-13","2019-03-14","2019-03-15","2019-03-16","2019-03-17","2019-03-18","2019-03-19","2019-03-20","2019-03-21","2019-03-22","2019-03-23","2019-03-24","2019-03-25","2019-03-26","2019-03-27","2019-03-28","2019-03-29","2019-03-30","2019-03-31","2019-04-01","2019-04-02","2019-04-03","2019-04-04","2019-04-05","2019-04-06","2019-04-07","2019-04-08","2019-04-09","2019-04-10","2019-04-11","2019-04-12","2019-04-13","2019-04-14","2019-04-15","2019-04-16","2019-04-17","2019-04-18","2019-04-19","2019-04-20","2019-04-21","2019-04-22","2019-04-23","2019-04-24","2019-04-25","2019-04-26","2019-04-27","2019-04-28","2019-04-29","2019-04-30","2019-05-01","2019-05-02","2019-05-03","2019-05-04","2019-05-05","2019-05-06","2019-05-07","2019-05-08","2019-05-09","2019-05-10","2019-05-11","2019-05-12","2019-05-13","2019-05-14","2019-05-15","2019-05-16","2019-05-17","2019-05-18","2019-05-19","2019-05-20","2019-05-21","2019-05-22","2019-05-23","2019-05-24","2019-05-25","2019-05-26","2019-05-27","2019-05-28","2019-05-29","2019-05-30","2019-05-31","2019-06-01","2019-06-02","2019-06-03","2019-06-04","2019-06-05","2019-06-06","2019-06-07","2019-06-08","2019-06-09","2019-06-10","2019-06-11","2019-06-12","2019-06-13","2019-06-14","2019-06-15","2019-06-16","2019-06-17","2019-06-18","2019-06-19","2019-06-20","2019-06-21","2019-06-22","2019-06-23","2019-06-24","2019-06-25","2019-06-26","2019-06-27","2019-06-28","2019-06-29","2019-06-30","2019-07-01","2019-07-02","2019-07-03","2019-07-04","2019-07-05","2019-07-06","2019-07-07","2019-07-08","2019-07-09","2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15","2019-07-16","2019-07-17","2019-07-18","2019-07-19","2019-07-20","2019-07-21","2019-07-22","2019-07-23","2019-07-24","2019-07-25","2019-07-26","2019-07-27","2019-07-28","2019-07-29","2019-07-30","2019-07-31","2019-08-01","2019-08-02","2019-08-03","2019-08-04","2019-08-05","2019-08-06","2019-08-07","2019-08-08","2019-08-09","2019-08-10","2019-08-11","2019-08-12","2019-08-13","2019-08-14","2019-08-15","2019-08-16","2019-08-17","2019-08-18","2019-08-19","2019-08-20","2019-08-21","2019-08-22","2019-08-23","2019-08-24","2019-08-25","2019-08-26","2019-08-27","2019-08-28","2019-08-29","2019-08-30","2019-08-31","2019-09-01","2019-09-02","2019-09-03","2019-09-04","2019-09-05","2019-09-06","2019-09-07","2019-09-08","2019-09-09","2019-09-10","2019-09-11","2019-09-12","2019-09-13","2019-09-14","2019-09-15","2019-09-16","2019-09-17","2019-09-18","2019-09-19","2019-09-20","2019-09-21","2019-09-22","2019-09-23","2019-09-24","2019-09-25","2019-09-26","2019-09-27","2019-09-28","2019-09-29","2019-09-30","2019-10-01","2019-10-02","2019-10-03","2019-10-04","2019-10-05","2019-10-06","2019-10-07","2019-10-08","2019-10-09","2019-10-10","2019-10-11","2019-10-12","2019-10-13","2019-10-14","2019-10-15","2019-10-16","2019-10-17","2019-10-18","2019-10-19","2019-10-20","2019-10-21","2019-10-22","2019-10-23","2019-10-24","2019-10-25","2019-10-26","2019-10-27","2019-10-28","2019-10-29","2019-10-30","2019-10-31","2019-11-01","2019-11-02","2019-11-03","2019-11-04","2019-11-05","2019-11-06","2019-11-07","2019-11-08","2019-11-09","2019-11-10","2019-11-11","2019-11-12","2019-11-13","2019-11-14","2019-11-15","2019-11-16","2019-11-17","2019-11-18","2019-11-19","2019-11-20","2019-11-21","2019-11-22","2019-11-23","2019-11-24","2019-11-25","2019-11-26","2019-11-27","2019-11-28","2019-11-29","2019-11-30","2019-12-01","2019-12-02","2019-12-03","2019-12-04","2019-12-05","2019-12-06","2019-12-07","2019-12-08","2019-12-09","2019-12-10","2019-12-11","2019-12-12","2019-12-13","2019-12-14","2019-12-15","2019-12-16","2019-12-17","2019-12-18","2019-12-19","2019-12-20","2019-12-21","2019-12-22","2019-12-23","2019-12-24","2019-12-25","2019-12-26","2019-12-27","2019-12-28","2019-12-29","2019-12-30","2019-12-31","2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05","2020-01-06","2020-01-07","2020-01-08","2020-01-09","2020-01-10","2020-01-11","2020-01-12","2020-01-13","2020-01-14","2020-01-15","2020-01-16","2020-01-17","2020-01-18","2020-01-19","2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-25","2020-01-26","2020-01-27","2020-01-28","2020-01-29","2020-01-30","2020-01-31","2020-02-01","2020-02-02","2020-02-03","2020-02-04","2020-02-05","2020-02-06","2020-02-07","2020-02-08","2020-02-09","2020-02-10","2020-02-11","2020-02-12","2020-02-13","2020-02-14","2020-02-15","2020-02-16","2020-02-17","2020-02-18","2020-02-19","2020-02-20","2020-02-21","2020-02-22","2020-02-23","2020-02-24","2020-02-25","2020-02-26","2020-02-27","2020-02-28","2020-02-29","2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05","2020-03-06","2020-03-07","2020-03-08","2020-03-09","2020-03-10","2020-03-11","2020-03-12","2020-03-13","2020-03-14","2020-03-15","2020-03-16","2020-03-17","2020-03-18","2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01","2020-04-02","2020-04-03","2020-04-04","2020-04-05","2020-04-06","2020-04-07","2020-04-08","2020-04-09","2020-04-10","2020-04-11","2020-04-12","2020-04-13","2020-04-14","2020-04-15","2020-04-16","2020-04-17","2020-04-18","2020-04-19","2020-04-20","2020-04-21","2020-04-22","2020-04-23","2020-04-24","2020-04-25","2020-04-26","2020-04-27","2020-04-28","2020-04-29","2020-04-30","2020-05-01","2020-05-02","2020-05-03","2020-05-04","2020-05-05","2020-05-06","2020-05-07","2020-05-08","2020-05-09","2020-05-10","2020-05-11","2020-05-12","2020-05-13","2020-05-14","2020-05-15","2020-05-16","2020-05-17","2020-05-18","2020-05-19","2020-05-20","2020-05-21","2020-05-22","2020-05-23","2020-05-24","2020-05-25","2020-05-26","2020-05-27","2020-05-28","2020-05-29","2020-05-30","2020-05-31","2020-06-01","2020-06-02","2020-06-03","2020-06-04","2020-06-05","2020-06-06","2020-06-07","2020-06-08","2020-06-09","2020-06-10","2020-06-11","2020-06-12","2020-06-13","2020-06-14","2020-06-15","2020-06-16","2020-06-17","2020-06-18","2020-06-19","2020-06-20","2020-06-21","2020-06-22","2020-06-23","2020-06-24","2020-06-25","2020-06-26","2020-06-27","2020-06-28","2020-06-29","2020-06-30","2020-07-01","2020-07-02","2020-07-03","2020-07-04","2020-07-05","2020-07-06","2020-07-07","2020-07-08","2020-07-09","2020-07-10","2020-07-11","2020-07-12","2020-07-13","2020-07-14","2020-07-15","2020-07-16","2020-07-17","2020-07-18","2020-07-19","2020-07-20","2020-07-21","2020-07-22","2020-07-23","2020-07-24","2020-07-25","2020-07-26","2020-07-27","2020-07-28","2020-07-29","2020-07-30","2020-07-31","2020-08-01","2020-08-02","2020-08-03","2020-08-04","2020-08-05","2020-08-06","2020-08-07","2020-08-08","2020-08-09","2020-08-10","2020-08-11","2020-08-12","2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17","2020-08-18","2020-08-19","2020-08-20","2020-08-21","2020-08-22","2020-08-23","2020-08-24","2020-08-25","2020-08-26","2020-08-27","2020-08-28","2020-08-29","2020-08-30","2020-08-31","2020-09-01","2020-09-02","2020-09-03","2020-09-04","2020-09-05","2020-09-06","2020-09-07","2020-09-08","2020-09-09","2020-09-10","2020-09-11","2020-09-12","2020-09-13","2020-09-14","2020-09-15","2020-09-16","2020-09-17","2020-09-18","2020-09-19","2020-09-20","2020-09-21","2020-09-22","2020-09-23","2020-09-24","2020-09-25","2020-09-26","2020-09-27","2020-09-28","2020-09-29","2020-09-30","2020-10-01","2020-10-02","2020-10-03","2020-10-04","2020-10-05","2020-10-06","2020-10-07","2020-10-08","2020-10-09","2020-10-10","2020-10-11","2020-10-12","2020-10-13","2020-10-14","2020-10-15","2020-10-16","2020-10-17","2020-10-18","2020-10-19","2020-10-20","2020-10-21","2020-10-22","2020-10-23","2020-10-24","2020-10-25","2020-10-26","2020-10-27","2020-10-28","2020-10-29","2020-10-30","2020-10-31","2020-11-01","2020-11-02","2020-11-03","2020-11-04","2020-11-05","2020-11-06","2020-11-07","2020-11-08","2020-11-09","2020-11-10","2020-11-11","2020-11-12","2020-11-13","2020-11-14","2020-11-15","2020-11-16","2020-11-17","2020-11-18","2020-11-19","2020-11-20","2020-11-21","2020-11-22","2020-11-23","2020-11-24","2020-11-25","2020-11-26","2020-11-27","2020-11-28","2020-11-29","2020-11-30","2020-12-01","2020-12-02","2020-12-03","2020-12-04","2020-12-05","2020-12-06","2020-12-07","2020-12-08","2020-12-09","2020-12-10","2020-12-11","2020-12-12","2020-12-13","2020-12-14","2020-12-15","2020-12-16","2020-12-17","2020-12-18","2020-12-19","2020-12-20","2020-12-21","2020-12-22","2020-12-23","2020-12-24","2020-12-25","2020-12-26","2020-12-27","2020-12-28","2020-12-29","2020-12-30","2020-12-31","2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05","2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10","2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15","2021-01-16","2021-01-17","2021-01-18","2021-01-19","2021-01-20","2021-01-21","2021-01-22","2021-01-23","2021-01-24","2021-01-25","2021-01-26","2021-01-27","2021-01-28","2021-01-29","2021-01-30","2021-01-31","2021-02-01","2021-02-02","2021-02-03","2021-02-04","2021-02-05","2021-02-06","2021-02-07","2021-02-08","2021-02-09","2021-02-10","2021-02-11","2021-02-12","2021-02-13","2021-02-14","2021-02-15","2021-02-16","2021-02-17","2021-02-18","2021-02-19","2021-02-20","2021-02-21","2021-02-22","2021-02-23","2021-02-24","2021-02-25","2021-02-26","2021-02-27","2021-02-28","2021-03-01","2021-03-02","2021-03-03","2021-03-04","2021-03-05","2021-03-06","2021-03-07","2021-03-08","2021-03-09","2021-03-10","2021-03-11","2021-03-12","2021-03-13","2021-03-14","2021-03-15","2021-03-16","2021-03-17","2021-03-18","2021-03-19","2021-03-20","2021-03-21","2021-03-22","2021-03-23","2021-03-24","2021-03-25","2021-03-26","2021-03-27","2021-03-28","2021-03-29","2021-03-30","2021-03-31","2021-04-01","2021-04-02","2021-04-03","2021-04-04","2021-04-05","2021-04-06","2021-04-07","2021-04-08","2021-04-09","2021-04-10","2021-04-11","2021-04-12","2021-04-13","2021-04-14","2021-04-15","2021-04-16","2021-04-17","2021-04-18","2021-04-19","2021-04-20","2021-04-21","2021-04-22","2021-04-23","2021-04-24","2021-04-25","2021-04-26","2021-04-27","2021-04-28","2021-04-29","2021-04-30","2021-05-01","2021-05-02","2021-05-03","2021-05-04","2021-05-05","2021-05-06","2021-05-07","2021-05-08","2021-05-09","2021-05-10","2021-05-11","2021-05-12","2021-05-13","2021-05-14","2021-05-15","2021-05-16","2021-05-17","2021-05-18","2021-05-19","2021-05-20","2021-05-21","2021-05-22","2021-05-23","2021-05-24","2021-05-25","2021-05-26","2021-05-27","2021-05-28","2021-05-29","2021-05-30","2021-05-31","2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07","2021-06-08","2021-06-09","2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29","2021-06-30","2021-07-01","2021-07-02","2021-07-03","2021-07-04","2021-07-05","2021-07-06","2021-07-07","2021-07-08","2021-07-09","2021-07-10","2021-07-11","2021-07-12","2021-07-13","2021-07-14","2021-07-15","2021-07-16","2021-07-17","2021-07-18","2021-07-19","2021-07-20","2021-07-21","2021-07-22","2021-07-23","2021-07-24","2021-07-25","2021-07-26","2021-07-27","2021-07-28","2021-07-29","2021-07-30","2021-07-31","2021-08-01","2021-08-02","2021-08-03","2021-08-04","2021-08-05","2021-08-06","2021-08-07","2021-08-08","2021-08-09","2021-08-10","2021-08-11","2021-08-12","2021-08-13","2021-08-14","2021-08-15","2021-08-16","2021-08-17","2021-08-18","2021-08-19","2021-08-20","2021-08-21","2021-08-22","2021-08-23","2021-08-24","2021-08-25","2021-08-26","2021-08-27","2021-08-28","2021-08-29","2021-08-30","2021-08-31","2021-09-01","2021-09-02","2021-09-03","2021-09-04","2021-09-05","2021-09-06","2021-09-07","2021-09-08","2021-09-09","2021-09-10","2021-09-11","2021-09-12","2021-09-13","2021-09-14","2021-09-15","2021-09-16","2021-09-17","2021-09-18","2021-09-19","2021-09-20","2021-09-21","2021-09-22","2021-09-23","2021-09-24","2021-09-25","2021-09-26","2021-09-27","2021-09-28","2021-09-29","2021-09-30","2021-10-01","2021-10-02","2021-10-03","2021-10-04","2021-10-05","2021-10-06","2021-10-07","2021-10-08","2021-10-09","2021-10-10","2021-10-11","2021-10-12","2021-10-13","2021-10-14","2021-10-15","2021-10-16","2021-10-17","2021-10-18","2021-10-19","2021-10-20","2021-10-21","2021-10-22","2021-10-23","2021-10-24","2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29","2021-10-30","2021-10-31","2021-11-01","2021-11-02","2021-11-03","2021-11-04","2021-11-05","2021-11-06","2021-11-07","2021-11-08","2021-11-09","2021-11-10","2021-11-11","2021-11-12","2021-11-13","2021-11-14","2021-11-15","2021-11-16","2021-11-17","2021-11-18","2021-11-19","2021-11-20","2021-11-21","2021-11-22","2021-11-23","2021-11-24","2021-11-25","2021-11-26","2021-11-27","2021-11-28","2021-11-29","2021-11-30","2021-12-01","2021-12-02","2021-12-03","2021-12-04","2021-12-05","2021-12-06","2021-12-07","2021-12-08","2021-12-09","2021-12-10","2021-12-11","2021-12-12","2021-12-13","2021-12-14","2021-12-15","2021-12-16","2021-12-17","2021-12-18","2021-12-19","2021-12-20","2021-12-21","2021-12-22","2021-12-23","2021-12-24","2021-12-25","2021-12-26","2021-12-27","2021-12-28","2021-12-29","2021-12-30","2021-12-31","2022-01-01","2022-01-02","2022-01-03","2022-01-04","2022-01-05","2022-01-06","2022-01-07","2022-01-08","2022-01-09","2022-01-10","2022-01-11","2022-01-12","2022-01-13","2022-01-14","2022-01-15","2022-01-16","2022-01-17","2022-01-18","2022-01-19","2022-01-20","2022-01-21","2022-01-22","2022-01-23","2022-01-24","2022-01-25","2022-01-26","2022-01-27","2022-01-28","2022-01-29","2022-01-30","2022-01-31","2022-02-01","2022-02-02","2022-02-03","2022-02-04","2022-02-05","2022-02-06","2022-02-07","2022-02-08","2022-02-09","2022-02-10","2022-02-11","2022-02-12","2022-02-13","2022-02-14","2022-02-15","2022-02-16","2022-02-17","2022-02-18","2022-02-19","2022-02-20","2022-02-21","2022-02-22","2022-02-23","2022-02-24","2022-02-25","2022-02-26","2022-02-27","2022-02-28","2022-03-01","2022-03-02","2022-03-03","2022-03-04","2022-03-05","2022-03-06","2022-03-07","2022-03-08","2022-03-09","2022-03-10","2022-03-11","2022-03-12","2022-03-13","2022-03-14","2022-03-15","2022-03-16","2022-03-17","2022-03-18","2022-03-19","2022-03-20","2022-03-21","2022-03-22","2022-03-23","2022-03-24","2022-03-25","2022-03-26","2022-03-27","2022-03-28","2022-03-29","2022-03-30","2022-03-31","2022-04-01","2022-04-02","2022-04-03","2022-04-04","2022-04-05","2022-04-06","2022-04-07","2022-04-08","2022-04-09","2022-04-10","2022-04-11","2022-04-12","2022-04-13","2022-04-14","2022-04-15","2022-04-16","2022-04-17","2022-04-18","2022-04-19","2022-04-20","2022-04-21","2022-04-22","2022-04-23","2022-04-24","2022-04-25","2022-04-26","2022-04-27","2022-04-28","2022-04-29","2022-04-30","2022-05-01","2022-05-02","2022-05-03","2022-05-04","2022-05-05","2022-05-06","2022-05-07","2022-05-08","2022-05-09","2022-05-10","2022-05-11","2022-05-12","2022-05-13","2022-05-14","2022-05-15","2022-05-16","2022-05-17","2022-05-18","2022-05-19","2022-05-20","2022-05-21","2022-05-22","2022-05-23","2022-05-24","2022-05-25","2022-05-26","2022-05-27","2022-05-28","2022-05-29","2022-05-30","2022-05-31","2022-06-01","2022-06-02","2022-06-03","2022-06-04","2022-06-05","2022-06-06","2022-06-07","2022-06-08","2022-06-09","2022-06-10","2022-06-11","2022-06-12","2022-06-13","2022-06-14","2022-06-15","2022-06-16","2022-06-17","2022-06-18","2022-06-19","2022-06-20","2022-06-21","2022-06-22","2022-06-23","2022-06-24","2022-06-25","2022-06-26","2022-06-27","2022-06-28","2022-06-29","2022-06-30","2022-07-01","2022-07-02","2022-07-03","2022-07-04","2022-07-05","2022-07-06","2022-07-07","2022-07-08","2022-07-09","2022-07-10","2022-07-11","2022-07-12","2022-07-13","2022-07-14","2022-07-15","2022-07-16","2022-07-17","2022-07-18","2022-07-19","2022-07-20","2022-07-21","2022-07-22","2022-07-23","2022-07-24","2022-07-25","2022-07-26","2022-07-27","2022-07-28","2022-07-29","2022-07-30","2022-07-31","2022-08-01","2022-08-02","2022-08-03","2022-08-04","2022-08-05","2022-08-06","2022-08-07","2022-08-08","2022-08-09","2022-08-10","2022-08-11","2022-08-12","2022-08-13","2022-08-14","2022-08-15","2022-08-16","2022-08-17","2022-08-18","2022-08-19","2022-08-20","2022-08-21","2022-08-22","2022-08-23","2022-08-24","2022-08-25","2022-08-26","2022-08-27","2022-08-28","2022-08-29","2022-08-30","2022-08-31","2022-09-01","2022-09-02","2022-09-03","2022-09-04","2022-09-05","2022-09-06","2022-09-07","2022-09-08","2022-09-09","2022-09-10","2022-09-11","2022-09-12","2022-09-13","2022-09-14","2022-09-15","2022-09-16","2022-09-17","2022-09-18","2022-09-19","2022-09-20","2022-09-21","2022-09-22","2022-09-23","2022-09-24","2022-09-25","2022-09-26","2022-09-27","2022-09-28","2022-09-29","2022-09-30","2022-10-01","2022-10-02","2022-10-03","2022-10-04","2022-10-05","2022-10-06","2022-10-07","2022-10-08","2022-10-09","2022-10-10","2022-10-11","2022-10-12","2022-10-13","2022-10-14","2022-10-15","2022-10-16","2022-10-17","2022-10-18","2022-10-19","2022-10-20","2022-10-21","2022-10-22","2022-10-23","2022-10-24","2022-10-25","2022-10-26","2022-10-27","2022-10-28","2022-10-29","2022-10-30","2022-10-31","2022-11-01","2022-11-02","2022-11-03","2022-11-04","2022-11-05","2022-11-06","2022-11-07","2022-11-08","2022-11-09","2022-11-10","2022-11-11","2022-11-12","2022-11-13","2022-11-14","2022-11-15","2022-11-16","2022-11-17","2022-11-18","2022-11-19","2022-11-20","2022-11-21","2022-11-22","2022-11-23","2022-11-24","2022-11-25","2022-11-26","2022-11-27","2022-11-28","2022-11-29","2022-11-30","2022-12-01","2022-12-02","2022-12-03","2022-12-04","2022-12-05","2022-12-06","2022-12-07","2022-12-08","2022-12-09","2022-12-10","2022-12-11","2022-12-12","2022-12-13","2022-12-14","2022-12-15","2022-12-16","2022-12-17","2022-12-18","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-24","2022-12-25","2022-12-26","2022-12-27","2022-12-28","2022-12-29","2022-12-30","2022-12-31","2023-01-01","2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-07","2023-01-08","2023-01-09","2023-01-10","2023-01-11","2023-01-12","2023-01-13","2023-01-14","2023-01-15","2023-01-16","2023-01-17","2023-01-18","2023-01-19","2023-01-20","2023-01-21","2023-01-22","2023-01-23","2023-01-24","2023-01-25","2023-01-26","2023-01-27","2023-01-28","2023-01-29","2023-01-30","2023-01-31","2023-02-01","2023-02-02","2023-02-03","2023-02-04","2023-02-05","2023-02-06","2023-02-07","2023-02-08","2023-02-09","2023-02-10","2023-02-11","2023-02-12","2023-02-13","2023-02-14","2023-02-15","2023-02-16","2023-02-17","2023-02-18","2023-02-19","2023-02-20","2023-02-21","2023-02-22","2023-02-23","2023-02-24","2023-02-25","2023-02-26","2023-02-27","2023-02-28","2023-03-01","2023-03-02","2023-03-03","2023-03-04","2023-03-05","2023-03-06","2023-03-07","2023-03-08","2023-03-09","2023-03-10","2023-03-11","2023-03-12","2023-03-13","2023-03-14","2023-03-15","2023-03-16","2023-03-17","2023-03-18","2023-03-19","2023-03-20","2023-03-21","2023-03-22","2023-03-23","2023-03-24","2023-03-25","2023-03-26","2023-03-27","2023-03-28"],"y":[0.3339694656488549,0.3236574746008708,0.3208955223880597,0.32847533632287,0.3248463564530289,0.3345633456334563,0.3463855421686747,0.3151055766107201,0.3256802721088435,0.3498201438848921,0.3301060396496081,0.3395654370894391,0.2924675324675325,0.3174721189591078,0.3162393162393162,0.329430763019782,0.3380900109769484,0.3381180223285486,0.3483833381881736,0.3535072170169663,0.3419078242229367,0.355379188712522,0.3536391004072959,0.3617429082545361,0.3589634302276021,0.3833002739208463,0.3741697416974169,0.3750356633380885,0.3573735199138859,0.3751856352099365,0.3620427881297447,0.3415070946411352,0.3520632133450395,0.3603839441535777,0.3566246659030164,0.3756076005302696,0.3730569948186528,0.3417366946778712,0.3433526011560693,0.3437703848662753,0.3644251626898048,0.3463719929591238,0.3661305581835383,0.3664041994750656,0.3572450439403229,0.3572109275730623,0.3412988473432668,0.3799337149743899,0.3933410762679056,0.3556678383128295,0.3833601933924254,0.37488928255093,0.3514774494556765,0.3485388453314326,0.3644121365360303,0.3412098298676748,0.358656537385046,0.3713527851458886,0.3713071630918656,0.3621677957269411,0.3563300142247511,0.3437088996054362,0.3360120542440985,0.3560666137985726,0.3804430863254393,0.3709872804360993,0.3879546343371138,0.3848177376925968,0.3937667560321716,0.4014026402640264,0.3667937314697162,0.3864367816091954,0.4010717230008244,0.3782256082575571,0.38,0.3840487680975362,0.3708684381075826,0.3659029649595688,0.3527667984189723,0.3753315649867374,0.3864356251164524,0.3673592898930805,0.3705692803437165,0.3837041317538284,0.3745819397993311,0.3893004115226337,0.3787590407308717,0.3400460299194477,0.3536812674743709,0.3697183098591549,0.3757519666820916,0.3723756906077348,0.3771401012780323,0.3436010267693436,0.3404612159329141,0.3524687685901249,0.368560105680317,0.3752645902630784,0.3480160174736075,0.3685483870967742,0.3626495964375174,0.3499750374438342,0.3504728132387706,0.3372908796546141,0.3812997347480106,0.3511450381679389,0.3625639009044436,0.3636043646603309,0.3650355169692186,0.3598335523434078,0.3430804473790594,0.3643364928909952,0.3809523809523809,0.3729344233779788,0.3538322112894873,0.3580621661000485,0.3610630602295958,0.3662587412587413,0.3825055596738325,0.3768115942028985,0.3828358208955224,0.3776003362050851,0.373974540311174,0.3888691953208082,0.4105003918056644,0.4052646396396397,0.3951297547102737,0.4075260208166533,0.3915187376725838,0.3521945251567518,0.3765157639450283,0.3796006264682849,0.3682992672580023,0.3714285714285713,0.3718189124028931,0.3712121212121212,0.3863859925241,0.3821612349914237,0.3525076765609007,0.3666245259165613,0.3758570029382957,0.3422103861517976,0.3292079207920792,0.3536842105263157,0.3762632197414806,0.3596237337192474,0.3585365853658536,0.3245033112582781,0.3641025641025641,0.3771157752200406,0.3830968537939544,0.3584195672624647,0.3717703349282296,0.3290187891440501,0.318783542039356,0.3988095238095238,0.3923745173745174,0.3815971262779773,0.3590944574551132,0.3706617306802406,0.3334273624823695,0.3599557522123894,0.3649390972056365,0.3431307793923381,0.3624760689215061,0.3819792067770504,0.3558542624690484,0.3838105726872247,0.3485922237119326,0.358202313853456,0.3716006884681584,0.3374627208075246,0.3669543773119605,0.3436499466382069,0.3658410732714138,0.3383038210624418,0.3640860215053763,0.3570347957639939,0.3501981813942644,0.3508557457212714,0.35,0.3571777741567541,0.3395061728395062,0.3436354112100576,0.3747429041546689,0.3493490806603141,0.3799937578027466,0.3709401709401709,0.3848337388483374,0.3851162790697674,0.3882884336183256,0.3887394656625426,0.3808104079143516,0.3757358076424224,0.3547321217842813,0.3534978723404255,0.3747785399536637,0.3705562526139691,0.3688461958301496,0.3819465532948679,0.3789134808853119,0.3656387665198238,0.3860047711007611,0.3767031494304221,0.3810359154279383,0.384355482748029,0.3860167798641629,0.3591781132932229,0.3901230970937432,0.3985643895614088,0.3938927243759957,0.3976440166642724,0.391721259509721,0.4078525417584562,0.4089634847613571,0.4118271954674221,0.4358919762944036,0.4260049161364951,0.4213809109370467,0.3895396757998851,0.3972104305639781,0.3927080434177567,0.4011830416950061,0.3977110631945596,0.3966937811598006,0.4362078173079368,0.4207024957636869,0.421294964028777,0.4156157228633289,0.421505905511811,0.3987378083763626,0.4052847863549997,0.405737368429695,0.4102657444484892,0.3955970263641937,0.4118582640210674,0.4141181970470217,0.3926819388811378,0.4122738474484582,0.4043926462940871,0.4012416328512468,0.3799153569681632,0.3909800722526512,0.4010615832429656,0.4104211100639849,0.4180354796320631,0.3960017014036581,0.3878813315955731,0.376699582543967,0.3907863383637808,0.3812199187220256,0.3802206847892127,0.3858803954572717,0.3943520482196673,0.3941689921474463,0.3755398001933613,0.3737831980310767,0.3808521729057667,0.4046507949062722,0.3829043673209223,0.388307871082402,0.3791520345371952,0.3831996224634261,0.3819993245525161,0.3784935455570873,0.3877505814597408,0.3765752707653903,0.3894052044609665,0.3834078987966677,0.3865566037735849,0.3939519900239011,0.3891514279843846,0.3915605994106214,0.3824718452209067,0.3833619756858252,0.3777159742742917,0.3955997461392004,0.3980446927374301,0.4014198782961461,0.4081665460975806,0.3879911977365608,0.3945876073721958,0.3878182585455057,0.3756340028403327,0.3798041389504804,0.4130434782608696,0.3997888270301401,0.4036111111111111,0.3930503196197345,0.3966450216450216,0.4216771249286937,0.4082856744315804,0.4141239173884077,0.402226157348849,0.4182773821634452,0.4110742888704777,0.4118357487922705,0.4166666666666667,0.4005535784895216,0.4137596899224806,0.4024835646457268,0.4130871869859258,0.4088771310993533,0.371875940981632,0.4196891191709845,0.4271554142509818,0.4166429991479693,0.4218455743879473,0.4061275382971144,0.4021060842433697,0.3975513143680231,0.4071719226856561,0.3938489786550379,0.4140744518103009,0.3868458274398868,0.3950191570881226,0.3841336116910229,0.4136516225289072,0.4256834367103147,0.4026607538802661,0.3822805578342904,0.3931675763088741,0.3940941385435169,0.3975576662143826,0.4032641958517511,0.3929193267556587,0.4002064693737096,0.4698240420983391,0.4768518518518519,0.4394904458598726,0.4454689146469968,0.418943533697632,0.3919446949203487,0.3920909795812872,0.4075691411935953,0.4168214380472274,0.4263322884012539,0.4128978224455611,0.4219281663516068,0.4231846584787761,0.4084254556319092,0.420827389443652,0.4465157567687528,0.4168946648426813,0.3959369817578773,0.41396176078257,0.4262870514820593,0.446459918080749,0.4252258280361325,0.4214330609679447,0.4199276553765209,0.3937965260545906,0.4247663551401869,0.4213164049448714,0.4305287533770745,0.4488436632747456,0.4549715433545363,0.4305904522613065,0.4331473444926638,0.4409090909090909,0.4186359269932757,0.3918789043728976,0.4140604807443784,0.4207137601177337,0.4226876912214639,0.4399503722084368,0.395680521597392,0.4330042313117066,0.4154154154154153,0.4130794701986754,0.4126811594202899,0.4170946441672781,0.4141457179377929,0.4162877221992559,0.404971932638332,0.4248419335100959,0.4125475285171102,0.4116630669546436,0.3910743801652893,0.4143810229799852,0.4159144098963557,0.4046822742474917,0.4296246648793566,0.3796192609182531,0.4116059379217274,0.3971210838272649,0.4142427281845537,0.4051466187911431,0.4056408129406885,0.3979303857008466,0.3946308724832215,0.4641460234680574,0.3870573870573871,0.4084600132187707,0.4276094276094276,0.4332953249714937,0.4037184594953519,0.4176182707993475,0.4136490250696379,0.3466666666666667,0.3830303030303031,0.375,0.3911637931034483,0.4057450628366248,0.3628158844765343,0.3811188811188811,0.3747152619589977,0.3867102396514161,0.4423199490121096,0.4319793681495809,0.4285714285714285,0.3598435462842243,0.3631578947368421,0.3457943925233645,0.4281481481481481,0.3846153846153846,0.3645484949832776,0.4109742441209406,0.4165853658536585,0.4266347687400319,0.4059609455292909,0.4140480591497228,0.4144645340751043,0.4235074626865672,0.3986486486486486,0.4403508771929824,0.4131578947368421,0.4213483146067416,0.4281274281274281,0.4044616876818623,0.4185336048879837,0.4180451127819549,0.4109149277688604,0.4012875536480687,0.3902053712480253,0.41051567239636,0.4226579520697168,0.3996569468267582,0.467032967032967,0.4064386317907445,0.4282560706401766,0.4318181818181818,0.416588124410933,0.4172813487881981,0.4341546304163126,0.4437086092715232,0.4053398058252427,0.3916349809885932,0.46033350176857,0.4304666056724611,0.421636615811373,0.4189723320158103,0.4259776536312849,0.3774403470715835,0.4090247452692867,0.4025157232704403,0.4391201885310291,0.4275680421422301,0.3788587464920487,0.4011208967173739,0.3654080389768575,0.478405315614618,0.3522427440633245,0.3509234828496042,0.375249500998004,0.3808353808353808,0.3610149942329873,0.3975308641975309,0.3414634146341464,0.369620253164557,0.3832004904966279,0.3834394904458599,0.3811659192825112,0.4199683042789224,0.4287790697674417,0.410958904109589,0.3811074918566775,0.4099722991689751,0.4060773480662984,0.3924466338259442,0.381404174573055,0.3863636363636364,0.3566739606126915,0.3248407643312102,0.3616600790513834,0.3849129593810445,0.3683083511777302,0.3915343915343915,0.3632958801498127,0.3806646525679758,0.3672922252010724,0.3493670886075949,0.4295190713101161,0.3953974895397489,0.3714285714285713,0.3512880562060889,0.3805668016194332,0.3522458628841608,0.3660714285714285,0.3951332560834299,0.4059040590405904,0.4347826086956521,0.3738019169329074,0.327319587628866,0.3996282527881041,0.3807692307692308,0.3679012345679013,0.3367768595041322,0.3973333333333333,0.4097719869706841,0.412291933418694,0.410941475826972,0.4142857142857143,0.4034653465346535,0.4254143646408839,0.4197901049475262,0.3936507936507936,0.3618513323983169,0.4144486692015209,0.4708904109589041,0.4111922141119221,0.4145077720207253,0.4282352941176471,0.4190231362467866,0.5080906148867314,0.4491725768321513,0.4923076923076923,0.4285714285714285,0.4771048744460857,0.4983277591973244,0.3953488372093023,0.4844192634560906,0.4251748251748252,0.4082191780821917,0.4503105590062112,0.4375,0.4202898550724637,0.4631578947368421,0.4778481012658228,0.4186046511627907,0.4151515151515152,0.43343653250774,0.4366576819407008,0.4841772151898734,0.417004048582996,0.4090909090909091,0.4155495978552279,0.441077441077441,0.3883928571428572,0.4285714285714285,0.4377104377104377,0.5048076923076923,0.4186046511627907,0.4668304668304668,0.4215851602023609,0.462,0.4429223744292237,0.4229828850855746,0.3781094527363184,0.403921568627451,0.4805194805194805,0.4954128440366973,0.4119402985074626,0.4737430167597765,0.4207920792079208,0.4726027397260274,0.3676470588235294,0.4630225080385852,0.4723225030084236,0.4447031431897555,0.4666666666666667,0.4619450317124736,0.4541062801932367,0.4934725848563969,0.4457611668185962,0.4439083232810615,0.4955185659411012,0.4978292329956585,0.4842696629213483,0.4923954372623574,0.4510250569476082,0.4605678233438486,0.4904632152588556,0.4830508474576271,0.4717241379310345,0.4757990867579909,0.5102739726027398,0.5050505050505051,0.4969325153374233,0.4759358288770054,0.467948717948718,0.4860759493670886,0.4332061068702289,0.5106888361045131,0.5301204819277109,0.5329087048832272,0.4535928143712575,0.5033621517771374,0.4776264591439689,0.4533258803801006,0.5041916167664671,0.497504159733777,0.4675456389452333,0.4766839378238342,0.4789687924016282,0.4547101449275362,0.5053956834532374,0.4636871508379888,0.4929245283018868,0.5245901639344263,0.4024096385542169,0.4589800443458979,0.4015957446808511,0.4065281899109792,0.4405940594059406,0.5,0.4188034188034188,0.3725490196078432,0.4064516129032258,0.4595419847328244,0.4070981210855949,0.5,0.459016393442623,0.4427860696517413,0.3594936708860759,0.3660477453580902,0.3489010989010989,0.3553921568627451,0.5021834061135371,0.4791666666666667,0.3986175115207373,0.3769230769230769,0.3834808259587021,0.3726027397260274,0.4117647058823529,0.5409836065573771,0.4978165938864629,0.4612403100775194,0.3645621181262729,0.3719806763285024,0.3956310679611649,0.3851351351351352,0.4196891191709845,0.4044444444444445,0.4218009478672986,0.5106837606837606,0.5157894736842106,0.4857142857142857,0.4674157303370786,0.5141843971631206,0.51931330472103,0.3618784530386741,0.3810975609756097,0.4098360655737705,0.3535714285714286,0.4238683127572017,0.5925925925925926,0.4615384615384616,0.3908629441624365,0.3788235294117647,0.3358208955223881,0.4132420091324201,0.3052631578947368,0.4961240310077519,0.4913793103448276,0.3651226158038147,0.4032786885245901,0.3910034602076125,0.3839009287925697,0.5030303030303029,0.3813559322033898,0.4230769230769231,0.3992932862190813,0.4204545454545455,0.4279475982532751,0.4463276836158192,0.4324324324324325,0.5012853470437018,0.5034965034965035,0.4585492227979275,0.4075949367088608,0.4086378737541528,0.5126903553299492,0.3594470046082949,0.468421052631579,0.4154589371980677,0.4387755102040816,0.456896551724138,0.5165289256198347,0.4513618677042802,0.5714285714285714,0.5202702702702703,0.4953703703703704,0.5462962962962963,0.492822966507177,0.4619289340101523,0.4626168224299066,0.384,0.4956140350877193,0.5525423728813559,0.4626436781609196,0.4163783160322953,0.4449197860962567,0.5315985130111525,0.4858569051580699,0.4894366197183099,0.4935400516795866,0.4713375796178344,0.4402277039848197,0.3986013986013986,0.471947194719472,0.489051094890511,0.5728155339805825,0.427807486631016,0.4752475247524752,0.454828660436137,0.4666666666666667,0.4809384164222874,0.5057034220532319,0.5597826086956522,0.5578947368421052,0.4147058823529412,0.4820295983086681,0.4074074074074074,0.4083333333333333,0.4016172506738545,0.4076923076923077,0.4162436548223351,0.4006211180124224,0.4271356783919598,0.48,0.4254545454545455,0.4568690095846645,0.4247787610619469,0.3312101910828025,0.4393939393939394,0.4549019607843137,0.4396887159533074,0.4076433121019109,0.4313725490196079,0.4970238095238095,0.4890965732087227,0.4164705882352941,0.4964819700967458,0.4623833058758924,0.4815503875968992,0.490006447453256,0.4876712328767123,0.4856972586412396,0.4729520865533231,0.4637931034482759,0.4621276595744681,0.5143038779402416,0.4735785953177257,0.5199692780337941,0.5153940886699507,0.4922680412371134,0.4739988045427376,0.4685963954123429,0.451867816091954,0.5020096463022508,0.4770773638968481,0.4709058188362327,0.4805755395683453,0.4946601941747573,0.476409666283084,0.4995685936151855,0.4974704890387858,0.4713513513513514,0.4946666666666666,0.4808795411089866,0.4803548795944233,0.4586583463338533,0.4846547314578005,0.4610389610389609,0.4680851063829787,0.4930643127364439,0.481890730509515,0.4413223140495868,0.4383419689119171,0.4585987261146497,0.4971428571428572,0.4780902550686723,0.484818805093046,0.4882186616399623,0.450473729543497,0.4590352220520674,0.4692643600940544,0.471559633027523,0.5004149377593361,0.511482254697286,0.5025083612040134,0.4842436974789916,0.5246350364963503,0.5136612021857924,0.4974025974025974,0.4929577464788733,0.4663951120162933,0.481658692185008,0.4965706447187929,0.4656144306651634,0.5119549929676512,0.4956268221574344,0.4988864142538976,0.5007587253414264,0.5045454545454545,0.4451294697903823,0.4800270819228165,0.4897172236503856,0.4476744186046512,0.4404145077720208,0.4855595667870036,0.4439461883408072,0.4516129032258064,0.4298375184638109,0.4310344827586207,0.3562340966921121,0.4716981132075472,0.4849498327759197,0.4566210045662099,0.4231578947368421,0.3915211970074813,0.3713646532438479,0.3925925925925926,0.467479674796748,0.3732718894009217,0.4357262103505843,0.4279661016949153,0.3946784922394678,0.4601063829787234,0.4356060606060606,0.5520361990950227,0.3781512605042017,0.3975155279503105,0.4225352112676056,0.5461254612546126,0.4507389162561576,0.4067796610169492,0.4324324324324325,0.3560606060606061,0.4301470588235294,0.3497536945812808,0.4006622516556291,0.4090909090909091,0.3734177215189873,0.4148148148148148,0.3904761904761905,0.350104821802935,0.3462783171521035,0.300395256916996,0.362962962962963,0.4933920704845815,0.4516129032258064,0.2030075187969924,0.4864864864864865,0.3640552995391705,0.4672897196261682,0.2419354838709677,0.3263888888888889,0.4556962025316456,0.425531914893617,0.4487179487179487,0.408695652173913,0.4907407407407408,0.3269230769230769,0.2195121951219512,0.3673469387755102,0.4125,0.3904109589041096,0.3956043956043956,0.4171428571428572,0.3222748815165877,0.419753086419753,0.4861111111111111,0.348314606741573,0.2994652406417112,0.2748538011695906,0.4089887640449438,0.4081632653061225,0.3547008547008547,0.5277777777777778,0.5068493150684932,0.3412322274881517,0.3514851485148514,0.390625,0.5540540540540541,0.3389830508474576,0.5242718446601942,0.4678899082568808,0.4382716049382716,0.2838709677419355,0.4296875,0.2845528455284553,0.2916666666666667,0.4360902255639097,0.3969465648854962,0.3669064748201439,0.4375,0.465,0.3955223880597015,0.3053435114503817,0.4432989690721649,0.532608695652174,0.4216867469879518,0.4179104477611939,0.4438775510204081,0.4298245614035088,0.4764705882352941,0.5037878787878788,0.3765432098765432,0.4299065420560748,0.4522613065326633,0.4280575539568345,0.3950617283950617,0.4207650273224044,0.3691275167785235,0.4044943820224719,0.3642857142857143,0.3611111111111111,0.3431372549019608,0.4462809917355372,0.4015151515151515,0.4848484848484849,0.4,0.3043478260869565,0.3363636363636364,0.3628318584070797,0.31875,0.3414634146341464,0.44,0.5243902439024389,0.3546099290780142,0.3723404255319149,0.2933333333333333,0.2967032967032967,0.3486842105263157,0.44,0.3522727272727273,0.4457831325301205,0.3392857142857143,0.4196891191709845,0.3932038834951456,0.4308510638297872,0.3176470588235294,0.4380952380952381,0.3,0.3155339805825243,0.3615384615384615,0.3202247191011236,0.2822085889570552,0.4035087719298245,0.3292682926829268,0.3333333333333333,0.335483870967742,0.3595505617977528,0.323943661971831,0.2644628099173554,0.2985074626865671,0.3888888888888889,0.29,0.3085106382978723,0.3588516746411483,0.2808988764044944,0.3283582089552239,0.3409090909090909,0.3877551020408163,0.3725490196078432,0.3142857142857143,0.2666666666666666,0.2184873949579832,0.3688524590163935,0.4545454545454545,0.2857142857142857,0.3076923076923077,0.3076923076923077,0.3888888888888889,0.2553191489361702,0.3012048192771085,0.3690476190476191,0.2962962962962963,0.4766355140186916,0.4916666666666666,0.4216867469879518,0.4227642276422765,0.4496124031007752,0.6176470588235294,0.5576923076923077,0.3490566037735849,0.5116279069767442,0.4957264957264957,0.4583333333333333,0.429245283018868,0.5111111111111111,0.4556213017751479,0.3154761904761905,0.3941605839416058,0.391304347826087,0.3925925925925926,0.3488372093023256,0.2857142857142857,0.4117647058823529,0.4230769230769231,0.3688524590163935,0.4347826086956521,0.48,0.4603174603174603,0.5483870967741935,0.5217391304347826,0.3278688524590164,0.4810126582278481,0.3717948717948718,0.3737373737373737,0.3924050632911392,0.3846153846153846,0.5192307692307693,0.3684210526315789,0.3666666666666665,0.4685714285714286,0.4364089775561097,0.4328358208955224,0.544973544973545,0.5510204081632653,0.5030303030303029,0.5128205128205128,0.4186046511627907,0.4113475177304965,0.5371900826446281,0.4615384615384616,0.4883720930232558,0.4464285714285714,0.3125,0.4054054054054054,0.3333333333333333,0.4130434782608696,0.3373493975903614,0.4777777777777778,0.4645669291338583,0.5357142857142857,0.4680851063829787,0.4942528735632184,0.4266666666666667,0.4880952380952381,0.6065573770491803,0.5141242937853108,0.4553571428571429,0.4433962264150944,0.4715189873417722,0.4860088365243005,0.4510869565217392,0.3605442176870749,0.5054545454545455,0.4161849710982659,0.4934210526315789,0.4590909090909091,0.4625550660792952,0.4405594405594406,0.4299065420560748,0.4692737430167598,0.4018691588785047,0.3664596273291925,0.4680851063829787,0.4285714285714285,0.3482142857142857,0.4077669902912621,0.4016393442622951,0.475609756097561,0.4239130434782609,0.4533333333333333,0.4527027027027027,0.4328358208955224,0.4266666666666667,0.3055555555555556,0.4700854700854701,0.4020618556701031,0.3968253968253968,0.4720496894409938,0.4175824175824176,0.5,0.4214285714285714,0.5111111111111111,0.4155844155844156,0.4506172839506173,0.4,0.46,0.4020618556701031,0.5176470588235295,0.3787878787878788,0.3505154639175257,0.4152542372881356,0.4214876033057851,0.3936170212765957,0.3194444444444444,0.5051546391752577,0.4396551724137931,0.436241610738255,0.4776785714285714,0.412280701754386,0.6039603960396039,0.5238095238095238,0.4489795918367347,0.4726027397260274,0.5033112582781457,0.465,0.5281385281385281,0.5,0.5381165919282511,0.4925925925925926,0.4137931034482759,0.4335664335664336,0.4642857142857143,0.3851351351351352,0.5098039215686274,0.5277777777777778,0.4385026737967914,0.4631578947368421,0.4615384615384616,0.4697508896797153,0.5308641975308642,0.5373134328358209,0.515625,0.4858156028368794,0.46875,0.4615384615384616,0.4642857142857143,0.5384615384615384,0.5247524752475248,0.4158415841584158,0.4748858447488584,0.5342465753424658,0.5035714285714286,0.5359712230215827,0.4761904761904762,0.5072886297376094,0.5075187969924813,0.5311778290993071,0.5172413793103449,0.4734693877551021,0.4592274678111588,0.4576271186440678,0.5153846153846153,0.425,0.4403292181069959,0.4290780141843972,0.4584837545126354,0.483271375464684,0.4861878453038674,0.4387755102040816,0.4285714285714285,0.3965517241379311,0.4621212121212121,0.4888888888888889,0.4895104895104895,0.3975903614457832,0.4810126582278481,0.4675324675324675,0.4810126582278481,0.4736842105263158,0.5193798449612403,0.484375,0.4214285714285714,0.4491525423728814,0.3805309734513274,0.5252525252525253,0.3969465648854962,0.4454545454545454,0.4864864864864865,0.5233644859813084,0.2941176470588235,0.4107142857142857,0.5111111111111111,0.4166666666666667,0.3684210526315789,0.4625,0.5783132530120482,0.5507246376811594,0.4285714285714285,0.5212765957446809,0.4322033898305085,0.4375,0.5086206896551724,0.3308823529411765,0.5,0.5744680851063829,0.5652173913043478,0.4694835680751174,0.5157894736842106,0.4891304347826087,0.4625850340136055,0.4727272727272727,0.4836272040302267,0.4782608695652174,0.5065502183406113,0.5390334572490706,0.5054347826086957,0.4424242424242424,0.4822695035460993,0.3535353535353535,0.4497041420118344,0.4285714285714285,0.5747126436781609,0.4155844155844156,0.4759825327510917,0.4201680672268908,0.4742268041237113,0.4666666666666667,0.4822695035460993,0.4785714285714286,0.3442622950819672,0.4491978609625669,0.4230769230769231,0.5288461538461539,0.3125,0.4367088607594937,0.4652777777777778,0.5043478260869565,0.5,0.4318181818181818,0.5411764705882353,0.3925233644859813,0.481203007518797,0.5,0.3962264150943397,0.4723618090452261,0.4814814814814815,0.3868613138686132,0.3962264150943397,0.4029126213592233,0.3298969072164948,0.4840182648401826,0.4285714285714285,0.4444444444444444,0.4776119402985074,0.4339622641509434,0.4732142857142857,0.4247787610619469,0.3846153846153846,0.3956834532374101,0.4538461538461538,0.4725274725274725,0.4636363636363636,0.4453781512605042,0.4609375,0.4573643410852713,0.5098039215686274,0.4782608695652174,0.4086021505376344,0.4519230769230769,0.4311926605504587,0.5074626865671642,0.4594594594594595,0.4464285714285714,0.4310344827586207,0.5227272727272727,0.5454545454545454,0.4285714285714285,0.4,0.3578947368421053,0.4174757281553398,0.3846153846153846,0.38,0.47,0.4382022471910113,0.4909090909090909,0.4432989690721649,0.4066666666666667,0.4133333333333333,0.4375,0.373134328358209,0.3867403314917127,0.4319526627218935,0.4285714285714285,0.5033557046979866,0.4095238095238095,0.3809523809523809,0.45,0.4350282485875706,0.44921875,0.4046511627906977,0.4344827586206896,0.3783783783783784,0.4651162790697674,0.3020134228187919,0.3385826771653543,0.3973509933774834,0.3481481481481481,0.4602272727272727,0.4556962025316456,0.3775510204081632,0.3381294964028777,0.3371428571428571,0.3696969696969697,0.392638036809816,0.328125,0.4591836734693878,0.4485981308411215,0.3475609756097561,0.35,0.3870967741935484,0.4163822525597269,0.3686006825938566,0.4114977307110439,0.5082417582417582,0.485484334578902,0.4680881107031912,0.4919298245614035,0.4856579147063288,0.5017835909631391,0.4993365767359576,0.510548523206751,0.5194174757281553,0.5111408199643493,0.4991067440821795,0.4962173314993122,0.5098231827111984,0.5106382978723404,0.4788844621513944,0.4720314033366045,0.5007841087297439,0.5176886792452831,0.5083281924737816,0.488517745302714,0.5044642857142857,0.4866573033707865,0.4616805170821791,0.4653979238754326,0.4827586206896552,0.4858286095365122,0.5130288596245447,0.5227329192546584,0.4972178060413355,0.5133625847626645,0.5178571428571429,0.5106584557081951,0.5233542747883092,0.5169653069004956,0.4790774299835256,0.4884949075820445,0.506896551724138,0.5094299639754185,0.5122931442080378,0.4770469798657718,0.484647112740605,0.5100112485939258,0.4779104173043567,0.4911429264741568,0.5047587791270102,0.4859287054409006,0.4859989433164767,0.4879549620319456,0.4964578085642317,0.4978236121341994,0.4945531799484699,0.4886456704638522,0.4826379542395693,0.5279083431257344,0.4977322944528434,0.5067458175930922,0.5224445646295295,0.4905842588121681,0.4882322844717319,0.4793608521970706,0.4977711738484398,0.4927564797477087,0.5030662710187933,0.5099188458070334,0.5080710250201775,0.4905247813411079,0.5087327376116978,0.4984174579376978,0.4744997743342861,0.4954650917527948,0.4951360115294541,0.4669840811811667,0.500499900019996,0.525406241939644,0.5057091025504216,0.5147002672775869,0.5244055068836045,0.5164156998272031,0.5007647958583362,0.4987857762359063,0.5029733689563044,0.5029062201787109,0.5075892857142857,0.4911133621880468,0.5043555952646862,0.507940654059137,0.4824237153004276,0.5094011250271934,0.5245227254573049,0.5459655016473932,0.5187554072426153,0.5052640540310451,0.4908930529728699,0.5053961062552091,0.4986754760494905,0.505705623871649,0.4729916555572014,0.5126268476900978,0.5168421898866469,0.5142802284836557,0.5084235189110788,0.5168205904617714,0.5087217718587146,0.4947443982976978,0.5236385921905096,0.5192943770672547,0.4999423697556478,0.5006726371218818,0.5084114932261427,0.5080316705223029,0.4939693149020203,0.4954119436675009,0.4968646493230858,0.5025242993299991,0.4930758828463343,0.518629570229877,0.490653296738919,0.4922774428892971,0.5019920318725101,0.5176500379842999,0.5004262574595055,0.4929901971439129,0.4902665776362422,0.4958660549983153,0.5059119299210433,0.5145963897369972,0.5176248241912799,0.4976359338061466,0.5065219011652463,0.4930364248546067,0.4915815013601941,0.4978303299035491,0.5130985687530347,0.5170762678236025,0.4961151464634437,0.5055454956994115,0.5019974252458624,0.5001245894212255,0.4992343032159265,0.5144732216102191,0.5184672206832872,0.5162405426558832,0.5153293647466533,0.5137342402842164,0.5089850958126331,0.4959426252928738,0.5048007246376811,0.5123868862186619,0.5046957660560735,0.5164308358462847,0.5105955833147446,0.5180517030955287,0.5106162456691048,0.5160351739299107,0.5108300395256917,0.5188322919975996,0.4999123514356834,0.498920792726797,0.4746397934538186,0.5104725012386829,0.5104058556836183,0.5083719273245457,0.4939121455922213,0.4918289625915928,0.4932336415906954,0.4927432979827793,0.4876391523047745,0.5134380892520427,0.5096524321720982,0.5096448925909689,0.5102045328911,0.4997763308315522,0.5099919751032471,0.5035987819507244,0.51180618333032,0.5318313775197285,0.5110299283405929,0.5236907730673317,0.5059054056483362,0.514976109215017,0.5246258412294506,0.5298530709724313,0.5317681193747039,0.5396562206378157,0.5241654403222059,0.5198633062793678,0.5200151261412134,0.5161674008810573,0.5327790033372928,0.5363759847867428,0.5382617824189182,0.5392256233529292,0.5138004246284501,0.514216575922565,0.527160227354486,0.5281333456119348,0.5210234924435134,0.5188392445877476,0.5408158339367608,0.5337904015670911,0.5369916354196712,0.5194414349343927,0.5439735347356595,0.545587804163002,0.5536039868724929,0.5371636052968817,0.5376831821353804,0.5266427969671441,0.5260693381359748,0.5626810095159288,0.5403733693207378,0.5287223168654174,0.5414050235478807,0.5442616715793477,0.5280600281381898,0.539821866528979,0.5556495979686839,0.5498112588215985,0.5312008390141584,0.5332285793511415,0.5301091061686949,0.5194984592498141,0.5344084526623106,0.5461328741569809,0.5427752762996316,0.5281166736783061,0.5381471389645777,0.5417416336811803,0.5528368252026303,0.5490743974851554,0.5521501544309813,0.5732551944592434,0.5351983490025224,0.5467071935157042,0.5722797115083098,0.556055765320941,0.5636960868323336,0.5553531673750253,0.5486577181208053,0.5538916256157635,0.5625879043600562,0.5713296398891967,0.5577062228654125,0.5805986296429859,0.5925756710451171,0.5880361173814899,0.5460581787521079,0.5696395513448764,0.5487883892704419,0.5401518178186177,0.5015150447466704,0.5336459810230123,0.5387354030190828,0.5407023144453312,0.5047543581616482,0.5277180919639021,0.5004824984647777,0.5196446873932354,0.5373148148148148,0.5132920037827047,0.5034521895393927,0.5206313802865231,0.5656208277703605,0.5661053589714415,0.5352700079096858,0.5359872611464969,0.5170862599636966,0.5307300509337861,0.5163221326522468,0.502261590652092,0.5184451219512195,0.5165017413897949,0.5344067048963388,0.5259484961666995,0.5423460503403514,0.5334406808902179,0.5282849781561555,0.5254788846978774,0.5208882498602124,0.5342641925408782,0.5219607019018598,0.5312791271451286,0.5197407189157337,0.5288555392874573,0.5230308934038408,0.5287894554283732,0.5466621906771533,0.5503311258278145,0.5454447264675704,0.5387830240310463,0.5323565323565324,0.5481978798586572,0.5534907401514669,0.5361210266870644,0.5366596638655462,0.5303703703703704,0.5291210730674197,0.5266700150678051,0.5254703992657183,0.5255600229753016,0.52,0.50397957651299,0.5360375460945357,0.5489939473253722,0.5584264640143048,0.5669736482173794,0.5512387387387387,0.5219094602437608,0.5339970345265833,0.537051282051282,0.5342134782983864,0.5266188324918062,0.531566366704162,0.5478343645882913,0.5593722755013077,0.5290215588723052,0.5665965014154025,0.542726286609618,0.5311483031148303,0.5492404400209534,0.5358841447122321,0.5414989051352688,0.5342128801431127,0.5428067078552515,0.54859025997803,0.5189271770497369,0.519143923031612,0.5261492338441041,0.538109672085974,0.5303012542492088,0.5279603858960129,0.5103205439533754,0.4911119619318493,0.4796691537972121,0.4929371294737612,0.5034358811713712,0.510156971375808,0.5075282308657465,0.5048046124279308,0.5031112737920937,0.5306102648685359,0.5127066115702479,0.5183146305314419,0.5172348574400681,0.5363333115311661,0.538812091751481,0.525,0.5363997632535723,0.5122912366297617,0.5277220077220077,0.5569122516556292,0.5274584929757343,0.5339287358438449,0.5267818387012762,0.5356509999156189,0.5322266530650097,0.5283286118980169,0.5546677320548675,0.5532800532800533,0.5477743870356729,0.5575888051668461,0.5057603230531692,0.5347218226799378,0.5264651049687931,0.5113720277654704,0.4883349080601788,0.4964833955109779,0.5145699239321241,0.5185076133591319,0.5130871461905245,0.5312357501139989,0.5318607681330272,0.5085687508367921,0.5059356966199505,0.5165949600491703,0.510706393355178,0.5102588783626324,0.5288388974736766,0.5424126984126985,0.5363495746326373,0.5319441264025647,0.5214101738570509,0.5377935662127492,0.5405263157894736,0.5231788079470199,0.5288676711920036,0.5461123375030661,0.5263598326359833,0.5134802665299846,0.5393499115209089,0.5339826381537158,0.5282403633399324,0.5212933753943217,0.5377026074700493,0.5327990135635019,0.5194585448392555,0.5060690943043884,0.5145917990395271,0.520468703648769,0.5356006031160998,0.512439275725011,0.5218508997429306,0.5306357172731074,0.5346748113546532,0.5200279134682484,0.5059422750424448,0.5233329313879175,0.5352534324352225,0.5022426960843739,0.508235294117647,0.5303178484107579,0.5232393702272085,0.536389977842168,0.5501543209876543,0.5212403656569278,0.5094714809000523,0.5042082818987768,0.5150386708679462,0.5144588963252936,0.5205380735046842,0.5324205631619736,0.5113636363636364,0.5237065637065637,0.5071962358151121,0.5060425858430846,0.5115435647943388,0.5503183918092146,0.5581451515517449,0.5374003678724709,0.5537607891491986,0.5297746443409291,0.5488292404340377,0.5188221515810607,0.5503173164097914,0.5417344803534062,0.5329448677722994,0.5530612244897959,0.551044355785027,0.5608883405305367,0.5263157894736842,0.5507414571244359,0.553958143767061,0.5448490230905861,0.5428468338444885,0.5276344878408253,0.5351826792963464,0.530147470599216,0.5367487328023172,0.5101494534909659,0.5529552442774172,0.5587557603686636,0.5465447154471544,0.5396525679758308,0.536917694715522,0.5514079895219385,0.54373603711978,0.5389755011135857,0.5420161157704325,0.496783416726233,0.51015102748205,0.5164208456243854,0.5268711656441718,0.5280546852748504,0.5180406735184779,0.5298288085768632,0.5231388329979879,0.5108695652173914,0.5297055730809674,0.5571273122959739,0.5591210324380886,0.5228519195612431,0.5078740157480315,0.5106454883406556,0.5184402161146348,0.5573074590661007,0.5436096718480138,0.5410668924640135,0.5398721867185329,0.5562632696390658,0.5581395348837209,0.5306919642857143,0.5208143421452446,0.5794419134396356,0.5605247465712582,0.5300751879699248,0.5213302752293578,0.5241742460507419,0.5556939945178171,0.5316674913409203,0.5364875701684042,0.5254739336492891,0.5227272727272727,0.5024463680843057,0.4895121658875704,0.5140682212372326,0.5196692892385651,0.5277327278624716,0.5372868791697554,0.5348314606741573,0.5447019867549668,0.5557559362789299,0.5264947010597879,0.5611604809200209,0.538543897216274,0.5457328740910831,0.5414793624846751,0.5413171140939598,0.5403745778323611,0.5378787878787878,0.5239562902773887,0.531003937007874,0.5400083787180562,0.5332256575911398,0.5465020576131687,0.4964501709176966,0.5410315305909376,0.5175494575622208,0.5483562081072455,0.5180591025173295,0.5413504572874099,0.5404053039779835,0.5428571428571428,0.5334481768590296,0.5028601144045762,0.5321257689678742,0.5558949297495418,0.5309979838709677,0.5316656111462951,0.5461139896373057,0.5355417529330573,0.526589869023197,0.5179014533853243,0.5531824910128992,0.5392230388480576,0.5713514097439776,0.6228169201142174,0.6103883771721584,0.6067263050617144,0.596822995461422,0.6087154570300638,0.6089393110989612,0.5975728155339806,0.5831297600229918,0.6018657839301835,0.5641025641025641,0.5472334682860999,0.5673230441724322,0.5408186599588384,0.5293398533007335,0.5245166713365088,0.5224481964696853,0.5510736504113988,0.5626450116009281,0.5771180304127443,0.5561853129620503,0.5482977683661834,0.5302303262955854,0.5339172568020872,0.5400238948626045,0.5386002886002886,0.5250634517766497,0.5636638147961751,0.5358863198458574,0.5006580679126086,0.5289336801040312,0.5230724299065421,0.5368715083798883,0.5627720703465088,0.553680981595092,0.5562829605382796,0.5482700892857143,0.5502793296089385,0.5428089413749473,0.5409517070581992,0.5677058543550689,0.5501485035412383,0.5587900841482829,0.5380841724761167,0.5478968792401628,0.5456128891791857,0.5582440373314898,0.5784463061690784,0.5339832114811807,0.5191512813447231,0.5325264750378215,0.5397775876817793,0.5131069067700023,0.5320901994796184,0.5347445927289461,0.5250706214689266,0.5660674865972879,0.5532710280373832,0.53125,0.5414246407826353,0.5407407407407407,0.5382808830436825,0.5563594821020563,0.5443425076452599,0.5843373493975904,0.5393489905232798,0.5532994923857868,0.5525835866261398,0.5691220988900101,0.5499640546369519,0.5241054210413542,0.5491821644633654,0.5454328642976389,0.5420751633986928,0.5623953098827471,0.5588806840264283,0.5365517241379311,0.5666167664670658,0.5437697160883279,0.5402784684236698,0.5441632653061225,0.5625841184387618,0.5265654648956357,0.5459928103193064,0.5529526598340654,0.5294117647058824,0.5087778528021607,0.5208728652751423,0.534,0.5099515868746638,0.5474802879670895,0.5254171517268141,0.5308714918759232,0.5380982367758187,0.5183183183183183,0.5466399197592778,0.5452478399272397,0.5560447003047748,0.5277419354838709,0.5403747870528111,0.4986510791366906,0.5302922309337135,0.5546666666666666,0.5855227882037534,0.5430555555555555,0.519795062878435,0.5033647375504711,0.5005353319057816,0.535977105478332,0.5303842716711349,0.5606143719144268,0.5492895204262878,0.5427350427350427,0.5350509286998203,0.5138888888888888,0.5184926727145848,0.5768463073852296,0.5384615384615384,0.541501976284585,0.5224744103248776,0.5476302205537307,0.5309330628803245,0.5471186440677966,0.5868566176470589,0.5483870967741935,0.5778811026237131,0.5433890817292522,0.5689212328767124,0.5342883267296555,0.5431519699812383,0.5327421555252387,0.5490438001233806,0.5752126366950182,0.5638676844783715,0.5463780918727915,0.5426328933285766,0.5362748001184483,0.5467751307379429,0.5383502170767004,0.5268703898840885,0.5391868002357101,0.5492957746478874,0.4941543257989088,0.4907178217821782,0.5075431034482759,0.5127420998980632,0.5559322033898305,0.5947503201024328,0.5364875701684042,0.5324869305451829,0.5256756756756756,0.5445486518171161,0.4987510407993339,0.5107142857142857,0.5540380047505938,0.5042016806722689,0.5282224094355518,0.5711982804943578,0.5172690763052209,0.5630498533724341,0.5329712955779674,0.5498981670061099,0.5545263157894736,0.5506072874493927,0.5440613026819924,0.5447214076246334,0.5518774703557312,0.5286284953395473,0.5394436844505244,0.5462386200287495,0.5201577563540754,0.5149670218163369,0.5531070956368445,0.5303081611920081,0.5284679089026915,0.5307595731324545,0.5461157024793388,0.5797147385103011,0.5712043938799529,0.5749248604551309,0.5637903652726309,0.5762949136724218,0.5765199161425576,0.5562737108332162,0.5456896551724137,0.5511710794297352,0.5535469734319365,0.5549247800170309,0.5520770620108368,0.5393853761921582,0.5274595021730542,0.5194936708860759,0.568873852102465,0.5247670426679745,0.5851926977687627,0.6395472232048107,0.661727349703641,0.6451331296377127,0.5664464993394979,0.5438483777706392,0.5608157319737801,0.5501432664756447,0.5473537604456824,0.6328671328671329,0.6291390728476821,0.5987616099071208,0.5901077375122429,0.5331715210355987,0.5555555555555556,0.5448877805486284,0.5457050987597611,0.4902409002989274,0.5085514114980424,0.5229142185663925,0.4926362297496318,0.5360134003350083,0.5705944798301487,0.5370975268315445,0.5392156862745098,0.5507518796992481,0.5564773452456924,0.5713650511338373,0.5359477124183006,0.5553250345781466,0.566430469441984,0.5920223932820154,0.569828722002635,0.5082482325216026,0.5331262939958592,0.5360824742268041,0.5815602836879432,0.5455333911535126,0.5460687960687961,0.5521172638436482,0.5404929577464789,0.5764604810996563,0.5387029288702929,0.5633027522935781,0.5352839931153184,0.5703422053231939,0.5394646533973919,0.5441048034934498,0.5366146458583433,0.5323645970937912,0.5809822361546501,0.5595959595959596,0.5452079566003617,0.5479262672811059,0.5394358088658607,0.5587010824313072,0.5681403828623519,0.5545142143080287,0.5384294068504595,0.5436357364608486,0.5349075975359343,0.5560625814863103,0.5532112218873174,0.5740694479140644,0.5738947368421052,0.5520088057237205,0.5678785857238159,0.5525298098614244,0.5366346639717029,0.5652298030259777,0.5643776824034334,0.5718796042938329,0.5597075548334687,0.5489401496259352,0.5529251081265649,0.5320817490494296,0.5392567756356524,0.5187174479166666,0.5754435547181661,0.5823918971562135,0.5390932420872541,0.5523378094159521,0.563780260707635,0.5716374269005848,0.5546719681908548,0.5457413249211357,0.5519667412855772,0.5541512312050556,0.5453529223890131,0.5496780128794848,0.5408863920099876,0.5400817353033637,0.5387509405568096,0.5359056806002144,0.5473242811501597,0.5488994148787963,0.5569217723129107,0.5914893617021276,0.548488949030221,0.5678863017840944,0.5562817719680465,0.5363066715812753,0.5618908382066277,0.5589530966571651,0.5320139697322468,0.5604155276107162,0.5579589572933998,0.5712116610993015,0.5692848020434227,0.563317384370016,0.5919093131807068,0.5781519861830743,0.5734035549703752,0.5877474585339754,0.5485216072782411,0.5765503875968992,0.5710464727515882,0.5623062616243025,0.5725681935151827,0.5661333333333334,0.5391040242976461,0.5357607282184655,0.5415384615384615,0.5439056356487549,0.5225893459204316,0.5713000449842555,0.5401746724890829,0.5201515673441268,0.5468914646996839],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"market_interest_index"},"xaxis":{"title":{"text":"date"}},"yaxis":{"title":{"text":"value"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('990c2328-22a7-4798-b63d-2f2cb79f4b95');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


> ... 둘 다 꽝인 것 같다. </br>
<br> 일단 노이즈가 너무 많아서 뭐가 뭔지 보이질 않았다. 그리고 market interest index의 경우 시황과는 상관없이 수렴증가하는 추세를 보여 index로서는 기능할 수 없었다. </br>
<br> positivity index의 경우 노이즈를 제거하면 어떨까 싶어서 저 상태에서 10일 이동평균선을 한 번 만들어보았다.


```python
# Making 10 Day Moving Average
result['positivity_index_10d'] = result['positivity_index'].rolling(10).mean()
result['positivity_index_10d'] = result['positivity_index_10d'].fillna(0)

# Plot the 10 Day Moving Average
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x = result['date'], y = result['positivity_index_10d'], name = 'positivity_index_10d'))
fig4.update_layout(title = 'positivity_index_10d', xaxis_title = 'date', yaxis_title = 'value')
fig4.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.18.2.min.js"></script>                <div id="e4b1b6b1-7d9b-454b-8214-21cf851c0b9a" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e4b1b6b1-7d9b-454b-8214-21cf851c0b9a")) {                    Plotly.newPlot(                        "e4b1b6b1-7d9b-454b-8214-21cf851c0b9a",                        [{"name":"positivity_index_10d","x":["2017-05-01","2017-05-02","2017-05-03","2017-05-04","2017-05-05","2017-05-06","2017-05-07","2017-05-08","2017-05-09","2017-05-10","2017-05-11","2017-05-12","2017-05-13","2017-05-14","2017-05-15","2017-05-16","2017-05-17","2017-05-18","2017-05-19","2017-05-20","2017-05-21","2017-05-22","2017-05-23","2017-05-24","2017-05-25","2017-05-26","2017-05-27","2017-05-28","2017-05-29","2017-05-30","2017-05-31","2017-06-01","2017-06-02","2017-06-03","2017-06-04","2017-06-05","2017-06-06","2017-06-07","2017-06-08","2017-06-09","2017-06-10","2017-06-11","2017-06-12","2017-06-13","2017-06-14","2017-06-15","2017-06-16","2017-06-17","2017-06-18","2017-06-19","2017-06-20","2017-06-21","2017-06-22","2017-06-23","2017-06-24","2017-06-25","2017-06-26","2017-06-27","2017-06-28","2017-06-29","2017-06-30","2017-07-01","2017-07-02","2017-07-03","2017-07-04","2017-07-05","2017-07-06","2017-07-07","2017-07-08","2017-07-09","2017-07-10","2017-07-11","2017-07-12","2017-07-13","2017-07-14","2017-07-15","2017-07-16","2017-07-17","2017-07-18","2017-07-19","2017-07-20","2017-07-21","2017-07-22","2017-07-23","2017-07-24","2017-07-25","2017-07-26","2017-07-27","2017-07-28","2017-07-29","2017-07-30","2017-07-31","2017-08-01","2017-08-02","2017-08-03","2017-08-04","2017-08-05","2017-08-06","2017-08-07","2017-08-08","2017-08-09","2017-08-10","2017-08-11","2017-08-12","2017-08-13","2017-08-14","2017-08-15","2017-08-16","2017-08-17","2017-08-18","2017-08-19","2017-08-20","2017-08-21","2017-08-22","2017-08-23","2017-08-24","2017-08-25","2017-08-26","2017-08-27","2017-08-28","2017-08-29","2017-08-30","2017-08-31","2017-09-01","2017-09-02","2017-09-03","2017-09-04","2017-09-05","2017-09-06","2017-09-07","2017-09-08","2017-09-09","2017-09-10","2017-09-11","2017-09-12","2017-09-13","2017-09-14","2017-09-15","2017-09-16","2017-09-17","2017-09-18","2017-09-19","2017-09-20","2017-09-21","2017-09-22","2017-09-23","2017-09-24","2017-09-25","2017-09-26","2017-09-27","2017-09-28","2017-09-29","2017-09-30","2017-10-01","2017-10-02","2017-10-03","2017-10-04","2017-10-05","2017-10-06","2017-10-07","2017-10-08","2017-10-09","2017-10-10","2017-10-11","2017-10-12","2017-10-13","2017-10-14","2017-10-15","2017-10-16","2017-10-17","2017-10-18","2017-10-19","2017-10-20","2017-10-21","2017-10-22","2017-10-23","2017-10-24","2017-10-25","2017-10-26","2017-10-27","2017-10-28","2017-10-29","2017-10-30","2017-10-31","2017-11-01","2017-11-02","2017-11-03","2017-11-04","2017-11-05","2017-11-06","2017-11-07","2017-11-08","2017-11-09","2017-11-10","2017-11-11","2017-11-12","2017-11-13","2017-11-14","2017-11-15","2017-11-16","2017-11-17","2017-11-18","2017-11-19","2017-11-20","2017-11-21","2017-11-22","2017-11-23","2017-11-24","2017-11-25","2017-11-26","2017-11-27","2017-11-28","2017-11-29","2017-11-30","2017-12-01","2017-12-02","2017-12-03","2017-12-04","2017-12-05","2017-12-06","2017-12-07","2017-12-08","2017-12-09","2017-12-10","2017-12-11","2017-12-12","2017-12-13","2017-12-14","2017-12-15","2017-12-16","2017-12-17","2017-12-18","2017-12-19","2017-12-20","2017-12-21","2017-12-22","2017-12-23","2017-12-24","2017-12-25","2017-12-26","2017-12-27","2017-12-28","2017-12-29","2017-12-30","2017-12-31","2018-01-01","2018-01-02","2018-01-03","2018-01-04","2018-01-05","2018-01-06","2018-01-07","2018-01-08","2018-01-09","2018-01-10","2018-01-11","2018-01-12","2018-01-13","2018-01-14","2018-01-15","2018-01-16","2018-01-17","2018-01-18","2018-01-19","2018-01-20","2018-01-21","2018-01-22","2018-01-23","2018-01-24","2018-01-25","2018-01-26","2018-01-27","2018-01-28","2018-01-29","2018-01-30","2018-01-31","2018-02-01","2018-02-02","2018-02-03","2018-02-04","2018-02-05","2018-02-06","2018-02-07","2018-02-08","2018-02-09","2018-02-10","2018-02-11","2018-02-12","2018-02-13","2018-02-14","2018-02-15","2018-02-16","2018-02-17","2018-02-18","2018-02-19","2018-02-20","2018-02-21","2018-02-22","2018-02-23","2018-02-24","2018-02-25","2018-02-26","2018-02-27","2018-02-28","2018-03-01","2018-03-02","2018-03-03","2018-03-04","2018-03-05","2018-03-06","2018-03-07","2018-03-08","2018-03-09","2018-03-10","2018-03-11","2018-03-12","2018-03-13","2018-03-14","2018-03-15","2018-03-16","2018-03-17","2018-03-18","2018-03-19","2018-03-20","2018-03-21","2018-03-22","2018-03-23","2018-03-24","2018-03-25","2018-03-26","2018-03-27","2018-03-28","2018-03-29","2018-03-30","2018-03-31","2018-04-01","2018-04-02","2018-04-03","2018-04-04","2018-04-05","2018-04-06","2018-04-07","2018-04-08","2018-04-09","2018-04-10","2018-04-11","2018-04-12","2018-04-13","2018-04-14","2018-04-15","2018-04-16","2018-04-17","2018-04-18","2018-04-19","2018-04-20","2018-04-21","2018-04-22","2018-04-23","2018-04-24","2018-04-25","2018-04-26","2018-04-27","2018-04-28","2018-04-29","2018-04-30","2018-05-01","2018-05-02","2018-05-03","2018-05-04","2018-05-05","2018-05-06","2018-05-07","2018-05-08","2018-05-09","2018-05-10","2018-05-11","2018-05-12","2018-05-13","2018-05-14","2018-05-15","2018-05-16","2018-05-17","2018-05-18","2018-05-19","2018-05-20","2018-05-21","2018-05-22","2018-05-23","2018-05-24","2018-05-25","2018-05-26","2018-05-27","2018-05-28","2018-05-29","2018-05-30","2018-05-31","2018-06-01","2018-06-02","2018-06-03","2018-06-04","2018-06-05","2018-06-06","2018-06-07","2018-06-08","2018-06-09","2018-06-10","2018-06-11","2018-06-12","2018-06-13","2018-06-14","2018-06-15","2018-06-16","2018-06-17","2018-06-18","2018-06-19","2018-06-20","2018-06-21","2018-06-22","2018-06-23","2018-06-24","2018-06-25","2018-06-26","2018-06-27","2018-06-28","2018-06-29","2018-06-30","2018-07-01","2018-07-02","2018-07-03","2018-07-04","2018-07-05","2018-07-06","2018-07-07","2018-07-08","2018-07-09","2018-07-10","2018-07-11","2018-07-12","2018-07-13","2018-07-14","2018-07-15","2018-07-16","2018-07-17","2018-07-18","2018-07-19","2018-07-20","2018-07-21","2018-07-22","2018-07-23","2018-07-24","2018-07-25","2018-07-26","2018-07-27","2018-07-28","2018-07-29","2018-07-30","2018-07-31","2018-08-01","2018-08-02","2018-08-03","2018-08-04","2018-08-05","2018-08-06","2018-08-07","2018-08-08","2018-08-09","2018-08-10","2018-08-11","2018-08-12","2018-08-13","2018-08-14","2018-08-15","2018-08-16","2018-08-17","2018-08-18","2018-08-19","2018-08-20","2018-08-21","2018-08-22","2018-08-23","2018-08-24","2018-08-25","2018-08-26","2018-08-27","2018-08-28","2018-08-29","2018-08-30","2018-08-31","2018-09-01","2018-09-02","2018-09-03","2018-09-04","2018-09-05","2018-09-06","2018-09-07","2018-09-08","2018-09-09","2018-09-10","2018-09-11","2018-09-12","2018-09-13","2018-09-14","2018-09-15","2018-09-16","2018-09-17","2018-09-18","2018-09-19","2018-09-20","2018-09-21","2018-09-22","2018-09-23","2018-09-24","2018-09-25","2018-09-26","2018-09-27","2018-09-28","2018-09-29","2018-09-30","2018-10-01","2018-10-02","2018-10-03","2018-10-04","2018-10-05","2018-10-06","2018-10-07","2018-10-08","2018-10-09","2018-10-10","2018-10-11","2018-10-12","2018-10-13","2018-10-14","2018-10-15","2018-10-16","2018-10-17","2018-10-18","2018-10-19","2018-10-20","2018-10-21","2018-10-22","2018-10-23","2018-10-24","2018-10-25","2018-10-26","2018-10-27","2018-10-28","2018-10-29","2018-10-30","2018-10-31","2018-11-01","2018-11-02","2018-11-03","2018-11-04","2018-11-05","2018-11-06","2018-11-07","2018-11-08","2018-11-09","2018-11-10","2018-11-11","2018-11-12","2018-11-13","2018-11-14","2018-11-15","2018-11-16","2018-11-17","2018-11-18","2018-11-19","2018-11-20","2018-11-21","2018-11-22","2018-11-23","2018-11-24","2018-11-25","2018-11-26","2018-11-27","2018-11-28","2018-11-29","2018-11-30","2018-12-01","2018-12-02","2018-12-03","2018-12-04","2018-12-05","2018-12-06","2018-12-07","2018-12-08","2018-12-09","2018-12-10","2018-12-11","2018-12-12","2018-12-13","2018-12-14","2018-12-15","2018-12-16","2018-12-17","2018-12-18","2018-12-19","2018-12-20","2018-12-21","2018-12-22","2018-12-23","2018-12-24","2018-12-25","2018-12-26","2018-12-27","2018-12-28","2018-12-29","2018-12-30","2018-12-31","2019-01-01","2019-01-02","2019-01-03","2019-01-04","2019-01-05","2019-01-06","2019-01-07","2019-01-08","2019-01-09","2019-01-10","2019-01-11","2019-01-12","2019-01-13","2019-01-14","2019-01-15","2019-01-16","2019-01-17","2019-01-18","2019-01-19","2019-01-20","2019-01-21","2019-01-22","2019-01-23","2019-01-24","2019-01-25","2019-01-26","2019-01-27","2019-01-28","2019-01-29","2019-01-30","2019-01-31","2019-02-01","2019-02-02","2019-02-03","2019-02-04","2019-02-05","2019-02-06","2019-02-07","2019-02-08","2019-02-09","2019-02-10","2019-02-11","2019-02-12","2019-02-13","2019-02-14","2019-02-15","2019-02-16","2019-02-17","2019-02-18","2019-02-19","2019-02-20","2019-02-21","2019-02-22","2019-02-23","2019-02-24","2019-02-25","2019-02-26","2019-02-27","2019-02-28","2019-03-01","2019-03-02","2019-03-03","2019-03-04","2019-03-05","2019-03-06","2019-03-07","2019-03-08","2019-03-09","2019-03-10","2019-03-11","2019-03-12","2019-03-13","2019-03-14","2019-03-15","2019-03-16","2019-03-17","2019-03-18","2019-03-19","2019-03-20","2019-03-21","2019-03-22","2019-03-23","2019-03-24","2019-03-25","2019-03-26","2019-03-27","2019-03-28","2019-03-29","2019-03-30","2019-03-31","2019-04-01","2019-04-02","2019-04-03","2019-04-04","2019-04-05","2019-04-06","2019-04-07","2019-04-08","2019-04-09","2019-04-10","2019-04-11","2019-04-12","2019-04-13","2019-04-14","2019-04-15","2019-04-16","2019-04-17","2019-04-18","2019-04-19","2019-04-20","2019-04-21","2019-04-22","2019-04-23","2019-04-24","2019-04-25","2019-04-26","2019-04-27","2019-04-28","2019-04-29","2019-04-30","2019-05-01","2019-05-02","2019-05-03","2019-05-04","2019-05-05","2019-05-06","2019-05-07","2019-05-08","2019-05-09","2019-05-10","2019-05-11","2019-05-12","2019-05-13","2019-05-14","2019-05-15","2019-05-16","2019-05-17","2019-05-18","2019-05-19","2019-05-20","2019-05-21","2019-05-22","2019-05-23","2019-05-24","2019-05-25","2019-05-26","2019-05-27","2019-05-28","2019-05-29","2019-05-30","2019-05-31","2019-06-01","2019-06-02","2019-06-03","2019-06-04","2019-06-05","2019-06-06","2019-06-07","2019-06-08","2019-06-09","2019-06-10","2019-06-11","2019-06-12","2019-06-13","2019-06-14","2019-06-15","2019-06-16","2019-06-17","2019-06-18","2019-06-19","2019-06-20","2019-06-21","2019-06-22","2019-06-23","2019-06-24","2019-06-25","2019-06-26","2019-06-27","2019-06-28","2019-06-29","2019-06-30","2019-07-01","2019-07-02","2019-07-03","2019-07-04","2019-07-05","2019-07-06","2019-07-07","2019-07-08","2019-07-09","2019-07-10","2019-07-11","2019-07-12","2019-07-13","2019-07-14","2019-07-15","2019-07-16","2019-07-17","2019-07-18","2019-07-19","2019-07-20","2019-07-21","2019-07-22","2019-07-23","2019-07-24","2019-07-25","2019-07-26","2019-07-27","2019-07-28","2019-07-29","2019-07-30","2019-07-31","2019-08-01","2019-08-02","2019-08-03","2019-08-04","2019-08-05","2019-08-06","2019-08-07","2019-08-08","2019-08-09","2019-08-10","2019-08-11","2019-08-12","2019-08-13","2019-08-14","2019-08-15","2019-08-16","2019-08-17","2019-08-18","2019-08-19","2019-08-20","2019-08-21","2019-08-22","2019-08-23","2019-08-24","2019-08-25","2019-08-26","2019-08-27","2019-08-28","2019-08-29","2019-08-30","2019-08-31","2019-09-01","2019-09-02","2019-09-03","2019-09-04","2019-09-05","2019-09-06","2019-09-07","2019-09-08","2019-09-09","2019-09-10","2019-09-11","2019-09-12","2019-09-13","2019-09-14","2019-09-15","2019-09-16","2019-09-17","2019-09-18","2019-09-19","2019-09-20","2019-09-21","2019-09-22","2019-09-23","2019-09-24","2019-09-25","2019-09-26","2019-09-27","2019-09-28","2019-09-29","2019-09-30","2019-10-01","2019-10-02","2019-10-03","2019-10-04","2019-10-05","2019-10-06","2019-10-07","2019-10-08","2019-10-09","2019-10-10","2019-10-11","2019-10-12","2019-10-13","2019-10-14","2019-10-15","2019-10-16","2019-10-17","2019-10-18","2019-10-19","2019-10-20","2019-10-21","2019-10-22","2019-10-23","2019-10-24","2019-10-25","2019-10-26","2019-10-27","2019-10-28","2019-10-29","2019-10-30","2019-10-31","2019-11-01","2019-11-02","2019-11-03","2019-11-04","2019-11-05","2019-11-06","2019-11-07","2019-11-08","2019-11-09","2019-11-10","2019-11-11","2019-11-12","2019-11-13","2019-11-14","2019-11-15","2019-11-16","2019-11-17","2019-11-18","2019-11-19","2019-11-20","2019-11-21","2019-11-22","2019-11-23","2019-11-24","2019-11-25","2019-11-26","2019-11-27","2019-11-28","2019-11-29","2019-11-30","2019-12-01","2019-12-02","2019-12-03","2019-12-04","2019-12-05","2019-12-06","2019-12-07","2019-12-08","2019-12-09","2019-12-10","2019-12-11","2019-12-12","2019-12-13","2019-12-14","2019-12-15","2019-12-16","2019-12-17","2019-12-18","2019-12-19","2019-12-20","2019-12-21","2019-12-22","2019-12-23","2019-12-24","2019-12-25","2019-12-26","2019-12-27","2019-12-28","2019-12-29","2019-12-30","2019-12-31","2020-01-01","2020-01-02","2020-01-03","2020-01-04","2020-01-05","2020-01-06","2020-01-07","2020-01-08","2020-01-09","2020-01-10","2020-01-11","2020-01-12","2020-01-13","2020-01-14","2020-01-15","2020-01-16","2020-01-17","2020-01-18","2020-01-19","2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-25","2020-01-26","2020-01-27","2020-01-28","2020-01-29","2020-01-30","2020-01-31","2020-02-01","2020-02-02","2020-02-03","2020-02-04","2020-02-05","2020-02-06","2020-02-07","2020-02-08","2020-02-09","2020-02-10","2020-02-11","2020-02-12","2020-02-13","2020-02-14","2020-02-15","2020-02-16","2020-02-17","2020-02-18","2020-02-19","2020-02-20","2020-02-21","2020-02-22","2020-02-23","2020-02-24","2020-02-25","2020-02-26","2020-02-27","2020-02-28","2020-02-29","2020-03-01","2020-03-02","2020-03-03","2020-03-04","2020-03-05","2020-03-06","2020-03-07","2020-03-08","2020-03-09","2020-03-10","2020-03-11","2020-03-12","2020-03-13","2020-03-14","2020-03-15","2020-03-16","2020-03-17","2020-03-18","2020-03-19","2020-03-20","2020-03-21","2020-03-22","2020-03-23","2020-03-24","2020-03-25","2020-03-26","2020-03-27","2020-03-28","2020-03-29","2020-03-30","2020-03-31","2020-04-01","2020-04-02","2020-04-03","2020-04-04","2020-04-05","2020-04-06","2020-04-07","2020-04-08","2020-04-09","2020-04-10","2020-04-11","2020-04-12","2020-04-13","2020-04-14","2020-04-15","2020-04-16","2020-04-17","2020-04-18","2020-04-19","2020-04-20","2020-04-21","2020-04-22","2020-04-23","2020-04-24","2020-04-25","2020-04-26","2020-04-27","2020-04-28","2020-04-29","2020-04-30","2020-05-01","2020-05-02","2020-05-03","2020-05-04","2020-05-05","2020-05-06","2020-05-07","2020-05-08","2020-05-09","2020-05-10","2020-05-11","2020-05-12","2020-05-13","2020-05-14","2020-05-15","2020-05-16","2020-05-17","2020-05-18","2020-05-19","2020-05-20","2020-05-21","2020-05-22","2020-05-23","2020-05-24","2020-05-25","2020-05-26","2020-05-27","2020-05-28","2020-05-29","2020-05-30","2020-05-31","2020-06-01","2020-06-02","2020-06-03","2020-06-04","2020-06-05","2020-06-06","2020-06-07","2020-06-08","2020-06-09","2020-06-10","2020-06-11","2020-06-12","2020-06-13","2020-06-14","2020-06-15","2020-06-16","2020-06-17","2020-06-18","2020-06-19","2020-06-20","2020-06-21","2020-06-22","2020-06-23","2020-06-24","2020-06-25","2020-06-26","2020-06-27","2020-06-28","2020-06-29","2020-06-30","2020-07-01","2020-07-02","2020-07-03","2020-07-04","2020-07-05","2020-07-06","2020-07-07","2020-07-08","2020-07-09","2020-07-10","2020-07-11","2020-07-12","2020-07-13","2020-07-14","2020-07-15","2020-07-16","2020-07-17","2020-07-18","2020-07-19","2020-07-20","2020-07-21","2020-07-22","2020-07-23","2020-07-24","2020-07-25","2020-07-26","2020-07-27","2020-07-28","2020-07-29","2020-07-30","2020-07-31","2020-08-01","2020-08-02","2020-08-03","2020-08-04","2020-08-05","2020-08-06","2020-08-07","2020-08-08","2020-08-09","2020-08-10","2020-08-11","2020-08-12","2020-08-13","2020-08-14","2020-08-15","2020-08-16","2020-08-17","2020-08-18","2020-08-19","2020-08-20","2020-08-21","2020-08-22","2020-08-23","2020-08-24","2020-08-25","2020-08-26","2020-08-27","2020-08-28","2020-08-29","2020-08-30","2020-08-31","2020-09-01","2020-09-02","2020-09-03","2020-09-04","2020-09-05","2020-09-06","2020-09-07","2020-09-08","2020-09-09","2020-09-10","2020-09-11","2020-09-12","2020-09-13","2020-09-14","2020-09-15","2020-09-16","2020-09-17","2020-09-18","2020-09-19","2020-09-20","2020-09-21","2020-09-22","2020-09-23","2020-09-24","2020-09-25","2020-09-26","2020-09-27","2020-09-28","2020-09-29","2020-09-30","2020-10-01","2020-10-02","2020-10-03","2020-10-04","2020-10-05","2020-10-06","2020-10-07","2020-10-08","2020-10-09","2020-10-10","2020-10-11","2020-10-12","2020-10-13","2020-10-14","2020-10-15","2020-10-16","2020-10-17","2020-10-18","2020-10-19","2020-10-20","2020-10-21","2020-10-22","2020-10-23","2020-10-24","2020-10-25","2020-10-26","2020-10-27","2020-10-28","2020-10-29","2020-10-30","2020-10-31","2020-11-01","2020-11-02","2020-11-03","2020-11-04","2020-11-05","2020-11-06","2020-11-07","2020-11-08","2020-11-09","2020-11-10","2020-11-11","2020-11-12","2020-11-13","2020-11-14","2020-11-15","2020-11-16","2020-11-17","2020-11-18","2020-11-19","2020-11-20","2020-11-21","2020-11-22","2020-11-23","2020-11-24","2020-11-25","2020-11-26","2020-11-27","2020-11-28","2020-11-29","2020-11-30","2020-12-01","2020-12-02","2020-12-03","2020-12-04","2020-12-05","2020-12-06","2020-12-07","2020-12-08","2020-12-09","2020-12-10","2020-12-11","2020-12-12","2020-12-13","2020-12-14","2020-12-15","2020-12-16","2020-12-17","2020-12-18","2020-12-19","2020-12-20","2020-12-21","2020-12-22","2020-12-23","2020-12-24","2020-12-25","2020-12-26","2020-12-27","2020-12-28","2020-12-29","2020-12-30","2020-12-31","2021-01-01","2021-01-02","2021-01-03","2021-01-04","2021-01-05","2021-01-06","2021-01-07","2021-01-08","2021-01-09","2021-01-10","2021-01-11","2021-01-12","2021-01-13","2021-01-14","2021-01-15","2021-01-16","2021-01-17","2021-01-18","2021-01-19","2021-01-20","2021-01-21","2021-01-22","2021-01-23","2021-01-24","2021-01-25","2021-01-26","2021-01-27","2021-01-28","2021-01-29","2021-01-30","2021-01-31","2021-02-01","2021-02-02","2021-02-03","2021-02-04","2021-02-05","2021-02-06","2021-02-07","2021-02-08","2021-02-09","2021-02-10","2021-02-11","2021-02-12","2021-02-13","2021-02-14","2021-02-15","2021-02-16","2021-02-17","2021-02-18","2021-02-19","2021-02-20","2021-02-21","2021-02-22","2021-02-23","2021-02-24","2021-02-25","2021-02-26","2021-02-27","2021-02-28","2021-03-01","2021-03-02","2021-03-03","2021-03-04","2021-03-05","2021-03-06","2021-03-07","2021-03-08","2021-03-09","2021-03-10","2021-03-11","2021-03-12","2021-03-13","2021-03-14","2021-03-15","2021-03-16","2021-03-17","2021-03-18","2021-03-19","2021-03-20","2021-03-21","2021-03-22","2021-03-23","2021-03-24","2021-03-25","2021-03-26","2021-03-27","2021-03-28","2021-03-29","2021-03-30","2021-03-31","2021-04-01","2021-04-02","2021-04-03","2021-04-04","2021-04-05","2021-04-06","2021-04-07","2021-04-08","2021-04-09","2021-04-10","2021-04-11","2021-04-12","2021-04-13","2021-04-14","2021-04-15","2021-04-16","2021-04-17","2021-04-18","2021-04-19","2021-04-20","2021-04-21","2021-04-22","2021-04-23","2021-04-24","2021-04-25","2021-04-26","2021-04-27","2021-04-28","2021-04-29","2021-04-30","2021-05-01","2021-05-02","2021-05-03","2021-05-04","2021-05-05","2021-05-06","2021-05-07","2021-05-08","2021-05-09","2021-05-10","2021-05-11","2021-05-12","2021-05-13","2021-05-14","2021-05-15","2021-05-16","2021-05-17","2021-05-18","2021-05-19","2021-05-20","2021-05-21","2021-05-22","2021-05-23","2021-05-24","2021-05-25","2021-05-26","2021-05-27","2021-05-28","2021-05-29","2021-05-30","2021-05-31","2021-06-01","2021-06-02","2021-06-03","2021-06-04","2021-06-05","2021-06-06","2021-06-07","2021-06-08","2021-06-09","2021-06-10","2021-06-11","2021-06-12","2021-06-13","2021-06-14","2021-06-15","2021-06-16","2021-06-17","2021-06-18","2021-06-19","2021-06-20","2021-06-21","2021-06-22","2021-06-23","2021-06-24","2021-06-25","2021-06-26","2021-06-27","2021-06-28","2021-06-29","2021-06-30","2021-07-01","2021-07-02","2021-07-03","2021-07-04","2021-07-05","2021-07-06","2021-07-07","2021-07-08","2021-07-09","2021-07-10","2021-07-11","2021-07-12","2021-07-13","2021-07-14","2021-07-15","2021-07-16","2021-07-17","2021-07-18","2021-07-19","2021-07-20","2021-07-21","2021-07-22","2021-07-23","2021-07-24","2021-07-25","2021-07-26","2021-07-27","2021-07-28","2021-07-29","2021-07-30","2021-07-31","2021-08-01","2021-08-02","2021-08-03","2021-08-04","2021-08-05","2021-08-06","2021-08-07","2021-08-08","2021-08-09","2021-08-10","2021-08-11","2021-08-12","2021-08-13","2021-08-14","2021-08-15","2021-08-16","2021-08-17","2021-08-18","2021-08-19","2021-08-20","2021-08-21","2021-08-22","2021-08-23","2021-08-24","2021-08-25","2021-08-26","2021-08-27","2021-08-28","2021-08-29","2021-08-30","2021-08-31","2021-09-01","2021-09-02","2021-09-03","2021-09-04","2021-09-05","2021-09-06","2021-09-07","2021-09-08","2021-09-09","2021-09-10","2021-09-11","2021-09-12","2021-09-13","2021-09-14","2021-09-15","2021-09-16","2021-09-17","2021-09-18","2021-09-19","2021-09-20","2021-09-21","2021-09-22","2021-09-23","2021-09-24","2021-09-25","2021-09-26","2021-09-27","2021-09-28","2021-09-29","2021-09-30","2021-10-01","2021-10-02","2021-10-03","2021-10-04","2021-10-05","2021-10-06","2021-10-07","2021-10-08","2021-10-09","2021-10-10","2021-10-11","2021-10-12","2021-10-13","2021-10-14","2021-10-15","2021-10-16","2021-10-17","2021-10-18","2021-10-19","2021-10-20","2021-10-21","2021-10-22","2021-10-23","2021-10-24","2021-10-25","2021-10-26","2021-10-27","2021-10-28","2021-10-29","2021-10-30","2021-10-31","2021-11-01","2021-11-02","2021-11-03","2021-11-04","2021-11-05","2021-11-06","2021-11-07","2021-11-08","2021-11-09","2021-11-10","2021-11-11","2021-11-12","2021-11-13","2021-11-14","2021-11-15","2021-11-16","2021-11-17","2021-11-18","2021-11-19","2021-11-20","2021-11-21","2021-11-22","2021-11-23","2021-11-24","2021-11-25","2021-11-26","2021-11-27","2021-11-28","2021-11-29","2021-11-30","2021-12-01","2021-12-02","2021-12-03","2021-12-04","2021-12-05","2021-12-06","2021-12-07","2021-12-08","2021-12-09","2021-12-10","2021-12-11","2021-12-12","2021-12-13","2021-12-14","2021-12-15","2021-12-16","2021-12-17","2021-12-18","2021-12-19","2021-12-20","2021-12-21","2021-12-22","2021-12-23","2021-12-24","2021-12-25","2021-12-26","2021-12-27","2021-12-28","2021-12-29","2021-12-30","2021-12-31","2022-01-01","2022-01-02","2022-01-03","2022-01-04","2022-01-05","2022-01-06","2022-01-07","2022-01-08","2022-01-09","2022-01-10","2022-01-11","2022-01-12","2022-01-13","2022-01-14","2022-01-15","2022-01-16","2022-01-17","2022-01-18","2022-01-19","2022-01-20","2022-01-21","2022-01-22","2022-01-23","2022-01-24","2022-01-25","2022-01-26","2022-01-27","2022-01-28","2022-01-29","2022-01-30","2022-01-31","2022-02-01","2022-02-02","2022-02-03","2022-02-04","2022-02-05","2022-02-06","2022-02-07","2022-02-08","2022-02-09","2022-02-10","2022-02-11","2022-02-12","2022-02-13","2022-02-14","2022-02-15","2022-02-16","2022-02-17","2022-02-18","2022-02-19","2022-02-20","2022-02-21","2022-02-22","2022-02-23","2022-02-24","2022-02-25","2022-02-26","2022-02-27","2022-02-28","2022-03-01","2022-03-02","2022-03-03","2022-03-04","2022-03-05","2022-03-06","2022-03-07","2022-03-08","2022-03-09","2022-03-10","2022-03-11","2022-03-12","2022-03-13","2022-03-14","2022-03-15","2022-03-16","2022-03-17","2022-03-18","2022-03-19","2022-03-20","2022-03-21","2022-03-22","2022-03-23","2022-03-24","2022-03-25","2022-03-26","2022-03-27","2022-03-28","2022-03-29","2022-03-30","2022-03-31","2022-04-01","2022-04-02","2022-04-03","2022-04-04","2022-04-05","2022-04-06","2022-04-07","2022-04-08","2022-04-09","2022-04-10","2022-04-11","2022-04-12","2022-04-13","2022-04-14","2022-04-15","2022-04-16","2022-04-17","2022-04-18","2022-04-19","2022-04-20","2022-04-21","2022-04-22","2022-04-23","2022-04-24","2022-04-25","2022-04-26","2022-04-27","2022-04-28","2022-04-29","2022-04-30","2022-05-01","2022-05-02","2022-05-03","2022-05-04","2022-05-05","2022-05-06","2022-05-07","2022-05-08","2022-05-09","2022-05-10","2022-05-11","2022-05-12","2022-05-13","2022-05-14","2022-05-15","2022-05-16","2022-05-17","2022-05-18","2022-05-19","2022-05-20","2022-05-21","2022-05-22","2022-05-23","2022-05-24","2022-05-25","2022-05-26","2022-05-27","2022-05-28","2022-05-29","2022-05-30","2022-05-31","2022-06-01","2022-06-02","2022-06-03","2022-06-04","2022-06-05","2022-06-06","2022-06-07","2022-06-08","2022-06-09","2022-06-10","2022-06-11","2022-06-12","2022-06-13","2022-06-14","2022-06-15","2022-06-16","2022-06-17","2022-06-18","2022-06-19","2022-06-20","2022-06-21","2022-06-22","2022-06-23","2022-06-24","2022-06-25","2022-06-26","2022-06-27","2022-06-28","2022-06-29","2022-06-30","2022-07-01","2022-07-02","2022-07-03","2022-07-04","2022-07-05","2022-07-06","2022-07-07","2022-07-08","2022-07-09","2022-07-10","2022-07-11","2022-07-12","2022-07-13","2022-07-14","2022-07-15","2022-07-16","2022-07-17","2022-07-18","2022-07-19","2022-07-20","2022-07-21","2022-07-22","2022-07-23","2022-07-24","2022-07-25","2022-07-26","2022-07-27","2022-07-28","2022-07-29","2022-07-30","2022-07-31","2022-08-01","2022-08-02","2022-08-03","2022-08-04","2022-08-05","2022-08-06","2022-08-07","2022-08-08","2022-08-09","2022-08-10","2022-08-11","2022-08-12","2022-08-13","2022-08-14","2022-08-15","2022-08-16","2022-08-17","2022-08-18","2022-08-19","2022-08-20","2022-08-21","2022-08-22","2022-08-23","2022-08-24","2022-08-25","2022-08-26","2022-08-27","2022-08-28","2022-08-29","2022-08-30","2022-08-31","2022-09-01","2022-09-02","2022-09-03","2022-09-04","2022-09-05","2022-09-06","2022-09-07","2022-09-08","2022-09-09","2022-09-10","2022-09-11","2022-09-12","2022-09-13","2022-09-14","2022-09-15","2022-09-16","2022-09-17","2022-09-18","2022-09-19","2022-09-20","2022-09-21","2022-09-22","2022-09-23","2022-09-24","2022-09-25","2022-09-26","2022-09-27","2022-09-28","2022-09-29","2022-09-30","2022-10-01","2022-10-02","2022-10-03","2022-10-04","2022-10-05","2022-10-06","2022-10-07","2022-10-08","2022-10-09","2022-10-10","2022-10-11","2022-10-12","2022-10-13","2022-10-14","2022-10-15","2022-10-16","2022-10-17","2022-10-18","2022-10-19","2022-10-20","2022-10-21","2022-10-22","2022-10-23","2022-10-24","2022-10-25","2022-10-26","2022-10-27","2022-10-28","2022-10-29","2022-10-30","2022-10-31","2022-11-01","2022-11-02","2022-11-03","2022-11-04","2022-11-05","2022-11-06","2022-11-07","2022-11-08","2022-11-09","2022-11-10","2022-11-11","2022-11-12","2022-11-13","2022-11-14","2022-11-15","2022-11-16","2022-11-17","2022-11-18","2022-11-19","2022-11-20","2022-11-21","2022-11-22","2022-11-23","2022-11-24","2022-11-25","2022-11-26","2022-11-27","2022-11-28","2022-11-29","2022-11-30","2022-12-01","2022-12-02","2022-12-03","2022-12-04","2022-12-05","2022-12-06","2022-12-07","2022-12-08","2022-12-09","2022-12-10","2022-12-11","2022-12-12","2022-12-13","2022-12-14","2022-12-15","2022-12-16","2022-12-17","2022-12-18","2022-12-19","2022-12-20","2022-12-21","2022-12-22","2022-12-23","2022-12-24","2022-12-25","2022-12-26","2022-12-27","2022-12-28","2022-12-29","2022-12-30","2022-12-31","2023-01-01","2023-01-02","2023-01-03","2023-01-04","2023-01-05","2023-01-06","2023-01-07","2023-01-08","2023-01-09","2023-01-10","2023-01-11","2023-01-12","2023-01-13","2023-01-14","2023-01-15","2023-01-16","2023-01-17","2023-01-18","2023-01-19","2023-01-20","2023-01-21","2023-01-22","2023-01-23","2023-01-24","2023-01-25","2023-01-26","2023-01-27","2023-01-28","2023-01-29","2023-01-30","2023-01-31","2023-02-01","2023-02-02","2023-02-03","2023-02-04","2023-02-05","2023-02-06","2023-02-07","2023-02-08","2023-02-09","2023-02-10","2023-02-11","2023-02-12","2023-02-13","2023-02-14","2023-02-15","2023-02-16","2023-02-17","2023-02-18","2023-02-19","2023-02-20","2023-02-21","2023-02-22","2023-02-23","2023-02-24","2023-02-25","2023-02-26","2023-02-27","2023-02-28","2023-03-01","2023-03-02","2023-03-03","2023-03-04","2023-03-05","2023-03-06","2023-03-07","2023-03-08","2023-03-09","2023-03-10","2023-03-11","2023-03-12","2023-03-13","2023-03-14","2023-03-15","2023-03-16","2023-03-17","2023-03-18","2023-03-19","2023-03-20","2023-03-21","2023-03-22","2023-03-23","2023-03-24","2023-03-25","2023-03-26","2023-03-27","2023-03-28"],"y":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6037030732746805,0.5901728644661849,0.582048795371418,0.5208215713611499,0.477253203519841,0.45767309145762153,0.47308815944034743,0.42078690438586186,0.41878808108528254,0.43638311555274784,0.38987539557488704,0.38351397415660504,0.3639643138921899,0.3876755620411744,0.3927786137427396,0.3729662415384194,0.31300648885925564,0.2541581476115016,0.21588918115043354,0.16209394904654129,0.1485955015374294,0.11320145728652273,0.10744855654062391,0.08420674954193468,0.03144476081292416,0.02572807535118974,0.035546449071989124,0.08614546879265941,0.08389684620112384,0.0892230917972277,0.1042700262406921,0.1099664515836162,0.10457191560470291,0.12862867567427744,0.140237398338997,0.14127263489088857,0.11521130106171722,0.10949792624950834,0.17030510183724365,0.23383552353036197,0.2808166583113374,0.3130842456560241,0.3146564206107568,0.30177119945386943,0.3307867380250311,0.30240274796179323,0.33731835017847145,0.3037790340421845,0.21636557578161342,0.17277888241495645,0.10229766235504278,0.09588987035160418,0.06603568005784187,0.041100669107056564,0.05340789751585377,0.11789415659533486,0.13853560391327288,0.20700406585111675,0.23119125451429395,0.230083926621626,0.2555765081820128,0.23297513261672737,0.23767647224306526,0.26303972453606056,0.24374641941402325,0.19845914728074857,0.1499840789594562,0.06437753238026392,0.051240203314348386,0.06136017833752823,0.04449459576868977,0.053693852439888154,0.09044066661089256,0.06505136925979976,0.026666323085176045,0.05636129488592704,0.07284904542994332,0.07941092420995514,0.09567229872148839,0.07635123013459798,0.03865876572153251,0.03851444012889212,0.029740696320040165,0.08911539787044971,0.12496306931046978,0.1086665350332046,0.12390919127350923,0.1784505932300007,0.20638548626535216,0.22490717444099442,0.27736471769799753,0.2716637170356567,0.2945966300402326,0.2818064903995302,0.28801740803575715,0.2834355310819464,0.31009914647899656,0.2843517701733123,0.2693943379441771,0.2939565278139148,0.2874224319990048,0.29489590549345795,0.25118454421213343,0.22414121558369854,0.19488466826747147,0.20636961763922407,0.17912168332366749,0.1635603449179116,0.18301161090835819,0.1752123295331417,0.19228664324840247,0.20393945416850098,0.23068463526638738,0.2747378590606232,0.3170435718038057,0.2923288208290294,0.25415120678761655,0.23581250868053755,0.19932816923400448,0.18513836027492522,0.15510931607278106,0.12577404497925906,0.06683923534959232,-0.016730674224952946,-0.06115415243202397,-0.04975362403981935,-0.05012641373613806,-0.03380652456504513,-0.042372115212674775,-0.06938563059744747,-0.09493594094926029,-0.07155037291558901,-0.045623046612491114,-0.008482761001726766,-0.006607531104521838,-0.031842666078719536,0.019907445155768377,0.04711039570696568,0.08051176529649426,0.12056911811254693,0.1599989109700282,0.14545956348442649,0.1534240417355004,0.17058222662233574,0.18550922298775802,0.21874930767355108,0.19244168150859303,0.18353726219468794,0.15232014476441907,0.14684658090656674,0.14591751036594938,0.17067417030081206,0.18383697619214767,0.15716796590581855,0.14571478454729073,0.1450607327861892,0.17739600944169318,0.20487863705243833,0.22749771543986466,0.21889134463247756,0.20524558142858754,0.13630550272336536,0.1304877555624572,0.17607405487550926,0.17656776180145856,0.1514121522530896,0.10500410964807386,0.08573911557291695,0.05966048458284233,0.05781977644514173,0.07079018700684987,0.12023034897444196,0.1316734093159502,0.10301166682195972,0.1386101887147947,0.17579330444942168,0.19763024501253326,0.19388838755852125,0.22043301768753848,0.2147313496435149,0.23558896732127765,0.25126845890709226,0.28165102109836415,0.29487402773040794,0.2599197807890069,0.23178841845454298,0.22036435925140233,0.20247108485606002,0.1970259962122266,0.20497799479492204,0.17435285905542286,0.15542822186646288,0.13355684448287236,0.13796239171928354,0.1734686979443977,0.19789857580784448,0.24217157127726358,0.2787623301973151,0.28940107632383544,0.3059896309631932,0.3440233555283921,0.37188543101151744,0.40305691554293244,0.40369083300631325,0.363088941892228,0.38065167404145483,0.3992133651769049,0.448532806887291,0.49486487506177446,0.5388207716855168,0.5402642970761969,0.544677581955332,0.5172145695804131,0.518021473422539,0.5238829177611929,0.5221600208376239,0.5456111772839684,0.5221058038643148,0.516283396314877,0.4837409740102833,0.4973849724992914,0.4828914665837026,0.5072030745585195,0.5165200794411806,0.5451537446048821,0.5497631005429804,0.440583115459453,0.3929078082745804,0.32599453800242045,0.30131727425474425,0.27063997924775307,0.2743035549417253,0.22298574534245988,0.20780753036979557,0.20728642406740388,0.18791247597888147,0.2631045684534421,0.292897706165124,0.3311846336825802,0.3364592625138772,0.3301599845385653,0.3121806890570878,0.36295750299016466,0.33757250881156786,0.29309078581039644,0.24612290544150559,0.18602461727030528,0.16994608013978646,0.1521869594283034,0.11464177851228852,0.09924144391617691,0.051291808909804046,-0.01973391036167171,-0.02122752197066249,-0.009865790743082033,0.033546131416352,0.03576432440508627,0.0017820194324988838,-0.04159458801093916,-0.039215013875477524,-0.053245525232829535,-0.03973915389515671,-0.04012489831202468,-0.0327488969682738,-0.050166884915076725,-0.09462867176912289,-0.10575125515123492,-0.13029679428101107,-0.13245875618444752,-0.1442785140064754,-0.1647249962576654,-0.19587810121490618,-0.20737423179336129,-0.2341923577132258,-0.23048468074723316,-0.21234431967199147,-0.1858293704526266,-0.1663562137708225,-0.1262856661659501,-0.10091557794947906,-0.06370962929601917,0.0024685903912054774,0.04475489863510713,0.08317574472446392,0.07329605107433326,0.08303389196234459,0.1015288773307537,0.10935418048333276,0.0639056626514117,0.040307103323334634,0.002917747759672107,-0.049961692864839834,-0.07689908095284845,-0.1002415939825269,-0.1050348824398569,-0.095566040064244,-0.06922219205224008,-0.010262835840544115,0.049831120072520885,0.10143673486502527,0.1344719267526932,0.1674475019080044,0.16555416222031666,0.1400297275865611,0.1368607764908628,0.09378693549098074,0.04726019365007443,-0.0001120933577264005,-0.014958890712676063,-0.0841030605723149,-0.13137017669760073,-0.16555173451245625,-0.1899855789549533,-0.17663869219705744,-0.12781050604643732,-0.08572963878184225,-0.09019206682884164,-0.08699650471931115,-0.007548410448721598,0.08469856278118036,0.12668356650089022,0.18957613623733988,0.26040348334969493,0.29505281456040794,0.2723249616300769,0.246593179537903,0.24977613442414168,0.27833284007868847,0.24931082992797088,0.22112845974824577,0.2398207636227238,0.22716827047243088,0.23393462806286092,0.2996783593663284,0.3336192528830275,0.3692979547258892,0.42001464991291926,0.42690811345854957,0.4312935851812908,0.44964335048586124,0.48392735888790783,0.4923799020946783,0.5003612342550663,0.4806613628363177,0.4987584010138658,0.5278248966622148,0.5356661386822923,0.5528232344039801,0.5513915129409471,0.5528865242996928,0.5115924931269469,0.5309131935138167,0.5317154303987051,0.5113437755983978,0.494497426415981,0.47880683241455363,0.4593074069634011,0.4525472908045257,0.4555488561339403,0.4392205397948873,0.4625272672144797,0.4194402805637433,0.3777935851797838,0.3599747275705037,0.3552540382013246,0.3469745371464279,0.30233257117750834,0.2549379920021932,0.21570616882330337,0.21148687217237888,0.22147238046668746,0.24028653089714863,0.24581378044654983,0.232963951295027,0.223875065921422,0.21294064922037012,0.24410935579227405,0.22629442702053212,0.19038213372930218,0.1692707803307381,0.12365811676996985,0.09430715877582774,0.06861876854593532,0.02792844471597532,0.007852886565706041,0.014948060162934493,0.0009434527983628622,0.023327848456092832,0.07847948810164815,0.14713714515940687,0.1512424773998279,0.1592177788537586,0.21105716619314263,0.24359459264301617,0.2482583237238493,0.2492046777270424,0.20047922361537815,0.1622480433020334,0.13654615743749612,0.02386268775104028,0.0007076668484125387,0.018185401637775447,-0.05153831494337158,-0.09284823876833843,-0.13645844652534384,-0.12614002290182078,-0.09991311527855226,-0.04340952973807608,-0.09344432120296267,-0.10025989855994093,-0.106674619387736,-0.12921591577395686,-0.12251630290735642,-0.07762338647601912,-0.029624934651677554,-0.11178557896324559,-0.06532687292056641,-0.06726450081688029,-0.008380128119177089,0.09598649839193307,0.1741056448684788,0.21962308179014012,0.2874170536961031,0.32443121763472865,0.3570642635868147,0.4569693295874574,0.4172603405082663,0.42104250181832664,0.3875568014041664,0.35339721769300514,0.3075309515946543,0.3085404238893553,0.3494791286850035,0.3411329285347871,0.3755972899554398,0.3622341671657991,0.4108670111852355,0.43412330679251376,0.5327880602138462,0.5796247395093751,0.6387450847179845,0.6572327499582981,0.669332990970332,0.6965382125530403,0.7280760602353027,0.7494694224620069,0.7758618146279207,0.744125130153005,0.728018758656742,0.674668590114693,0.6417123570803746,0.6046997025943199,0.6075223654190581,0.5811350866281187,0.5299456295326292,0.4634678741523124,0.4326699594209325,0.44733915287489073,0.344482484102064,0.3076495230877317,0.302189067518611,0.25971809869035906,0.2082863112286196,0.20371930754530077,0.20806378135307796,0.22025377054123912,0.23406616457833113,0.23264319774103534,0.29945340017741867,0.3297031160011108,0.26732322515987733,0.3340820945175341,0.32650090601642545,0.3555811259952279,0.39794246408904865,0.4306251514953162,0.39475506525458254,0.3660044034081687,0.34377418795356685,0.35721874882835947,0.4403498623974798,0.44914347433264074,0.48404218089092,0.4559943492462811,0.3255325540666754,0.30493183448093464,0.2864142595831231,0.27710521727652193,0.2837755840658036,0.28457466036304757,0.19669389650110616,0.1435981432893916,0.08053546093153045,0.01805266727328364,0.027297276587964936,0.03325830265714651,0.06320063000559559,0.11107436092704459,0.167135358642367,0.2141993685405541,0.2817531098865168,0.3283748069233893,0.31822459191816466,0.36756410997965133,0.43765240273890205,0.4529725490248947,0.4822673178088518,0.4977184907852905,0.45092843288039475,0.40083400370640776,0.3796627346898197,0.32640169346320363,0.40586662652442274,0.4293103823177196,0.4202121156410447,0.3777718847282532,0.3520469683481811,0.30752728760058745,0.31959688393686253,0.29746404502905616,0.31474506972934174,0.3570199867816149,0.3396928146542112,0.3566268991863346,0.3694925070124694,0.4329231750661706,0.45242524430640174,0.46138319481593576,0.48787984656085903,0.5574188637763118,0.6184132989426471,0.6233743161841934,0.6465855383138904,0.6604215477159943,0.6791893233154422,0.6783524983483906,0.6449857797790302,0.6210940573372713,0.552517350011039,0.5254122007886439,0.4784950210104168,0.517172318520018,0.506074467723813,0.49761872892100667,0.48081300277364714,0.4879419944300413,0.5408457105130342,0.6101805189759271,0.6387725147217651,0.6053472667334752,0.6222661224526551,0.5713689169409882,0.5507998068046899,0.5136434511614415,0.4314969647776721,0.3438108433476258,0.2676265810420383,0.2316442674090608,0.1759089155276268,0.1316154847998751,0.07159243409091376,0.04633653573960268,-0.026469808827014628,-0.04869092385844048,-0.043567486009895316,-0.004346812340440482,0.00968139699243962,0.052433142506275586,0.14081818828251297,0.2054076719332461,0.24482605645071193,0.2585637294063904,0.3185684527793664,0.3286982749442612,0.3662743498125569,0.3646942006712555,0.31870840815752105,0.19227731414788365,0.14751453231047346,0.16622770213415167,0.12396593305924578,0.14334095832100346,0.13458785907241724,0.11363972655798663,0.11799685138735065,0.1524286960552948,0.25051228809203013,0.33095447310208226,0.34074727133672267,0.32341623813091935,0.3477012192120684,0.3274714878924649,0.3262941222639615,0.3680833600900123,0.33710950542458173,0.27521177599422386,0.20606736737479667,0.20001891177627013,0.1854470471579949,0.15081746768389243,0.16956723885766242,0.17339112389624858,0.1832044806416063,0.18394780021324597,0.23149877596370422,0.23771258718815322,0.25782399280248536,0.2670420214820251,0.254725308051296,0.3474054419657965,0.317761149369359,0.30171494727029924,0.3230894153984787,0.3053608898755872,0.26317499808720307,0.29505802889108507,0.3482459321874674,0.36091426070013216,0.40608138082806483,0.3658092186118235,0.39996637263520746,0.45403336627112995,0.4743458347135017,0.5351526518456386,0.6078196391054975,0.6300828676244036,0.6300828676244036,0.6121035565803441,0.5938514188846966,0.58520982271771,0.6500116793467581,0.6433926992212994,0.582692576236625,0.5672775082538991,0.5122728745619719,0.5039765957156089,0.5109635637116576,0.5226499977255886,0.5706586800144022,0.5771604840296333,0.5838770485612528,0.5725834521870985,0.5869346803493188,0.6201480638515804,0.6991791567529396,0.7203966087473033,0.7192936907535479,0.8179004517208794,0.8297833249917488,0.8528581400616917,0.800151763280034,0.8559201052984708,0.9167553995468305,0.8887605913616262,0.8960078562429251,0.8831144879053359,0.8597950991885647,0.809855406394328,0.7920570908747923,0.7550472944477049,0.8108811139790685,0.7596201394231193,0.6876649159115937,0.6544161440791209,0.6220374363697311,0.6674730447507833,0.7107945468568899,0.6812854510598851,0.6936770444523932,0.7095261430109012,0.6573964506475726,0.6991264072230894,0.7368468508553462,0.8067480808565806,0.8372862458116987,0.7694618683059617,0.6742609868583382,0.7103658794345227,0.6842644968975609,0.7678406256734396,0.7783766772392222,0.7795462812155414,0.7515562880145688,0.6840423016339825,0.614937888597666,0.5906216828493103,0.6717147044709433,0.6468157040520082,0.6835367152594924,0.5788579931791401,0.625074038357611,0.5870188981317315,0.6315303321799364,0.6472360376933837,0.6795718974409666,0.7129772830166597,0.6895455978254262,0.7087496140926874,0.6237869990780079,0.642762217591854,0.6064895621122017,0.5433494871626132,0.5102929472856317,0.517698413868523,0.5277028350004233,0.6098573714289021,0.6025486671269407,0.6148251514508314,0.704206939053041,0.7504002105122853,0.7591344115278001,0.7524078525671413,0.799484405971591,0.807998036291111,0.7760388882787275,0.7262648498569939,0.7015551706233236,0.6304734734857311,0.5614984791245861,0.5081217598237464,0.5057970748538716,0.5720027342028631,0.5728095516678141,0.5534745628842708,0.591190696078802,0.6155413206621115,0.6318837523233702,0.7168692275834683,0.7127038048530699,0.7247691052266981,0.7118563421539728,0.6909406112200257,0.6077096042778545,0.5894774485984591,0.5447954192194512,0.4814470971148654,0.464445988325784,0.33040243807239644,0.3181099945567162,0.28372154822927986,0.1991234277790342,0.20579897997916366,0.20924820660895146,0.1692937226013227,0.19628974730348725,0.2184260168901601,0.2698310312106463,0.3626799071891578,0.3748947127222658,0.3839249785296361,0.4008441051263644,0.36768508422397395,0.3781765218613488,0.39905420156242943,0.36074791019775426,0.2924823849022834,0.22770031194012766,0.16171960397134255,0.17556908385178865,0.17645223728851306,0.20408969689812836,0.2452796429934053,0.24714211604285757,0.2932464644416215,0.2772558750856225,0.3215911191801518,0.3387521093000545,0.3762324581285779,0.3710986425359659,0.3687223629119207,0.367202941437829,0.33903504106847765,0.33845540079477465,0.30711119055460795,0.33327250860275603,0.32418383961009745,0.33215215855250874,0.31435859471800986,0.26034132771581997,0.23691109779574576,0.1968395540217826,0.2139213272530296,0.17416393704230124,0.12204169652144412,0.09423610315144347,0.08653436579761498,0.05868096098068716,0.0455146913509271,0.10587591143679684,0.14556531124742583,0.18537507476718112,0.1598249928129303,0.1686630807571588,0.20309112266406126,0.21218263171107155,0.193050352712066,0.15476810717689984,0.1489026893939862,0.09767651448536047,0.06465789235171551,0.044044191219712384,0.0730959852437548,0.11389835684329379,0.15706323188450882,0.1646084352810026,0.18304516181475838,0.2550785552922349,0.27694651325354747,0.3135095034208501,0.29241319014511624,0.3269314794868306,0.36120458936067584,0.4170817443416901,0.39474428829768027,0.42029836198684534,0.4257795165698585,0.4155239549662763,0.4141495951562062,0.40939833800144176,0.4549749226597381,0.41698685110901634,0.3332358657069122,0.24861195580301612,0.2813806965095641,0.2686909550582569,0.27621329718201565,0.24653745671141786,0.19023182698852134,0.17686197789229308,0.18047036546469525,0.22922552070893348,0.3146006717439008,0.3568762202867126,0.3294873452547681,0.33314023740352516,0.3477146224689278,0.38854356235592047,0.4164749446340469,0.4566412268498895,0.49953076284154446,0.543252143483819,0.4578057420005222,0.4638812462428297,0.45072361046395776,0.40494030110140977,0.39378505251083495,0.4047259629292176,0.41588121151979235,0.4643587061763972,0.43598188886333267,0.3484350151279427,0.3506869014814777,0.30149196183177185,0.2940811646163996,0.3765225212717612,0.4304221713450299,0.5012257506503995,0.43999297326857584,0.2957545904914824,0.2354470547497615,0.22983810808465713,0.26463007033081354,0.2412211581309108,0.2616372681711028,0.2153749159762915,0.1342818943546586,0.07565349085290021,0.11659604178532634,0.13537971553353284,0.15895591706087492,0.13018770981569688,0.10540409342523874,0.10684527129136907,0.09871897488349246,0.13092732480040378,0.19080289442767312,0.21239391909974206,0.2571963913524381,0.3425819174457744,0.30284381847495384,0.33468919158680727,0.38577175396340635,0.412943576109683,0.44015533170075,0.41972528534947695,0.41758125222569004,0.39294121087784234,0.42170941812302043,0.41100902798286476,0.4632222974509331,0.4855610696255714,0.4963824280895947,0.558286348930217,0.5015095691290543,0.44952202320046347,0.47120279648697583,0.3821805059765521,0.3374790948744484,0.3677361633690016,0.36703929643739225,0.382169869206895,0.3690362690007863,0.3757054064506534,0.44346970585305157,0.48855796463294715,0.4819156110496129,0.5964363172770236,0.5607004468121103,0.5238783533185309,0.5309794568828499,0.5263368471953946,0.5074158254164807,0.4316968707277766,0.4291650899293476,0.4382622677499203,0.42409721604361766,0.3404066457816587,0.4282547245640478,0.4282547245640479,0.40841377156414643,0.39888275358371394,0.37368545629160355,0.4452886037309569,0.40474209292014046,0.40826004363225776,0.4125160050741373,0.4033352501488251,0.33267219705910184,0.2953005560797434,0.29608011827798897,0.27768783446189615,0.3156442724294969,0.3389165912465992,0.3685431728609164,0.3915955387220997,0.32228082066610514,0.329976924779718,0.39093022883159917,0.43216878392687297,0.4369697058455091,0.4474234867361454,0.3961139095060924,0.29005361323597983,0.32328789756390003,0.30097354243247904,0.3389225046029694,0.3791837085155452,0.40599490614131356,0.37836956847849773,0.30513277910717507,0.30482365985020776,0.32042408469786593,0.37962919106672355,0.3633772981169461,0.4144598604935451,0.40333729698252274,0.49132960575285073,0.5004645836116735,0.5171699920779901,0.5908522164842508,0.6030219099820029,0.6555933336079769,0.7909792421446508,0.7156020619070127,0.711346100465133,0.7538344198616597,0.6702096174415978,0.567247675723482,0.5710217085217667,0.5784325057371389,0.5565635856406559,0.5753688087909499,0.5049491071162968,0.5637277736065087,0.508847286423388,0.44643185551608866,0.5014902671065962,0.6847484134814271,0.6604537956203882,0.7148153402792864,0.7069681787351368,0.6781618689937843,0.6177948978862722,0.5677173690950231,0.5736659144236411,0.5705406600732308,0.5044008118486942,0.3359846659856905,0.36185411954354335,0.26159719557873007,0.24431291429478902,0.2864723632985939,0.24831097406999542,0.3407699275388804,0.2958749055340902,0.30880628120212206,0.3678450258623397,0.341224721784874,0.35018593765384276,0.32299256610547855,0.35483793921733203,0.28397280250774093,0.2929340183767096,0.21118952865295734,0.25173603946377376,0.2880934561036742,0.17440034080664957,0.16794648869289247,0.0873839722942289,0.13135939045290856,0.09951401734105511,0.1630179723163216,0.15813895589937838,0.1411331096131887,0.1933852759661069,0.17779179580403096,0.21574075797452127,0.2308957477872414,0.2267284781471846,0.23283058877975388,0.22482631801240024,0.16062549610552446,0.16513753963357142,0.20102700970127593,0.21119027425565706,0.283582158178327,0.26747022326399184,0.32253493505925807,0.4573422498892274,0.5836235803241143,0.5962798526549572,0.6700397469680351,0.7088913019474743,0.6543401837720861,0.5712233359263542,0.5306768251155377,0.5473822335818543,0.44466990741145285,0.327719513299024,0.15536118753425804,0.14693515317248407,0.01375929760704615,-0.04636167091637983,-0.04883093217541697,-0.06273368400072073,-0.08349762047854516,-0.1336232377535292,-0.056662031064969406,-0.05997725179665945,-0.014122232140611734,-0.05515067159502307,-0.0036656941884514473,0.022143076581571643,-0.0061361321341872775,-0.021609492161699845,-0.07424247319189545,-0.0924746288712909,-0.10329598733531417,-0.06055158585262023,-0.004991086360310282,0.04671067304108089,0.04709090163605474,0.07168672454409283,0.13691924314806186,0.23988118486617765,0.2439633843182032,0.29096374724277674,0.2966795886267716,0.30440646670585275,0.23614676973937349,0.2734307578185822,0.3119970058997807,0.3152405334750961,0.3480909401722997,0.2897998051184053,0.1963688813807219,0.2944518066818945,0.367767058199259,0.3481796058793932,0.36382999325619453,0.38026029838532216,0.33458445813575066,0.26667821406793,0.244363858936509,0.26983765060975173,0.43028434058095966,0.3798096828119989,0.3421619256885077,0.34022762940419454,0.3761512058842352,0.27404324032093286,0.2889964137180292,0.2765666420502715,0.2892418126141859,0.21549463205104566,0.0765665991212004,0.01894998573329037,-0.018236930134674527,-0.008975413739728854,-0.004352479408953014,0.0332952777145382,0.03762138998356683,0.058752299350287515,0.03489798818756144,0.07751154241573463,0.1956861613068665,0.3055473901736775,0.34518626775907424,0.43286480708293895,0.4194484926335981,0.48921968360480417,0.5014434222053805,0.5641623434821953,0.5884443103216473,0.5832589741606219,0.5433479625380431,0.4816885405967274,0.4447525302501226,0.36898396008037093,0.38757538103495626,0.35749057787154775,0.3600100122288227,0.3508429145876441,0.32741478501517757,0.3254345222855596,0.3225617933218686,0.3185432616242772,0.33105957591967783,0.33981237985590484,0.2366320179563049,0.1918310423623183,0.15240186161131436,0.10445898464078252,0.13842316577903171,0.14625635010602242,0.13662194919753756,0.1839507196422301,0.172481535333559,0.21003222739029628,0.3833241243108084,0.40212934746110235,0.4203615031404978,0.4612888814443168,0.4781965144487102,0.4691798843570192,0.5323901595037238,0.4936308611695196,0.48511137656319636,0.41675360360881697,0.30807582050670623,0.3575381854256773,0.3974981842912539,0.38381226601953417,0.34551304079392364,0.3965956031705226,0.4119330012464313,0.39150295489515824,0.4354396208735428,0.5276384961724222,0.49952175702936347,0.5030762507713046,0.4854306070371489,0.5273016405229674,0.512538040642361,0.48261025398104457,0.5776160134155418,0.608364483390338,0.6621869431857484,0.7625171540721268,0.8514529390235837,0.8553750103389117,0.9159885906959435,0.9211179201346983,0.9579288522911378,0.9758607074445566,0.9130599196056579,0.9781964842601572,0.9399531613932183,0.8190253757376554,0.796772578010456,0.7341569103929471,0.7023115372810936,0.6467169313764477,0.6867359895139371,0.659060689321978,0.6136689401725669,0.5560831307688824,0.5104935345953767,0.4457534187182026,0.42831808000372484,0.4201502769022981,0.4525289846116879,0.4597221538848005,0.47123712615936275,0.43869488611589996,0.49209090603266475,0.4551162034820562,0.40537594313867115,0.38955354261718184,0.4292416790588591,0.45760442078600405,0.37414315070001536,0.412008535765766,0.3702725461897347,0.330968287378774,0.19060398743332357,0.16414802193023095,0.2278644765111318,0.23656561421009478,0.2479600401450166,0.26806641400538606,0.25914330063259183,0.19935960055702978,0.11514132458498647,0.12657424314904517,0.08783978644954818,0.09383174397767584,0.0826726374368296,0.13863421623037187,0.12640033101114742,0.1786901515111734,0.23869582726056676,0.21664849498473368,0.35375118388364385,0.41591699924322895,0.5517293476585484,0.5950908027960137,0.6526028002431525,0.6117811860951471,0.5948014290342554,0.5098213299737232,0.49869876646270067,0.5419635507328974,0.44320817554333286,0.4811271721739695,0.43004460979737047,0.46082727365179527,0.4902923478285827,0.5052277985849593,0.5165722619907378,0.565407538782131,0.44967896974680277,0.45947701058282303,0.4866703821311873,0.40110377112541523,0.49681504376485625,0.5164361106603185,0.458487324451576,0.4621990907472263,0.4800682696215638,0.5092488485851282,0.6738294042456257,0.7687374597153402,0.8608846687769555,0.9470616580765293,0.9202010655374326,0.89745474429156,0.9148570196894861,0.92132352282483,0.9144242356761347,0.9153022044413394,0.9105969500142503,0.8832336548905901,0.8197270206381925,0.7805503942631922,0.7386660708577582,0.7692042358128763,0.7953337675407349,0.7859685107180657,0.7705534427353399,0.7546027548325616,0.7253526189437887,0.7073400023207366,0.6868409666600699,0.7630549718647597,0.8618926177294706,0.8496555977631417,0.7791574339424467,0.7667612397816096,0.7451389289346461,0.78717800476713,0.8174966306659046,0.7990857439290995,0.8233077137666018,0.7633456015116897,0.6793868293155662,0.6642109386771873,0.670029134215461,0.6793471439255565,0.6960904383555768,0.6658296660887311,0.5890490871304738,0.5456767124590433,0.5132915888614286,0.4728716603459951,0.4358150435890912,0.3362801317592631,0.2844768104428178,0.2598483024552274,0.21772695594759703,0.16557979306572396,0.15625847871756862,0.16627736085901174,0.14396240058570012,0.14528367995198083,0.11313708570635614,0.16416974786869737,0.20653017268356305,0.18210478653186696,0.13336590870046983,0.09852788998720262,0.04669541180808735,0.0005389403634804251,-0.0013281412478285044,-0.004124337943220935,0.019151443243784594,-0.004753594415581238,-0.00279736124842192,0.05541629693608478,0.12728383090967335,0.15658410691847374,0.20277933365896775,0.22212846281502677,0.17659322246437797,0.14784684530842612,0.1348768437741717,0.12693877651369348,0.11760455246279558,0.0837384809029356,0.029804515011904658,-0.006859297158882592,-0.00511409445492971,0.0011084839777252692,0.05229983592812899,0.08484040920534949,0.08379220683695805,0.09866596079799687,0.08419504780090717,0.08342772359122183,0.09449461886647546,0.11545891989689767,0.10742746726423816,0.041108415657578105,-0.011289596845994504,-0.04222038915852209,-0.009524776873403112,-0.017446427326781894,-0.02583303877055132,-0.058989512464030766,-0.05381879927888139,-0.05186770260851061,-0.07436202856373349,-0.05193347158240771,-0.054336176263452475,-0.042993867318625645,-0.07882764756769692,-0.07957599331640425,-0.09829002938875765,-0.1162190618891253,-0.12222770406911805,-0.12752097891211084,-0.1062360416691793,-0.04362061097467282,-0.01305498330422993,-0.01446041402088671,0.013714848598729068,0.023172936752387567,0.06514904502701555,0.11990104276691267,0.10562272232713596,0.11299672755479133,0.1259410958976664,0.10060157280076758,0.12084794545923638,0.14685213233158462,0.12877419800846152,0.12460258006334927,0.0845010882454723,0.08050418053143779,0.15214027578163508,0.20412514959793127,0.25511835666090243,0.2869282829635037,0.29070232154605663,0.2586243160875547,0.2191071112736863,0.19964536386699677,0.243586354157386,0.24939960817914136,0.2070289520930248,0.13101911702105523,0.07400862584538956,0.055296170663054055,0.059956973640445076,0.10328104671114824,0.1510550440995306,0.20158335509633404,0.20885177241661784,0.2182270971129796,0.23680974514928282,0.2819211176694067,0.3084580733127073,0.31788235795277386,0.307475454325561,0.2899280059619948,0.29120928460300455,0.27370112435690364,0.2398578477050059,0.23212303416074684,0.2282896929743913,0.21279791767389605,0.1483508229813617,0.12358955833347059,0.08127084066480364,0.08608937619280209,0.0504613617166885,0.03311422346446423,0.042822416362814544,0.056427863443616555,0.05598703005736274,0.0691486482086586,0.11694740139188949,0.13155148121625682,0.1773030059626965,0.17944285334186352,0.18495581058886826,0.1846797004289316,0.16262499855759383,0.09324916244714651,0.052042830883527944,0.03225141600235822,0.017483808321114326,-0.026233669848441287,-0.048482294812770006,-0.06170638990984221,-0.05183867459458833,-0.04823520864014173,-0.020197841214277184,0.03150254604708266,0.017381624510796938,0.005914005870789536,-0.012316901977727005,0.011154215253515825,-0.021935134957686505,-0.07079271158649965,-0.0949674777430527,-0.1130372602643337,-0.13124630825417555,-0.1375489131689705,-0.1176083371318108,-0.12383196905032615,-0.09627830579357509,-0.08884208387571282,-0.04605045422058895,0.018738136274788052,0.06030452428936989,0.08040638661375933,0.09652179175839126,0.1010051713282218,0.11272618269199894,0.1022816827193963,0.09935711198015788,0.06808997184138296,0.06546907852506509,0.022213193728260406,0.009392754343944296,-0.011986445624512343,-0.05386327007857483,-0.10169103885782252,-0.12837809991293037,-0.16270731225017718,-0.197504551443473,-0.18683258065828656,-0.22121585360431562,-0.2375529552206394,-0.2650222104513229,-0.2611431741218807,-0.23790746367047602,-0.21754799875609318,-0.2272609946169793,-0.21418771496625127,-0.2157424866806275,-0.21716120011310086,-0.19783924866201502,-0.15152914637783715,-0.11588874426988441,-0.12811254725468524,-0.14258677090477526,-0.13353816725076653,-0.10898122703375146,-0.10597619731849525,-0.10047072559914963,-0.09221493666198116,-0.10809183419966421,-0.14295374721875126,-0.1811908077585806,-0.14466517181386865,-0.12880103926315545,-0.13308880082811686,-0.13874024373275268,-0.12984067547385963,-0.12575417040946776,-0.1561485641658748,-0.16504632833677346,-0.17661741277657222,-0.15659513538702668,-0.17378660776128346,-0.18597242973695796,-0.19361075982228934,-0.20132233424968415,-0.18725552799110712,-0.16009548547334096,-0.11484339595986767,-0.0762774298335399,-0.04558493520572986,-0.029449698310201288,-0.023512315274030006,-0.012660466902383002,0.006643125446122338,0.05998752536143652,0.06613397223989767,0.06815611010947983,0.09257738663893027,0.12044697080312883,0.16092109459927226,0.17981806217580265,0.1941369585906437,0.24269900069343486,0.2493274363050614,0.2181951072750255,0.22626226621962173,0.19975370255151814,0.14752920280661178,0.13710532175174917,0.12114800904555625,0.1557482111649096,0.20571173938394144,0.2197345068788584,0.25205310509581674,0.2874616493127245,0.32051286776161236,0.36420917661027613,0.3970030896119058,0.39932220746812125,0.41396804696012407,0.3763055616237922,0.30253827860257887,0.25081861504572994,0.23164781981950924,0.23572875233291643,0.2426941760281715,0.24515932816011093,0.25214390017619515,0.2656101452478886,0.30431213375724225,0.32802206556291325,0.35977673616785955,0.37859958233359997,0.3812689662299188,0.36764024693284636,0.3341252530165146,0.31654616634123245,0.3250897465238641,0.3180396073625681,0.28731875006229607,0.26230431614802063,0.276062150419076,0.2798354759834183,0.2815717064518902,0.27084984204979196,0.2768564776935428,0.3021831377627425,0.305308664956634,0.32098301187695333,0.31829963065705946,0.3360521824883427,0.34173252983864055,0.37376748171374774,0.40133797835665047,0.43525516736776976,0.450317652410759,0.4301115706536012,0.3757369947001319,0.37125096918857764,0.3687565500893028,0.3476789849433234,0.3252966273535492,0.25034499604083427,0.20978720392003364,0.18762224007928957,0.1680206771415636,0.18315403138952183,0.2429446763871364,0.20820841212678842,0.14813940549257396,0.10696514630326133,0.06103892651117151,0.0845675332657522,0.06265128635601942,0.0016443683752451698,-0.02570167252330276,-0.05384929266418905,-0.0835219200081885,-0.06851803282829536,-0.008383935179393714,0.041638435243685705,0.13138768479877427,0.1613993423225172,0.22056890659500752,0.2832960025395893,0.32412227936226,0.3596985166064067,0.39264129656356284,0.4017754494664699,0.3816455461306495,0.36920818832465924,0.2858471475698451,0.2687066846112066,0.2551193104387585,0.24179200762217085,0.2373938555289102,0.21622481008560063,0.18462534124128532,0.1619006980032271,0.17227790602899157,0.17923761717915182,0.2078176725244188,0.1979109548161641,0.1751436212089707,0.17686292377700313,0.17765109594413184,0.1604515587018726,0.16529229697987552,0.19037504390472754,0.20084677069590193,0.1748197461647647,0.18182430950065753,0.21452755291787282,0.2233288946790241,0.24563460970751735,0.24429276347943749,0.24506638042118628,0.26278871588798924,0.24202948090424123,0.2407317433993,0.29408234026428665,0.2881530053377392,0.261433448674483,0.25489031105140597,0.2385065608040279,0.2533606756149111,0.2693331856654045,0.28843556868305614,0.28507182736331427,0.26259287431665607,0.2512532366977922,0.2467215573519505,0.24122132319213244,0.23535473687087222,0.23473073955252785,0.22739842886897949,0.21259220616274419,0.18141724651919627,0.16652052890236888,0.17586145097066444,0.1468989727098769,0.15501953978536354,0.1509517443664922,0.16824700433082418,0.10177794038098313,0.05760666088587437,0.010737617890151763,-0.016587114599280743,-0.019111143012127435,-0.027554438125012803,-0.0502404145408476,-0.0965478858907837,-0.12701958580030734,-0.18164566461749238,-0.1486906513272704,-0.15197195096443902,-0.09772510982724146,-0.06964800380025553,-0.05136546507512376,-0.045444577067978144,-0.04260823935011297,-0.02014617119340932,0.003500853584381608,0.0604776184186571,0.0886945393174973,0.12852474441788544,0.12122869210003943,0.12518227044938268,0.10937674546214707,0.08790274779754391,0.10153034433067354,0.12452170769052276,0.11377574492960849,0.11224331412222306,0.13109047927055598,0.12516212240865515,0.1332927744671871,0.0786634447478702,0.0605157241362619,0.03929543041798944,0.011019916918089723,-0.02695989921297236,-0.026636888115707573,-0.03080739566628147,-0.05004373077700802,-0.062114582815615126,-0.06770169176152174,-0.037566288575775895,-0.031058259419179425,-0.02823017357455383,-0.017299459162087506,0.005649967043940241,-0.0278133552494531,-0.08125224707259653,-0.14016201864301422,-0.19743101500207666,-0.22516404346056515,-0.22389822217102875,-0.22243518093378684,-0.2039610894269502,-0.18442209490464717,-0.20004497310118027,-0.17972761506436424,-0.14937218035188,-0.0847173450335901,-0.021829971986224033,0.02904133972822497,0.059144026297722106,0.08548338836989773,0.11610824635066903,0.13083010071670478,0.15962279948459873,0.1813636780646825,0.1799445952968042,0.12158373995590169,0.09321537386990161,0.05098476762904688,0.011326649730909138,0.011614824185461068,-0.025205976685622633,-0.06519069111741047,-0.12318951252034423,-0.16157618125603762,-0.16900027738877665,-0.16881923402214316,-0.15281127546054038,-0.16071398947371923,-0.14663387339725947,-0.1402868062451857,-0.1352509159485533,-0.10961968931002157,-0.05221624708916635,-0.020240119092591297,-0.033943421964622356,-0.015518359206320036,-0.028950581114254092,-0.021563711107461296,-0.029358004260760785,-0.04065525082677377,-0.01872757278686754,-0.005261451225227981,-0.010982449929402971,-0.008034621824350751,-0.026347901837916698,-0.008900187087424558,-0.002685893065126438,-0.003602729873814388,-0.0001740110925946975,-0.0016372601366761482,0.0036530399718523233,0.000982211102727132,0.008791577268188303,0.04020349804498252,0.11254169566099895,0.14456698175905158,0.1763132550028979,0.1951184781531919,0.22082284023431015,0.24100166317733943,0.23154440493957104,0.22723899575999398,0.22913777131933152,0.21313735140995674,0.20741920867869998,0.1710774299227325,0.15730302680010358,0.13670976112202476,0.0730828519905286,0.02455390082648276,0.01249729876421401,-0.00759493056753596,-0.04065246545740699,-0.07669074694305192,-0.114583811866998,-0.10445798588798483,-0.10079713170804641,-0.09118934290658928,-0.056226080197434455,-0.034491172529258166,-0.055867242354853464,-0.019151930270623462,0.01996120955112447,0.07370635249857353,0.09225522037349884,0.08318317660889045,0.0735878575516259,0.04354688047423105,0.0400267257015137,0.024350538374831348,0.06550482972413453,0.024984352611020052,-0.04246343172557282,-0.12126252107824506,-0.1508446685287027,-0.17172702304216578,-0.18688593022084948,-0.16252536123923159,-0.1938790422682049,-0.18973234191177618,-0.22954723408624123,-0.24881700358008846,-0.2467584911806934,-0.32582940889374323,-0.424546967839062,-0.5340721891824927,-0.6504448614107832,-0.7588908722828269,-0.838531811658909,-0.9270474243939402,-0.9932930575158687,-1.0424291597029376,-1.0728133101268686,-0.9877958416004212,-0.9021309513260503,-0.7790628561913637,-0.7055331022289528,-0.613308798664761,-0.5396640207524179,-0.4864904863900407,-0.46791979119520277,-0.4262253773637218,-0.36368932086870787,-0.3256109781292816,-0.263324115542955,-0.25945255912899806,-0.20270736364738906,-0.1792279350299057,-0.15127671034205445,-0.07961024662039028,-0.03519648010158905,-0.013771319162704724,-0.03446961610246294,-0.060202709129513844,-0.12067049480153695,-0.15850086708726546,-0.2148966972465592,-0.24781622292438232,-0.2624600614636599,-0.3068877076709021,-0.30584004504981943,-0.33266984228722557,-0.34109382197259885,-0.32568084531818026,-0.2886803396452508,-0.26258070775316583,-0.19431876715296298,-0.13037553678897923,-0.10113721084758749,-0.08916808509905375,-0.06660537883533958,-0.027160453065243416,0.009852490225276435,0.009283575505685534,-0.003939723019872306,0.005040733905492113,-0.032116093275219096,-0.058835766548579624,-0.048229575408006244,-0.047209874617783656,-0.05719149293505984,-0.03976984053499695,-0.018456648945000643,-0.013056281694565524,-0.027079614236978466,-0.0420358878690109,-0.038866793982567475,-0.04655136153273886,-0.0490210346438001,-0.02864862793779105,-0.02072757379252358,-0.025292698996204455,-0.030380691590401243,-0.006907361662865366,0.009377345021750506,0.05745288558111609,0.06746458283763575,0.08435084390747016,0.07029952538640807,0.04848570381259812,0.058993355665804756,0.06922942121074938,0.07779284674363364,0.08303246017481822,0.11997321395093774,0.11871336656940538,0.1528226882914082,0.1650066804166519,0.17887019169178675,0.18862191276930926,0.1797691450185499,0.17106257132569563,0.1708171993910459,0.15524398913057436,0.14226345938338641,0.14696244615885837,0.1625172749370738,0.1586980737757048,0.15959729169163378,0.14900032762096255,0.1390491736455764,0.1196629575196119,0.12288921793055359,0.07972109603016227,0.03809081683838551,-0.021060837999612468,-0.09289739896582908,-0.11893569487763025,-0.11558837390614111,-0.08642173488597177,-0.08303281275809053,-0.11061214157194262,-0.13877304450903197,-0.1104048712367824,-0.07154189354758334,-0.031560954578583364,0.005138728700133021,0.0027084507985573117,-0.021140039549128724,-0.06683548753594526,-0.07881770249828425,-0.03466567551292095,-0.062043483518476786,-0.050857597407693286,-0.062032889630720456,-0.09084853322708004,-0.08011960872796085,-0.07913630140978421,-0.08245450986808503,-0.0738935752499211,-0.0534095354053193,-0.05712183112292699,-0.03883476336222305,-0.06776298477591658,-0.10769627780610946,-0.11004226102480227,-0.1452652622141785,-0.15228865368706193,-0.15782076594879804,-0.13618808111977215,-0.14943001299486963,-0.16059851844632342,-0.16798688894963687,-0.16388646136771903,-0.15219695615731096,-0.15030720696182817,-0.14866445812487816,-0.12674084667664925,-0.10958125010045856,-0.1271895891093879,-0.1209705568489716,-0.13880203886147552,-0.09030814074317436,-0.02995463855614854,-0.008950821019685043,0.0060293119026800365,0.032704321818709345,0.024293555062918196,-0.017783767671163818,0.023154840456270996,0.04254823140026154,0.0918024156441696,0.08446064989743887,0.08074993969091877,0.10663972828618573,0.11406137634496019,0.089969875720606,0.12209239919930744,0.1740254987668968,0.16246897335447258,0.19623201698216,0.17567052302184194,0.15627613176986768,0.12955477233585733,0.15658207081762296,0.1657037149800155,0.21766886521434553,0.20247412725583355,0.22237596503202411,0.24389559532232025,0.23762973982683944,0.28885358700692365,0.32451285737842295,0.34593774470467975,0.2615798243160165,0.20438097774078426,0.1457975023915886,0.11322848565622752,0.04980668241702517,-0.028922592789174174,-0.10971118592634499,-0.14882729792733515,-0.18712652315294576,-0.24134443836195857,-0.17563021643717955,-0.09866397669641624,-0.06399856628810525,-0.06506109375990823,-0.028453634451691974,0.06760286718905106,0.08806125968877673,0.06321058579489941,0.06771911531082266,0.1175837325955484,0.07182783806446078,0.044107521490780656,0.03810482405515013,0.05799802384909989,0.050904780079712775,0.036953966942395555,0.06857250136824578,0.1151012250887519,0.13578909455779883,0.0886090378037532,0.08951027760063171,0.10955190316817824,0.11250821817588655,0.13413286883355954,0.12430664768671895,0.08396381861342934,0.07329257348974962,0.038188193345396665,-0.009044589778944775,-0.015256151925278235,-0.0045445076892691155,-0.05246861494431748,-0.0587715303823278,-0.06535715978420338,-0.013237261347757578,0.013233505667679214,0.05869390991089877,0.07374535295291147,0.11381271437360632,0.14122384246000916,0.11040804581059671,0.09527391986325681,0.06675745067671889,0.08858320232974273,0.046167003664178675,0.036771763029792216,0.01149078855963081,0.03142268672981968,0.005035654665306413,0.003704623885512904,0.06050653395986262,0.11130785638455494,0.17995415589282782,0.14373279868151362,0.19635520422020974,0.20428487582131166,0.21639039507587995,0.19925316720539044,0.24711328106580815,0.284452160145538,0.2935175969723511,0.321342084531587,0.3098515470669679,0.33145215127901595,0.32987110578568346,0.3433264992982724,0.3489192010211565,0.36000373185819645,0.31537406808380986,0.31142631953121136,0.2910544667615966,0.2727652444592278,0.2628581415528497,0.23705198982364575,0.19870609764769157,0.14252387302031982,0.11582865366414159,0.11244730745776343,0.1507163564140476,0.15333059506249058,0.18430385269781868,0.22976667765171221,0.2533136449070478,0.28484788520109444,0.2929778812107261,0.35554942740929646,0.3726475684285336,0.34576792347097896,0.3210477935583951,0.3491037762789193,0.3299065708195854,0.27844754106112957,0.25743508538367305,0.26081874105666203,0.27266745434960354,0.2638251387123282,0.2464928890108477,0.28659796057478937,0.3312763114465672,0.3050549626617537,0.3012883163395395,0.3346989302810363,0.33486329825452116,0.3301922565300426,0.3322050130048638,0.2863526300971782,0.2699162492779011,0.22473434430231026,0.15544636707437112,0.13275167972309737,0.11931694013922278,0.08406661979112419,0.03878513478968485,-0.010554945158166013,-0.0670636354793239,-0.06052903933813969,-0.06742519164146084,-0.04271088994045351,-0.013048684732249382,-0.0039614953879429735,0.035215153507425496,0.04630256590919113,0.10053924664574798,0.13131782583875665,0.16381922662075793,0.15843971147312935,0.1722710119407085,0.14115946328106851,0.14147705843830963,0.10901686090711638,0.0434105877152348],"type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"title":{"text":"positivity_index_10d"},"xaxis":{"title":{"text":"date"}},"yaxis":{"title":{"text":"value"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e4b1b6b1-7d9b-454b-8214-21cf851c0b9a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


>...역시나 아무것도 안 보인다.

<h1>6) 분석 및 결론</h1> </br>

> 우선 이번 분석에서 한계점이 될 만한 점부터 먼저 찾아보았다. </br>
<br> 1) 언어 모델의 분석 능력의 한계 : BERT는 가벼운 모델이다보니 그만큼 텍스트 분류 능력도 많이 떨어진다. 더군다나 이번의 분석 대상은 커뮤니티 사이트 게시글이기 때문에 심하게 비정형화된 데이터를 가지고 정확하게 분류를 하기엔 한계가 명확했을 것이다.
<br> 2) 과연 내 직관대로 커뮤니티 사이트의 반응이 전체 사람들의 시장에 대한 감성을 대표할 수 있을까? 이건 대답할 수 없기 때문에 정확한 답을 내릴 수는 없겠지만, 한계점으로 지적될 수 있을 것 같다. </br>
<br> 관찰 결과를 토대로 결론을 내리면, 커뮤니티 사이트의 개미들의 반응만으로 시장의 향방을 알기에는 무리가 있다.

<h1>7) 배운 점과 반성할 점</h1> </br>

> 이 프로젝트는 6개월 정도 걸렸다. 그렇게 복잡한 프로젝트가 아님에도 불구하고 왜 이렇게 쓸 데 없는 시행착오가 많았는가? 곰곰히 생각해보니 이러한 시행착오는 크게 두 가지 이유 때문에 많이 발생한 것 같다. </br>
<br> 첫번째는 큰 목표를 잊어버린 채 눈 앞의 문제를 해결하는 데에만 골몰하여 결국 멀리 돌아가는 실수를 범했다. 예를 들어 라벨링 작업을 할 때, 처음에는 데이터의 숫자가 너무 많은 것에 겁먹은 나머지 처음부터 병렬에만 골몰하고 텐서화에는 전혀 신경쓰지 않았으며, 그로 인해 병렬 처리를 더 빠르게 하기 위해 Go 코드로 실행하였고, 그걸 디버깅하는 데 한 달이나 시간을 낭비해버렸다. </br>
<br> 두번째는 언어 모델에 대한 정확한 이해 없이 달려들다 보니 정밀하고 좀 더 효율적인 모델을 만들기까지 지나치게 많은 시간을 낭비했다. 처음에는 BERT 모델에 왜 torch layer를 올리는지, encoder의 torch layer가 어떻게 생겼는지조차 전혀 알지 모른 채 마구잡이로 모델을 만들었고, 이것 때문에 여러 차례의 훈련을 거쳐도 모델의 정확도가 올라가지 않는 일을 많이 경험했다. </br>
<br> 이후에 다른 프로젝트를 진행할 때는 사전 조사를 조금 더 철저하게 하고, 큰 목표를 설정하는 로드맵을 미리 짜 놓고 프로젝트를 시작해야겠다. 이 로드맵을 먼저 짜 놓아야 새로운 도구를 들일 때 이것이 정말로 나한테 필요한 도구인지 정확하게 판단할 수 있을 거라고 생각한다.
