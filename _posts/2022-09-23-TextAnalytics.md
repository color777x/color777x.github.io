---
title: "자연어 분석을 통한 유아 언어 발달 현황 분석"
header:
categories:
  - Toy Project
tags:
  - Text analytics
  - Python
---



# Project 배경

영유아아동의 언어 발달을 진단하기 위한 도구로 '영유아아동 언어발달검사 (SELSI)', '취학 전 아동의 수용 표현언어검사 (PRES)' 등 많은 학문·의학적 접근 방법은 많이 있지만, 디지털 교육 시장에서 종사하는 사람으로써 데이터적으로 접근하고 싶었다.

방법은 간단하다. 

1. 서비스를 이용하고 있는 유아 사용자들에게 '말하는' 활동 제공
2. 말들을 Text로 변환
3. 형태소로 분리
4. 분리된 형태소를 통계적으로 분석 뒤 피드백
5. 이후 데이터가 쌓이면 또래를 기준으로 비교하여 언어 발달 진단까지 연결
6. 결핍된 지점의 형태소가 많이 포함되어 있는 동화책이나 동화책 음원을 추천

1~4번까지는 완료되어 서비스하고 있는 상태이나, 5~6번은 데이터도 쌓은 뒤 여러 전문가의 자문도 받아서 진행해야할 것 같다. 회사 내부적으로 유관부서의 공감과 지지를 얻기 힘들었다. 교육 시장에서 이십년가까이 있지만 여전히 아쉬운 지점이다. 새로운 것들에 대한 도전과 다른 것에 대한 수용 등 어려운 부분이 많다.

하지만 다행스럽게도 현재 필자의 포지션은 팀을 이끌고 있는 상태이고 '우격다짐'? 같은 느낌으로 시간나는 대로 팀원들을 괴롭히면서 프로젝트를 끌고 왔다. python으로 분석하는 부분은 필자가 구현하고 java로 만들어진 서비스에는 팀원 개발자들의 도움을 받아 진행했다. 아이디어의 출발부터 서비스 적용까지 약 10개월 정도 걸린 것 같다. 중간 중간 우선순위 업무에 많이 밀렸다. 실제 개발은 2개월 정도 걸렸다.



# Project Review

## 기획

먼저 내부 설득을 위해서 근거 자료가 필요했다. 언어 발달과 관련한 논문을 참고했다. 

> 한국아동의 문법형태소 습득에 관한 연구: 조사 "가, 이 , 는 , 도 , 를”
>
> 언어 교육프로그램이 유아의 언어 능력에 미치는 효과에 대한 메타분석
>
> 한국아동이 초기에 획득한 문법적 형태소의 종류 및 획득 시기
>
> 한국아동의 언어 발달 연구
>
> 초기언어 발달에서 환경적 요인들의 역할 -음소지각, 어휘 습득, 구문 발달을 중심으로
>
> 유아의성별과 놀이상황 유형별 평균발화길이와 어휘다양도

논문 자료들을 바탕으로 서비스의 근거를 마련하고 설계하였다. 우선 아이들에게 그림을 그릴 수 있는 주제를 제시하고 그림을 그리면서 설명할 수 있는 활동을 만들었다. 또 다른 활동은 상황을 제시하고 설명하는 활동이었다. 이 활동을 하면서 아이들의 말을 분석하여 학부모 서비스에서 리포트 형태로 제공한다.

## 개발

- 개발 언어: Python
- 사용 library: stt, konlpy, Kkma, ffmpeg 등
- 서비스 환경: ubuntu, conda env, flask, mariadb 등

### 영상 변환 (mp4 -> wav)  및 google stt를 활용한 텍스트 추출

유아 서비스에서 api 제공해주어 영상 정보를 받아왔다. 영상의 포멧은 mp4로 저장되어 있어서 텍스트 추출을 위해 wav로 변환해줄 필요가 있었다. 그래서 우선 python의 ffmpeg라이브러리를 활용하였다. ubuntu환경에서 python ffmpeg가 잘 적용이 안되어서 apt-get으로 설치 후 subprocess의 Popen을 활용하여 쉘 커맨드 실행 및 리턴받는 방법으로 변환했다. ffmpeg에서 가장 삽질이 많았던 것 같다.

```python
args = shlex.split('ffmpeg -i ' + self.input_file + ' -y ' + self.output_file)
proc = subprocess.Popen(args)
# 이것은 subprocess로 하위 프로세스가 완료되기 전까지 체크하기 위한 방법이었다.
while True:
  poll = proc.poll()
  if poll is None:
    print('proc is alive')
    else:
      print('proc is dead')
      break
      time.sleep(1)
      proc.wait()
      result_code = 200        
```

이렇게 해서 얻은 wav파일을 google stt라이브러리를 활용하여 텍스트 추출을 하였다. 

```python
import speech_recognition as sr
import mov_converter
import text_analytics

r = sr.Recognizer()

def google_wavtotext(self):
  nowDate = datetime.datetime.now()
  reqDate = nowDate.strftime("%Y%m%d%H%M%S")

  savename = self.videoUrl
  outputname = self.childIdSq+"_"+reqDate+".wav"

  # FFMconverter 객체 생성
  convert = mov_converter.FFMConvertor(savename, outputname)
  outfile2, result_code = convert.convert_mp4_subprocess()

  if result_code == 200:
    korean_audio = sr.AudioFile(outfile2)
    with korean_audio as source:
      audio = r.record(source)
      path = os.path.dirname(os.path.abspath(__file__))
      with open(path+"google stt 인증키가 포함된 json파일 경로", "r") as f:
        credentials_json = f.read()
        try:                
          # google stt 무료 버전. 횟수 제한이 있다.
          # text = r.recognize_google(audio_data=audio, language='ko-KR')

          # 유료 버전. 회사에 다른 서비스에서 쓰고 있어서...ㅋ
          text = r.recognize_google_cloud(audio_data=audio, language='ko-KR', credentials_json=credentials_json)
          # 형태소 분석 객체 생성
          analytics = text_analytics.Analytics(text)
          # analytics_Kkma에서 분석 후 결과 리턴
          nlp_result = analytics.analytics_Kkma()
          except sr.UnknownValueError:
            nlp_result = 100
            print("Google Speech Recognition could not understand audio")
          except sr.RequestError as e:
            nlp_result = 100
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
  # 분석하면서 생성된 wav 파일 삭제
  file_del_result = convert.file_delete()
             
```

### 형태소 분석 (Kkma)

KoNLPy는 한국어 정보처리를 위한 파이썬 패키지이다. (https://konlpy.org/ko/latest/index.html) 해당 패키지에서 제공하는 라이브러리 중 Kkma는 67개나 되는 형태소를 분석해주는 반면에 속도가 느리다. 그래서 실시간 분석 서비스에는 부적합하다고 판단하지만, 우리 서비스는 실시간 서비스가 아니기 때문에 자세한 분석을 택했다.

```python
from konlpy.tag import *

def analytics_Kkma(self):  
  # kkma 라이브러리 내용
  kkma = Kkma()
  # pos함수는 입력된 문장을 리스트 형태로 반환해준다.  
  malist = kkma.pos(self.input_text)
  # 리스트 안에 튜플 형태로 반환되는 결과 값을 for문으로 가져와서 후처리 해준다.
  # 튜플의 첫번째 값은 입력된 문장의 분석된 텍스트이고, 두번째 값은 형태소 품사 정보이다. 
  result = {}
  for i in range(len(malist)):
    if malist[i][1] not in result:
      v = malist[i][0]
      k = malist[i][1]
      result[k] = v
      else:
        v = malist[i][0]
        k = malist[i][1]
        result[k] = result[k] + ',' + v
  # 이후 기타 처리   
```

이렇게 분석되어진 결과를 서비스 db에 입력하고 몇 가지 설정한 형태소 분석의 규칙에 따라 부모들에게 피드백을 한다. 



그러나 형태소를 분석해서 부모에게 유의미한 서비스를 제공하는 것은 쉽지 않은 일 같다. 학문적으로 증명되지 않은 정의에 대해서 전달하는 것에 대한 가치 문제와 단순하게 말의 형태소의 구성 속에서 특징을 찾아 내는 것 자체도 쉽지 않은 일이었다. 

하지만! 개선 방안은 추후 찾아서 더 날카롭게 다듬으면 되는 것이고~ 우선은 이렇게 서비스해보면서 관찰해보아야 겠다. 아무것도 하지 않으면 아무것도 변하지 않아.