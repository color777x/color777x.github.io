---
title: "Toy Project #1. 추천 엔진"
header:
categories:
  - Toy Project
tags:
  - Recommender System
  - Python

---

* 프로젝트 개요: 유아 교육 콘텐츠 서비스에 추천 엔진 적용하기
* 적용 기법: Item Based Collaborative Filtering
* 알고리즘: Cosine Similaity
* 데이터: 5점 척도 평가 데이터



* 개발 환경
  * Centos7.3
  * Python3.8
  * maria db
  
* Python 라이브러리
  *  python==3.8.5
  *  pandas==1.2.1
  *  pymysql==1.0.2
  
* 개발 로직
  1. maria db 접속

     ~~~python
     # SSH 접속
     tunnel = SSHTunnelForwarder(('호스트 주소', 포트),
                              ssh_username = '서버 계정',
                              ssh_password = '서버 비번',
                              remote_bind_address = ('private서버 주소', 포트))
     tunnel.start()
     conn = pymysql.connect(host='127.0.0.1',
                                port=tunnel.local_bind_port,
                                user='DB 사용자 계정',
                                passwd='DB 비번',
                                db='DB 이름')
     
     # 일반 접속
     conn = pymysql.connect(host='DB 서버 주소', port=포트, user='DB 사용자 계정', password='dB 비번', db='DB 이름', charset='utf8')
     ~~~

  2. Table에서 데이터 가져오기 및 전처리

     ~~~python
     # 평가 데이터 쿼리
     cursor = conn.cursor()
     sql = "SELECT * FROM CONTENT_EVALUATION_INFO"
     cursor.execute(sql)
     resultList = cursor.fetchall()
     
     # 전처리를 위한 DataFrame으로 변환
     resultList = pd.DataFrame(resultList)
     
     # 평가한 전체 콘텐츠 가져오기 - 중복 콘텐츠 ID 제거
     condata = resultList.drop_duplicates("CONTENT_EVALUATION_CONTENT_ID")
     condata = condata["CONTENT_EVALUATION_CONTENT_ID"]
     ~~~

  3. Contents Master Table 생성

     ~~~python
     for i in condata2.index:
         conid = condata2["CONTENT_EVALUATION_CONTENT_ID"].loc[i]
         concd = condata2["CONTENT_EVALUATION_CONTENT_CD"].loc[i]
         now = datetime.now()
         formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
         insert_sql = "INSERT INTO RM_CONTENTS_INFO (RM_CONTENTS_ID_SQ, RM_CONTENTS_CD, UPDATE_DT) \
         									VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE RM_CONTENTS_ID_SQ = %s"
         val = [
             (conid),
             (concd),
             (formatted_date),
             (conid)
         ]
         cursor.execute(insert_sql, val)
         conn.commit()
     ~~~

  4. Cosine Similarity 

     ~~~python
     # Pivot Table 생성
     data_IBCF = resultList.pivot_table('MB_STUDY_CONTENT_EVALUATION_SCORE', index = 'MB_STUDY_CONTENT_EVALUATION_CONTENT_ID_SQ', columns = 'MEMBER_CHILD_SQ').fillna(0)
     # Cosine Similarity
     con_sim = cosine_similarity(data_IBCF, data_IBCF)
     con_sim_df = pd.DataFrame(data = con_sim, index = data_IBCF.index, columns = data_IBCF.index)
     ~~~

  5. 추천 Table에 Input

     ~~~python
     for i in condata.index:
         # 기존 추천 콘텐츠 데이터가 있으면 일단 지운다. 없으면 말고...
         delete_sql = "delete from RM_CONTENTS_DTL_INFO where RM_CONTENTS_ID = %s"
         val1 = [
             (condata[i])
         ]
         cursor.execute(delete_sql, val1)
         # 추천 콘텐츠 10개 씩 테이블에 Insert
         for j in range(1, 11):
             now = datetime.now()
             formatted_date = now.strftime('%Y-%m-%d %H:%M:%S')
     
             #콘텐츠 CD가져오기 위해서
             condata_id = condata2.query('CONTENT_EVALUATION_CONTENT_ID == "' + condata[i] + '"')
             condata_cd = condata_id['CONTENT_EVALUATION_CONTENT_CD'].values
             condata_cd = condata_cd[0]
     
             # 유사도 순서에 따라 정렬하여 1을 제외하고 젤 위에서부터 가져오기
             recommend = con_sim_df[condata[i]].sort_values(ascending=False).index[j]
             percentage = con_sim_df[condata[i]].sort_values(ascending=False)[j]
             percentage = percentage * 100
     
             insert_sql2 = "INSERT INTO RM_CONTENTS_DTL_INFO (RM_CONTENTS_ID, RM_RECOMMEND_CONTENTS_ID, RM_CONTENTS_CD, SQ_NUM, PERCENTAGE, UPDATE_DT) VALUES (%s, %s, %s, %s, %s, %s)"
             val2 = [
                 (condata[i]),
                 (recommend),
                 (condata_cd),
                 (j),
                 (percentage),
                 (formatted_date)
             ]
             cursor.execute(insert_sql2, val2)
     
     conn.commit()
     # tunnel.close() --SSH로 접속했을 경우
     ~~~

  6. Centos에서 Crontab 설정하여 정기적으로 배치로 데이터를 입력한다.

     ~~~
     # crontab 리스트 확인
     > crontab -l
     
     # crontab 편집 명령어 실행
     > crontab -e
     
     # * 자리 왼쪽부터 분, 시간, 일자, 월, 요일 
     # 첫번째 경로는 anaconda 가상환경 python 실행 경로, 두번 째는 실행 파일 경로, 세번 째는 로그 파일 저장 경로
     * */1 * * * /home/shin/anaconda3/envs/python_anal/bin/python3 /home/shin/python_anal/IBCF.py >> /home/shin/python_anal/cron.log
     ~~~

  7. 프로젝트 종료

  

* 프로젝트 후기

  ​    한 달에 한 번씩 AI 프로젝트나 데이터 분석 프로젝트를 진행하려고 했었는데 현업을 핑계로 미루다가 지금에서야 시작했다. '낮기밤개'한다고 떠벌렸지만 아직은 Toy Project 수준이다. 다른 AI전문 개발자들의 실력을 모르는 불안감도 있기 때문에, 내가 개발한 것에 대해서 낮추어 부르는 마음이 반영되어 Toy Project이다. 

  나는 교육 시장에서 IT기획을 업으로 삼고 살아왔다. 서른 초반 4년 정도 개발자의 시간이 있었지만, 그래도 기획과 PM으로 보낸 시간이 헐씬 길다.

  그래서인지 내가 만든 추천 엔진이 좋은 것인지도 사실 잘 모른다. 여러 자료를 찾아보고 유투브 강의를 들으면서 응용한 프로젝트이다. 그래서 Toy Project를 진행하면서 알게된 알고리즘도 함께 다룰 예정이다.

  

  어쨌든 추천 서비스는 최근 몇 년 사이 교육 시장에서도 다시 관심을 받고 회자되고 있어서 첫 프로젝트로 선정해보았다. 서비스 적용하기에 프로젝트 규모가 만만해보였을 수도 있다.

  누구에게 추천을 해주려면 취향을 알아야 한다. 디지털 서비스도 마찬가지로 고객이 어떤 상품을 좋아하는지 알아야 한다. 디지털 서비스는 '많은 사람들이 좋아하는 > 당신과 비슷한 사람들이 좋아하는 > 너가 좋아할 것 같은'의 개념으로 바뀌고 있다. 이런 이야기가 진부해보인다.

  그런데 교육의 관점에서 선호도 기반으로 추천한다는 것이 과연 옳은 것인가 하는 생각도 있다. 교육은 교수법 안에서 지식을 전달하고 앞 뒤 맥락이 강하게 연결되어 있는 영역이기 때문에 테스트를 통해서 취약한 부분을 강화하고 더 난이도 높은 수준의 교육을 제공하는 목적으로 다른 콘텐츠를 추천하는 것이 일반적이다.

  우리 서비스는 유튜브와 같이 고객이 직접 선택적으로 이용하기 때문에 선호도 기반의 추천 엔진을 적용하는 것에 무리가 없을 것이라고 판단하였다.

  서비스에 적용 후 성과는 어떠한가? 성공적인가? 라는 질문을 한다면 현재로써는 성공적이다. 전체적으로 접속자 통계가 해당 시점 이후부터 증가하고 있고 콘텐츠의 활용 통계도 함께 증가하였다. 물론 해당 시점에 외부적인 요인이 어떻게 작용했는지는 모두 검증할 수는 없다.

  개발적으로는 원격지의 DB 서버에 접속하기 위해서 네트워크 구성이나 방화벽 정책을 잘 몰라서 SSH로 접속할지 직접 접속할지 좌충우돌하면서 진행했던 것이 가장 기억에 남는다. 누군가 옆에서 도와줄 수 있는 사람이 있으면 참 좋겠다라는 생각이 간절했다. 

  괜찮다. 어차피 늦게 시작했고 길게 보면서 가려한다.

  다음 Toy Project 리뷰는 '감성분석엔진'이다.  

  

  

