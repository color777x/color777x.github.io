---
title: "추천 엔진 프로젝트"
header:
categories:
  - Toy Project
tags:
  - Recommender System
  - Python

---

#### Toy Project

* 프로젝트 개요: 유아 교육 콘텐츠 서비스에 추천 엔진 적용하기
* 적용 기법: Item Based Collaborative Filtering
* 알고리즘: Cosine Similaity
* 데이터: 5점 척도 평가 데이터



* 개발 환경
  * Centos7.3 / crontab 활용 (매일 새벽 2시)
  * Python3.8
  * maria db
* Python 라이브러리
  *  
  *  
  *  
* 프로젝트 구조
  1. maria db 접속
  2. 콘텐츠 평가 Table에서 데이터 가져오기
  3. 중복데이터 없는 콘텐츠 DataFrame만들기
  4. 추천 콘텐츠 Master Table 만들기
  5. Pivot Table만들기
  6. Cosine Similarity 계산하기
  7. 계산 결과에 따라 유사도 내림차순으로 정렬
  8. Master Table의 콘텐츠만큼 추천 Table에 Input
  9. 종료
* 코드 리뷰

~~~python
~~~



* 프로젝트 후기

