---
title: "Anaconda-설치한 패키지 리스트 확인"
header:
categories:
  - Development Environment
tags:
  - Anaconda
---





### 패키지 리스트를 왜 확인해야하지?

​    다른 개발 환경에서 같은 프로젝트를 세팅해야 하기 때문이다. Python 라이브러리들이 버전을 많이 타서 git에서 받아도 안되는 경우가 많다.

* Anaconda Prompt 실행
* 가상 환경 실행

~~~python
conda activate 가상환경이름
~~~

* list 명령어 실행

~~~python
conda list
~~~

* 결과 확인

모든 리스트가 다 나오는데, 개발 소스에서 import한 라이브러리만 정리해서 requirements.txt 파일에 만들어서 보관한다.

아래는 Toy Project #1. 추천엔진의 라이브러리다.

-------------------

python==3.8.5

pandas==1.2.1

pymysql==1.0.2

-------------------



