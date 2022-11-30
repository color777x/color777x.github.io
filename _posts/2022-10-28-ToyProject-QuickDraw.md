---
title: "[Toy Project 3] 낙서 인식 프로젝트 (QuickDraw)"
header:
categories:
  - Toy Project
tags:
  - QuickDraw
  - Python
use_math: true
---



# Project 배경

  인간이 스스로를 표현하는 원초적인 방법 중 하나인 낙서. 인간의 낙서(비정형 데이터)를 컴퓨터는 얼마나 인식할 수 있을까? 여기부터 구글은 출발했을 수도 있다. 아니면 그림 검색을 생각한것인가? 최근에 알았지만  '오토드로우' 서비스를 준비하고 있었던건 아닐까 싶었다. 어쨌든 구글이 [Quick Draw](https://quickdraw.withgoogle.com/?locale=ko)서비스를 선보인지 몇 년이 지났다. 

고맙게도 구글은 손그림 데이터를 오픈해주어서, 이것을 기반으로 유아에게 낙서 인식 놀이 서비스를 만들어볼까하고 생각했다. 구글링 결과 실력자들이 만들어놓은 소스들이 많았고 통상적으론 CNN알고리즘을 이용했다. 사실 CNN이야 하면서 가볍게 들어갔지만, 역시나 전처리와 학습에서 컴퓨터의 성능, 프론트 서비스 등 현실적으로 걸림돌이 있었다. 

우리 서비스의 대상은 유아다. 오픈된 344개의 라벨링 중 유아에게 적합하지 않은 것들도 있었고, 전체 데이터를 한번에 학습 시키기에는 컴퓨터 성능도 모자랐다. 그리고 학습 데이터의 정확도는 외국 성인들의 데이터라는 것과 비정형 데이터라는 것을 감안하더라도, 실제 체감 정답률은 더 낮았던 것 같다.

그래서 근본적인 해결책은 아니지만... 일조의 꼼수를 생각했다. 실제 구글도 그렇게할지는 모르겠지만, 주제 카테고리별로 나누어 모델링을 하고 문제를 해당 카테고리 내에서 제출하여 해당 카테고리의 모델링으로 정답을 비교하는 것으로 서비스를 변경했다. 실력이 모자라서 모델링의 성능만으로 승부하기에는 한계가 있었다. 

아무튼... 또 팀원들과 함께 진행을 했다.

# Project Review

## 기획

  유아에게 있어서 낙서 또는 그림이 차지하는 비중은 상당히 크다고 인식하고 있고, 그 의미를 파악하기 위해 유아 미술 심리 등의 전문적인 학문들이 등장하기도 했다. 최근에는 AI 기반으로 미술 치료에 대한 연구도 활발하게 이루어지고 있다. 

이 서비스는 유아 미술 치료로 발전하기 전 아주 기초적인 단계로써 유아가 그리는 그림을 인식하게하고 데이터를 수집하게 하기 위함을 장기적인 목표로 하였다. 하지만, 멀리 또는 크게 바라보지 않더라도 단순한 놀이 정도로 생각해도 괜찮을 것 같았다.

작업 진행은 아래와 같은 순서로 하였다. 

첫번째는 현실적인 이슈로 아래 그림 처럼 데이터의 카테고리부터 정의했다. 대략 15개 정도의 카테고리가 나뉘어졌고, 하나의 모델 당 20개 정도의 라벨이 붙는다고 보면 된다.

<img src="/assets/img/post/22.10.28/quick_d.PNG" width="1000px" alt="QuickDraw 카테고리화">

두번째는 화면에서 사용자가 선을 그을 때마다 모델링에서 정답을 맞추는 부분을 기획했고, 유아임을 감안하여 그리는 시간과 스테이지 수를 조절하였다. 

사실 첫번째의 카테고리화 외에는 구글 Quick Draw서비스를 똑같이 만들었다. 그래서 기획 단계에서 많은 이야기를 할 것이 없다. 카테고리화가 핵심이다.

## 개발

  개발은 두 부분으로 나누어 진행되었다. 첫번째는 데이터를 모델링을 만드는 부분이고, 두번째는 모델링을 기반으로 서비스를 만드는 부분이다. 처음에는 컴퓨터 성능 때문에 Colab Pro를 결제해서 썼었다. 그러나 결국 Colab Pro에서도 해결되지 않아, 카테고리화된 단위로 jupyter notebook으로 진행하였다.

### 1. npy 데이터 읽어오기 및 인덱스 라벨링

구글은 .npy형태로 데이터를 제공하였다. .npy에 대해서는 별도 포스팅으로 설명하겠다. 첫 번째 카테고리의 데이터를 읽어왔다. 각 데이터는 대략 11만에서 많게는 20만 개의 이미지를 제공한다. 해당 이미지는 28*28 픽셀로 되어 있다.

```python
# face
beard = np.load('data/beard.npy')
ear = np.load('data/ear.npy')
eye = np.load('data/eye.npy')
face = np.load('data/face.npy')
mouth = np.load('data/mouth.npy')
nose = np.load('data/nose.npy')
```

배열 데이터의 끝에 0~7까지 8개의 라벨링을 했다. np.c_라이브러리는 두 개의 배열을 하나로 합치는 기능을 제공한다. 데이터 세트 중 1개 행의 길이는 785이다. 샘플 데이터를 그려보자.

```python
# face
beard = np.c_[beard, np.zeros(len(beard))]
ear = np.c_[ear, np.ones(len(ear))]
eye = np.c_[eye, 2*np.ones(len(eye))]
face = np.c_[face, 3*np.ones(len(face))]
mouth = np.c_[mouth, 4*np.ones(len(mouth))]
nose = np.c_[nose, 5*np.ones(len(nose))]

print(len(beard[0]))
```

### 2. 이미지 데이터 확인하기

```python
def plot_samples(input_array, rows=30, cols=10, title=''):
    
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        
plot_sample(beard, title='수염')
```

마지막 라벨 데이터를 제외하고 784까지 읽어와서 28*28로 reshape해주고 plt.imshow를 사용하여 그림을 그려주면 아래와 같은 그림이 나온다.

<img src="/assets/img/post/22.10.28/beard.PNG" width="1000px" alt="QuickDraw 카테고리화">

### 3. X, y 데이터 만들기

라벨을 제외한 X 데이터를 생성한다. 

```python
X = np.concatenate((
# face
beard[:50000,:-1],
ear[:50000,:-1],
eye[:50000,:-1],
face[:50000,:-1],
mouth[:50000,:-1],
nose[:50000,:-1]
), axis=0).astype('float32')
```

y 라벨 데이터도 생성한다.

```python
y = np.concatenate((
# face
beard[:50000,-1],
ear[:50000,-1],
eye[:50000,-1],
face[:50000,-1],
mouth[:50000,-1],
nose[:50000,-1]
), axis=0).astype('float32')

```

5만 개의 X, y 데이터를 8:2의 비율로 train 데이터와 test 데이터로 나누어 준다.

```python
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2, random_state=0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
```

그리고 카테고리형 라벨링(종속변수)의 경우 One Hot Encoding을 한다. 각 라벨링 값에 대한 독립성을 부여하기 위해서이다.

```python
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
```

다음은 CNN Model에 인풋값 shape을 맞추기 위해 X_train을 reshape한다. 

```python
X_train.shape
> (240000, 784)
# 각 라벨당 5만개를 학습 데이터로 가져왔으니, 5만 X 9개 라벨의 80% 
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
X_train_cnn.shape
> (240000, 28, 28, 1)
```

### 4. 모델링 설계 및 학습

이제 전처리는 끝났다. 위에서도 언급했듯이 CNN Model을 사용할 예정이고, tf.keras.models 모듈의 Sequential 클래스를 사용했다. CNN Model의 이론적 배경은 별도로 정리하겠다. 

이 프로젝트에서는 다양한 파라메타 값을 바꿔가면서 학습시켜보았으나, 기본적으로 아래와 같은 구조를 가진 모델로 진행하였다. 학습 시키는 데이터가 흑백이기 때문에 채널은 1, 28X28의 이미지 사이즈, 3X3의 필터 크기로 진행하였다. 두 개의 CNN 계층을 두고 중간에 Dropout 0.3을 설정한 뒤 Flatten을 진행하였다. 그리고 일반적인 선형회귀모델의 딥러닝 계층을 두었다.

```python
def cnn_model():
    model = Sequential()    
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.3))
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
  
%%time
np.random.seed(0)
model_cnn = cnn_model()
history = model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=22, batch_size=200)
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])  
```

참고로 입력 대비 출력의 데이터 사이즈와 파라미터의 개수를 계산하고 넘어가자. 우선 Convolution Layer의 출력 데이터 shape size의 계산은 아래와 같이 한다. 아래 수식은 출력 데이터의 높이만 언급한 것이고 가로는 동일하게 대응하여 계산하면 된다. 

- OH: 출력 데이터의 높이
- H: 입력 데이터의 높이
- P: 패딩
- FH: 필터 높이
- S: 스트라이드 

$$
OH = \frac{H+2P-FH}{S} + 1
$$



그리고 Convolution Layer의 파라미터 개수는 아래와 같이 구한다.  

- W: Conv weight
- K: 커널 size
- C: 채널 수
- N: 커널 수
- B: Conv biase

$$
W = K^2 * C * N \\
B = N \\
O = W + B
$$



MaxPooling Layer의 출력 데이터 size의 계산은 아래과 같이 한다.  

- O: 출력 데이터
- I: 입력 데이터의 높이
- Ps: 풀링 사이즈
- S: 스트라이드 

$$
O = \frac{I - Ps}{S} + 1
$$



그래서 아래와 같이 나온다.

<img src="/assets/img/post/22.10.28/quick_param.png" width="1000px" alt="QuickDraw 파라메터">

<img src="/assets/img/post/22.10.28/quick_acc.png" width="1000px" alt="QuickDraw 정확도">

<img src="/assets/img/post/22.10.28/quick_loss.png" width="1000px" alt="QuickDraw 손실">

### 5. 결과 데이터 확인 및 모델 저장

이번 프로젝트의 경우 손그림 비정형 데이터이기 때문에 각 오브젝트 라벨별 틀린 예측의 메트릭스를 볼 필요가 있었다.

```python
def confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

위 과정을 반복하면서 현 데이터의 한계(28*28)에서 틀린 오브젝트가 많은 라벨을 드랍시킨 모델을 만들었다.

```python
from keras.models import load_model

model_cnn.save('model/'+model_name+'_model_acc'+last_accuracy+'_loss'+last_loss+'_'+nowtime+'.h5')
```



## 프론트 서비스

사실 프론트 서비스 쪽은 굳이 정리하지 않아도 될 듯하지만 간단하게 기술해보자. 

* legacy: springframe work
* 서비스 웹서버: flask
* DB: maria

서비스 카테고리를 DB에 정의하고 legacy 서버쪽 api를 통해 사용자에게 제시될 주제를 받아온다. 사용자는 제시된 주제를 보고 40초 안에 그림을 그린다. 선을 그리고 뗄때마다 모델에서 판단한다. 대략 아래와 같은 화면이다. 구글의 Qucik Draw와 비슷하게 만들었지만 많이 부족하다. 나중에 더 업그레이드를 해야겠다.

<img src="/assets/img/post/22.10.28/quick_d01.jpg" width="1000px" alt="서비스 화면1">

<img src="/assets/img/post/22.10.28/quick_d02.jpg" width="1000px" alt="서비스 화면2">

<img src="/assets/img/post/22.10.28/quick_d03.jpg" width="1000px" alt="서비스 화면3">

## 마무리

대학원 다닐 때 Toy Project로 해본 경험으로 힘들게 끌고 왔다. 뿌듯한 부분도 있지만 이내 곧 실제 서비스에서의 느낌을 구글처럼 끌어올리기에는 한계가 너무 많다는 현실에 부디친다. 구글처럼 원화가들을 동원하여 오토드로우 서비스 작업을 진행해볼까도 생각해보지만, 해당 서비스에 투입되는 우선순위 비중이 높지 않다.

서비스는 beta를 붙여서 AI연구소라는 이름으로 나가고 있다. 마음같아서는 생각하고 있는 꿈꾸는 미래 서비스에 대한 공감을 얻어 함께 발전시켜가고 싶지만, 우선은 여기까지에서 만족하고 천천히 걸어갈 생각이다.