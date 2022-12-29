---
title: "합성곱 신경망 (Convolution Neural Networks)"
header:
categories:
  - algorithm
tags:
  - CNN
  - 딥러닝 알고리즘
use_math: true
---



# Convolution Neural Networks 합성신경망

  Convolution Neural Networks(합성신경망, 이하 CNN)에 대해서 어떻게 정리할까 고민하다가 알고리즘이 나온 배경이나 역사는 생략하고 본론부터 정리하기로 했다. 요약하자만 1950년대부터 시작된 David Hubel과 Torsten Wiesel의 [연구 논문](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/pdf/jphysiol01247-0121.pdf)(고양이의 시각 피질 실험)으로부터 영향을 받은 Kunihiko Fukushima, Yann Lecon 박사 등을 시작으로 기존의 신경망 접근이 아닌 CNN 방식의 LeNet-5, AlexNet, ZFNet 등으로 발전되어왔다. 

  Fully-connected multi-layered neural network를 **이미지 기반의 데이터**에 사용했을 때 학습 시간, 과적합 등의 문제가 발생하기 때문에 기존의 신경망 대신 CNN알고리즘에 주목하게 되었다. 이제 본격적으로 CNN에 대해서 정리해보겠다.

## 1. Convolution 개요

  Convolution 또는 합성곱에 대한 위키백과의 정의는 '하나의 함수와 또 다른 함수를 반전 이동한 값을 곱한 다음, 구간에 대해 적분하여 새로운 함수를 구하는 수학 연산자'로 되어 있다. 

이 말이 무슨 말인지 이해하기 힘들었다. 아래 합성곱에 대한 식으로 설명하자면  $τ​$로 적분했을 때 최종적으로 으로는 $t​$에 대한 함수로 만들어지는데, 위 정의에 있어서 $f(τ)​$가 앞에 있는 '하나의 함수', $g(t-τ)​$이 '반전 이동한 값'이 된다. 아래 그래프 에서 $g(t)​$를 반전 이동하면 $g(t-τ)​$의 그래프가 되고, $f(τ)​$와 $g(t-τ)​$의 각 값을 곱한 결과를 적분한다는 의미이다.


$$
연속변수: (f*g)(t) = \int_{-\infty}^{\infty }f(τ)g(t-τ)dτ
$$

$$
이산변수: (f*g)(t) = \sum_{τ=-\infty}^{\infty }f(τ)g(t-τ)
$$

<details><summary>합성곱 연산 설명 그래프</summary>출처 위키백과</details>

![](/assets/img/post/22.12.02/375px-Convolution3.png){: #magnific}

![](/assets/img/post/22.12.02/convolution.gif){: #magnific}

#### 그렇다면 합성곱을 이미지 연산에 어떻게 사용한다는 것일까? 

![합성곱 이미지 연산](/assets/img/post/22.12.02/cnn_03.gif){: #magnific width="600"}

  나름대로 그려본 합성신경망 연산에 대한 설명이다. 위에서 설명한 합성곱 연산이 이미지 연산에서는 이런 개념으로 들어간다. 그림에서 보면, 녹색 네모(kernel or filter)가 옆으로 움직이면서 input image의 값을 convolution 계산을 하여 feature map이라는 것에 하나씩 채워지는 것을 볼 수 있다. 위 식에서 이야기하는 $f(τ)$가 input image이고 $g(t-τ)$의 값에 해당하는 것이 kernel or filter(이하 filter)인 것이다. 이렇게 한칸씩 옆으로 이동하면서 input image전체의 convolution을 계산하여 새로운 featuer map을 만들어 고유 특징값만을 추출해 내는 기법인 것이다.

  다만, 실제로 CNN에서 C는 Cross-Correlation(교차상관, 이하 Convolution)을 사용한다고 한다. (이 부분에 대해서는 위에서 설명한 개념이 수식과 일치하지 않아서 헷갈리는 부분이 있을 듯 하다. 나와 동일하게 헷갈리셨던 분이 잘 설명해놓은 블로그를 링크 걸어둔다. [참고 블로그](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sw4r&logNo=220904800372))

그럼 CNN에 포함되어 있는 여러 개념들을 지금부터 하나씩 알아보자.

----



## 2. Convolution Layer

#### 2.1 Channel 

  입력되는 이미지가 흑백이냐 컬러냐에 따라 채널 값이 결정된다. 위 예제는 흑백이기 때문에 1채널을 사용하면 된다. (28X28X1) 그러나 컬러 이미지의 경우는 RGB컬러로 이루어진 값인 3개의 채널을 사용한다.(28X28X3)

#### 2.2 Kernel or Filter

  기존 신경망 대비 가장 다른 부분은 Feature 을 뽑아내는 filter의 사용일 것이다. 이 filter의 역할이 가중치 파라미터에 해당하며 사용자가 임의로 설정한 랜덤값들로 이루어진다. 이 filter가 input data 전체를 stride(filter의 이동범위)하면서 Convolution 연산을 수행한다. 

  filter의 개수는 사용자가 정한다. 당연히 filter의 개수가 많으면 Feature map을 다양하게 추출되지만, 상대적으로 연산량이 많아지므로 각 layer에서 연산량을 비교적 일정하게 유지하여 시스템의 균형을 맞춘다. 

  filter의 사이즈는 이미지의 지역정보를 얼만큼 가져갈 것인가를 결정하는데 3X3이라던가 5X5를 많이 사용한다. 입력 데이터의 사이즈에 비례하여 사용하겠지만, 일반적으로 수행하는 이미지 연산 (32X32) 또는 (28X28)의 이미지 연산에서는 3X3, 5X5를 많이 사용한다.

#### 2.3 Padding

  Padding은, Filter가 Stride하면서 뽑아낸 Feature map의 사이즈를 보전(?)해주기 위한 방법이다. Convolution 연산을 수행하면서 해당 지역에 특징값들을 뽑아내면 당연히 input data의 사이즈보다 작아지게 되어 있다. 이때 가장자리의 정보들이 사라지는 문제가 발생하기 때문에 Padding값으로 인해서 작아지는 부분을 채워넣어준다. 통상적으로 0으로 채워 넣는 방법(Zero Padding)을 선택합니다. 

- N * N 사이즈의 Input data, F * F 사이즈의 Filter의 Feature map: N-F+1
- 위와 동일한 조건에 1Padding의 Feature map: N+2P-F+1

예를 들어 28X28 Input data, 3X3 Filter의 경우 padding없이 계산하면 28-3+1 = 26 * 26이 나오고, 여기에 1padding을 적용하면 28+2*1-3+1 = 28이 되어 Input Data의 Shape이 유지된다.

#### 2.4 Stride

  Stride는 Filter가 Input Data의 전체를 훑을 때 몇 칸씩 이동하느냐의 문제이다. 어쨌뜬 Filter가 전체 이미지의 특징을 뽑아내려면 꼼꼼하게 뽑아내야하니 통상적으로 1로 설정을 한다. 다만, 입력 데이터가 큰 영상의 경우 연산량에 대한 문제로  인하여1이상의 값을 사용하기도 한다.

```python
model = Sequential()    
model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
print("1. Convolution Layer:", model.output_shape)
```

```python
1. Convolution Layer: (None, 26, 26, 30)
```

----



## 3. Pooling(Sub-Sampling) Layer

  Convolution Layer에서 한번 계산이 끝나면 다음으로는 Pooling Layer에 태운다. Pooling Layer를 하는 이유는 Feature Map의 모든 data가 필요하지 않고 특징을 뽑아낼 정도만의 data를 선택함으로써 신경망의 계산 효율을 높인다. 

  Pooling Layer의 기법은 Max Pooling, Average Pooling, Stochastic Pooling 등 많이 있습니다. 아래 그림과 같이 Max Pooling은 말 그대로 Feature Map에서 가장 큰 값만을 취하지만 Overfitting의 단점이 있고, Average Pooling은 평균값을 취하지만 학습 결과가 좋지 않다는 단점이 있다. 이 단점을 보완하기 위해 나온 것이 Stochastic Pooling이다. 

![](/assets/img/post/22.12.02/pooling.png){: #magnific}

 

```python
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    print("1. Convolution Layer:", model.output_shape)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("2. Pooling Layer:", model.output_shape)
```

```python
2. Pooling Layer: (None, 13, 13, 30)
```

----

## 4. Dropout Layer

  Pooling Layer 이후 일반적으로 Dropout Layer를 거친다. CNN의 기본 개념에서는 벗어나는 개념들이지만, 효율과 성능 개선을 위해 Pooling Layer와 Dropout Layer를 사용한다. 물론 설계자의 선택이다. Dropout은 캐나타 토론토 대학교 팀에서 발표하고 AlexNet에서 검증한 방식([관련논문](https://arxiv.org/abs/1207.0580))이다. 

  Dropout은 학습과정에서 네트워크에서 일부 노드를 생략하고 학습을 진행하는 방식이다. 일부 네트워크를 생략한다고 해서 학습에 영향을 끼치지 않는다. 오히려 각 네트워크마다 다른 망구조를 가진 모델 덕분에 Overfitting문제를 해소하고 연산에 대한 속도도 확보하게 된다. 

  ![](/assets/img/post/22.12.02/dropout.png){: #magnific}  

```python
def cnn_model():
    model = Sequential()   
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    print("1. Convolution Layer:", model.output_shape)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("2. Pooling Layer:", model.output_shape)    
    model.add(Conv2D(15, (3, 3), activation='relu'))
    print("3. Convolution Layer:", model.output_shape)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print("4. Pooling Layer:", model.output_shape)        
    model.add(Dropout(0.3))
    print("5. Dropout Layer:", model.output_shape)    
```

```python
1. Convolution Layer: (None, 26, 26, 30)
2. Pooling Layer: (None, 13, 13, 30)
3. Convolution Layer: (None, 11, 11, 15)
4. Pooling Layer: (None, 5, 5, 15)
5. Dropout Layer: (None, 5, 5, 15)
```

----

## 5. Flatten(Fully-Connected) Layer

  지금까지 위에서 진행되었던 성능/학습 효율화들의 기법을 통해서 'Input Data의 중요한 의미가 있는 부분들을 추출하는 작업'을 했다면, 마지막으로 해당 데이터가 어떤 class에 해당하는지 판단하는 작업을 진행한다. Dropout Layer를 거쳐나온 OutputData를 Flatten Layer에서 Shape을 변경하여 일반적인 DNN에서 사용하는 Softmax 등의 활성화 함수를 사용하여 최종 판별하게 된다.

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
```

```tex
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_7 (Conv2D)            (None, 26, 26, 30)        300       
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 13, 13, 30)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 11, 11, 15)        4065      
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 5, 5, 15)          0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 5, 5, 15)          0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 375)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 128)               48128     
_________________________________________________________________
dense_11 (Dense)             (None, 50)                6450      
_________________________________________________________________
dense_12 (Dense)             (None, 9)                 459       
=================================================================
Total params: 59,402
Trainable params: 59,402
Non-trainable params: 0
```

최종적으로 위와 같은 Output Shape과 Parameter값이 정리가 된다. 추가로 각 레이어의 Param값의 계산은 아래와 같다.

- **패딩과 스트라이드**
  - OH: 출력 데이터의 높이
  - H: 입력 데이터의 높이
  - P: 패딩
  - FH: 필터 높이
  - S: 스트라이드 

$$
OH = \frac{H+2P-FH}{S} + 1
$$



* **Convolution Layer**
  * W: Conv weight
  * K: 커널 size
  * C: 채널 수
  * N: 커널 수
  * B: Conv biase

$$
W = K^2 * C * N \\
B = N \\
O = W + B
$$



* **MaxPooling Layer**
  * O: 출력 데이터
  * I: 입력 데이터의 높이
  * Ps: 풀링 사이즈
  * S: 스트라이드 

$$
O = \frac{I - Ps}{S} + 1
$$



----

  이만 CNN 알고리즘의 정리를 마친다. 왜이렇게 정리가 힘들던지... 토이 프로젝트 정리하는 것보다 시간이 오래 걸렸다. 공부한지 시간도 지났고, 그 당시 개념만 대충 이해하고 구현 위주로 봤던 부분들을 하나 하나 다시 정리하려니... 그래도 뿌듯하다. 하지만 왜 이런걸 정리하려 하는지는 뚜렷한 목적성을 찾지 못했다. 그냥 하고 싶었다. 살면서 흔적을 남기는 일이랄까? 음... 다음에는 철지난 알고리즘 말고 최근 유행하는 것으로 정리해보아야겠다.