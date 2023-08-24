
## 딥러닝 용어 정리



- Channel
- Feature map
- Convolution
- Kernel size
- Stride
- Padding 
- Convolution layer
- Activation function
- Pooling
- Flatten layer
- Dense layer
- CNN
- Feature extraction
- Classification
- 회귀, 분류 

- Loss Function
	- 회귀 - MAE, MSE, RMSE
	- 분류 - Binary Cross Entropy, Categorical Cross Entropy

- Optimizer(최적화 함수)
	- 하강법 
	- 여러가지 촤적화 방법 

- Metrics
---

### channel

``` txt
color 이미지는 RGB 3채널 (width * hight * channel)
  
channel의 갯수 = depth
```

![Alt text](https://velog.velcdn.com/images/jee-9/post/78f027ba-832d-4015-b743-0a42bb049714/image.png)

---


### feature map

``` txt
입력으로부터 커넗을 사용하여 합성곱(Convolution)연산을 통해 나온 결과


ㅁ 목적

각각의 특징들을 패턴으로 읽어내는 것이 목적.

말 그대로 특징을 잡아내는 맵
```

![Conv Image](https://drive.google.com/uc?id=1t6R7jwoHzvC-ycuACXjFKgsXvzjLectr)

  

![Alt text](https://velog.velcdn.com/images/jee-9/post/b229fc16-d83b-4b0c-ba1d-07d7e39c2e87/image.png)

  ---
  

### feature map 크기


$N \times N$ 입력 이미지와 $F \times F$ 필터일 경우, feature map의 크기 $O \times O$ 는 다음과 같이 결정된다.

  

$$ O = {{(N+2P-F)}\over S} + 1 \quad (P : padding size, S : stride)$$

---


### Convolution

``` txt
이미지가 들어왔을 때 특징을 뽑아내는 과정
```

![Alt text](https://velog.velcdn.com/images/seongguk/post/fe454740-904a-4d30-8be4-23652a483f9a/1_1okwhewf5KCtIPaFib4XaA.gif)


  ---
  

### Kernel Size

``` txt
kernel size는 convolution의 시야(view)를 결정 

  

보통 2D에서 3x3 pixel로 사용
```

  ---
  

### stride

``` txt
이미지를 횡단할 때 커널의 스텝 사이즈를 결정

  
기본값은 1이지만 Max Pooling과 비슷하게 이미지를 다운샘플링 하기 위해 Stride를 2로 사용.
```

![Alt text](https://velog.velcdn.com/images/seongguk/post/70901c25-2ecd-4a3b-b042-e7e70c74f1c0/image.png)


---


### Padding

``` txt
샘플 테두리를 어떻게 조절할지 결정

  

Padding된 Convolution은 input과 동일한 output차원을 유지하는 반면, 패딩되지 않은

Convolution은 커널이 1보다 큰 경우 테두리의 일부를 제거 가능
```

![Alt text](https://velog.velcdn.com/images/seongguk/post/c1428bac-0ec3-4cf2-88af-a783094cee6a/image.png)

  
  ---
  

### Convolution Layer

``` txt
전체가 아닌 일부분만 빨간색 테두리(필터 혹은 커널)를 통해 가중치와 입력값을 곱한 값에 활성화 함수를 취하여 특징을 뽑는 과정.
```

![Alt text](https://velog.velcdn.com/images/seongguk/post/c31cb135-0601-4b1d-aa48-7291ee540ac9/img.gif)

  

![Alt text](https://velog.velcdn.com/images/seongguk/post/ffc1e9c6-e5cb-408c-b1f7-779e373fbc32/image.png)

  
  ---
  

### 활성화 함수(Activation Function)

``` txt
필터 들을 통해 feature map이 추출되면 이 feature map에 활성화 함수를 적용하여 값을 활성화


있다, 없다 를 구분하기 위해 활성화 함수를 사용

  
  
ㅁ 종류 및 특징

- Sigmoid

모든 입력값에 대해 0과 1사이로 변환하는 역할.

0.5를 기준으로 0.5 이하면 0, 초과면 1로 변환하여 두 가지 클래스를 분류하는

이진분류(binary classification) 문제에도 활용할 수 있다.

  

- tanh

시그모이드 함수와 형태는 유사하지만 -1과 1사이의 값을 취할 수 있어 0과 음수값을 갖을 수 있다.

또한 0부근에서 시그모이드 보다 더 가파른 기울기를 갖는 다는 것이 차이점 이다.
  
  

- ReLu (가장 많이 쓰임)

두개의 직선을 이어 만든 것으로 비선형 함수이지만 선형과 매우 유사한 성질을 가지고 있다.

따라서 계산이 쉽고 미분도 쉽게 풀 수 있다. 또한 가장 중요한 점은 모델 최적화와 유용한 성질들을 보존하고 있어서 최적화를 쉽게 할 수 있다.

  
  
- softamx (분류문제에 최적화 된 함수)

소프트맥스의 모든 성분의 합은 항상 1이다. 따라서 각 성분을 확률처럼 사용할 수 있는 중요한 특징을 가지고 있다. 마지막 출력단에 많이 쓰임
```


---


### pooling

``` txt
ㅁ pooling 의 역할

차례대로 처리되는 데이터의 크기를 줄인다. 이 과정으로 모델 전체 매개변수의 수를 줄일 수 있다. 즉 데이터의 공간적인 특성을 유지하면서 크기를 줄여주는 층이며, 특정 위치에서 큰 역할을 하거나, 전체를 대변할 수 있는 특징을 추출 가능 하다

  

ㅁ pooling을 거치는 이유

더 높은 정확도를 위해서는 필터가 많아야 하는데, 필터가 늘어날 수록 Feature Map이 늘어난다. 이는 딥러닝 모델의 Dimension이 늘어난다는 것이고, High Dimension 모델은 그만큼 파라미터의 수 또한 늘어난다. 이는 Over fitting의 문제점 뿐만 아니라, 모델의 사이즈와 레이턴시에도 큰 영향을 끼친다.
```

  

- #### Max Pooling

![Alt text](https://velog.velcdn.com/images/seongguk/post/adafefcc-5241-4fd3-976b-ef973be1fc7e/full_padding_no_strides.gif)

  
  

- #### Average Pooling

![Alt text](https://velog.velcdn.com/images/seongguk/post/5290d7d2-7088-4139-9aa9-4cfd782c154f/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.gif)


---
  

### Flatten Latyer

``` txt

추출된 주요 특징을 전결합층에 전달하기 위해 1차원 자료로 바꿔주는 layer이다.

이미지 형태의 데이터를 배열형태로 Flatten하게 만들어준다

batch size에 영향을 주지 않는다.

  

CNN에서 Cov layer나 Max pooling 레이어를 반복적으로 거치면 주요 특징만 추출되고 추출된 주요 특징은 전결합층에 전달되어 학습된다. Conv Layer나 Max Pooling 레이어는 주로 2차원 자료를 다루지만 전결합층에 전달하기 위해서는 1차원 자료로 바꿔주는 것이 필요한다 이 때 사용되는 것이 flatten layer이다.

  

Pooling을 거쳐 만들어진 최종적인 Convolution layer를 Neural Network의 Input으로 사용하기 위해 행렬이 아닌 배열로만들어주는 과정이 필요한데 Flatten Layer가 이 역할을 한다. 이 과정까지 거치면 처음 Input에서 특징만 추출한 최종적인 Layer가 완성되고, 이를 Neural Network에 적용해 최종적으로 우리가 원하는 분류문제를 수행하게 된다.

```

  
  ---
  

### Dense Layer (Fully Connected Layer)

``` txt
신경망 모델을 구성하는 주요한 요소이다.

추출된 정보들을 하나의 레이어로 모으고, 원하는 차원으로 축소시켜 표현하기 위한 레이어

  

Dense Layer의 각 뉴런이 이전 계층의 모든 뉴런으로부터 입력을 받게 됨(Fully Connected)

  

다층 퍼셉트론 신경망에서 사용되는 레이어로 입력과 출력을 모두 연결해줌

각 연결선은 가중차(Weight)을 포함하고 있는데 연결 강도를 의미하며 가중치가 높을 수록 해당 입력 뉴런이 출력뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 작다
```

![Dense Layer](https://blog.kakaocdn.net/dn/bIR2pA/btqS3Is70ge/ErOs4occOqVpv8JLQDhejk/img.png)

  
---


### CNN

``` txt
기본적으로 CNN은 Input -> Padding -> Convolution Layer -> Pooling -> Convolution Layer -> Pooling -> Convolution Layer ->..............-> Flattening -> Output
```

![CNN Image](https://drive.google.com/uc?id=1WyHcyDpVbXehY_y-GVu2TzvEWuUCAh3I)
  
![CNN Image](https://blog.kakaocdn.net/dn/4Ut0l/btqS1VfsvQ5/S2HuIT5GkW49Z8g2xSKX0k/img.png)

![CNN Image](https://blog.kakaocdn.net/dn/qoIrV/btqS1Wegnxz/5nVOKj1eno2issrgfSW3p0/img.png)

  
---


### Feature Extraction

``` txt
입력 데이터의 고유한 특징을 찾아가는 단계
```

---

  
### Classification

``` txt
찾은 특징을 가지고 class를 고르는 단계
```


---
  

### 회귀와 분류

``` txt
ㅁ 회귀

회귀 문제는 우리가 원한느 결과값이 연속적인 변수인 것을 예측하는 문제다. 예를 들어 집값 예측, 온도 예측이 있다.

  

ㅁ 분류

분류 문제는 우리가 원하는 결과값이 클래스라고 하는 유한한 모임으로 분류되는 문제이다. 예를 들어 질병 예측, 만족도 예측이 있다. 여기서 만족도를 1, 2, 3이라고 구분 짓는 것을 라벨링 이라고 하며 1, 2, 3 숫자들을 라벨링 이라고 한다. 만약 1,2, 3을 0과 1로만 구성된 벡터 (1,0,0), (0,1,0), (0,0,1) 로 표현하는 방법을 원-핫 인코딩(one-hot encoding)이라고 부르며 표현된 벡터를 원-핫 벡터(one-hot vector)라고 한다.
```

![Image](https://drive.google.com/uc?id=1EhTdjVhhQLLB9FFb0FzSXIqdbvzHa3w_)


  ---


### Loss Function (손실함수)

``` txt
모델 최적화에 사용되는 목적 함수
```

| Loss Function | Description |
|:---:|:---:|
| 회귀 | MSE, MAE, Huber, Log-Cosh |
| 분류 | Binary Cross Entropy, Categorical Cross Entropy, Sparse Categorical Cross Entropy, Hinge, Squared Hinge, Kullback Leibler Divergence, Poisson, Cosine Proximity |


### 회귀

- #### MAE (Mean Absolute Error)
$$ L{(y, \hat{y})} = {1\over n} \sum_{i=1}^{n}{\vert \hat y_i - y_i \vert}$$
``` txt
평균 절대 오차

회귀 문제를 위한 기본적인 손실 함수로서 예측값 y^과 실제값 y의 수직 거리의 평균으로 표현된 오차.

실제값과 예측값의 차인 y-y^ 을 잔차(residual)라고 한다. (즉, 수직거리는 잔차의 크기다.)
```
  

- #### MSE (Mean Squared Error)
$$ L{(y, \hat{y})} = {1\over n} \sum_{i=1}^{n}{(\hat y_i - y_i)^2}$$
``` txt
평균 제곱 오차

회귀 문제를 위한 기본적인 손실 함수로서 예측값 y^과 실제값 y의 수직 거리의 제곱의 평균으로 표현된 오차.

MAE와 다르게 미분가능한 장점이 있지만 제곱식이 있기 때문에 전체 데이터의 분포에서 멀리 떨어져 있는 이상치(outlier)가 많은 경우에는 MAE보다 성능이 좋지 않을 수 있다.
```

  
- #### RMSE (Root Mean Squared Error)
$$ L{(y, \hat{y})} = \sqrt{{1\over n} \sum_{i=1}^{n}{(\hat y_i - y_i)^2}}$$
``` txt
평균 제곱근 오차

MSE는 원래 단위를 제곱하여 결과를 내기 때문에 같은 단위 크기로 만들어 주기 위해 MSE에 제곱근을 씌운 손실 함수이다.
```




### 분류

- #### 이진 교차 엔트로피 함수(Binary Cross Entropy)
$$ L{(y, \hat{y})} = - {1\over n} \sum_{i=1}^{n}{(y_i \log(\hat{y_i}) + (1 - y_i) \log(1-\hat{y_i}) )} \quad (log는 자연로그)$$

``` txt
이진 분류 문제를 위한 교차 엔트로피 함수이다.


이진 분류는 두 가지 상황을 분류하는 문제로서 원-핫 벡터로 표현하지 않아도 된다. 예를 들어 자동차 보유를 분류하는 문제라면 미보유는 0, 보유는 1로 표현하여 벡터((1,0), (0,1))를 사용하지 않고 표현할 수 있다.
```

  

- #### 범주형 교차 엔트로피 함수(Categorical Cross Entropy)
$$ Loss = -\frac{1}{N}\sum_{j=1}^{N}\sum_{i=1}^{C}t_{ij}\log(y_{ij})$$

``` txt
- CCE 는 클래스가 3개 이상인 데이터를 대상으로 사용하는 손실함수이다.

- CCE 는 주로 softmax 함수를 활성화 함수로 사용한다.

- 출력층의 노드 수는 클래스의 수와 동일하다.

- 실제 데이터인 라벨은 원-핫 벡터로 구성되어 있다.

- 출력된 벡터는 각 클래스에 속할 확률이 나오며, 총합은 1이다.
```


---


### Optimizer(최적화 함수)

``` txt
- 최적화 개념 
딥러닝 분야에서 최적화란 손실함수(Loss Function)값을 최소화하는 파라미터를 구성하는 과정.

학습 데이터를 입력하여 네트워크 구조를 거쳐 예측값 (y^)을 얻게되는데 이 예측값과 실제 답(y)과의 차이를 비교하는 함수가 손실함수.

즉 모델이 에측한 값과 실제 값의 차이를 최소화 하는 네트워크 구조의 파라미터를 찾는 과정이 최적화.
```

$$min_{w} L(y, \hat{y}) $$

- ####  하강법
$$ w \leftarrow w + \mu\Delta w, \nabla f(w)^T \Delta w < 0    \quad \\ $$$$ \mu = 학습률(learning rate, step size), \quad \Delta w = 탐색 방향$$
``` txt
- 탐색 방법 Delta w에 따라 하강법이 정해진다
- 경사하강법, 뉴턴 방법 등
```

	- Gradient Descent (경사 하강법)
		w ← w − μ ▿ f(w)

	- SGD Stochastic Gradient Descent (확률적 경사 하강법)
		w ← w − μ ▿ L(w)

 ![Image](https://drive.google.com/uc?id=1Pf7--2Mir_jxkvaamd6-IiumjUpRpppl)



- #### 여러가지 최적화 방법 
![Image](https://drive.google.com/uc?id=1Db7N4yUhllfVellA-Zpv1bUSbTbABGCK)

	- 스케줄링 - learning rate을 조절하자!
		- StepLR
		- ExponentialLR
		- Cosine Annealing 등등

---


### Metrics (평가지표)
``` txt
- 훈련을 모니터링 하기 위해 사용
- 분류에서는 accuracy 사용
```