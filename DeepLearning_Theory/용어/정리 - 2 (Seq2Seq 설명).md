목차 

1. Seq2Seq
	- Embedding
	- Encoder
	- Decoder
	- 모델 구현 및 코드 설명



# Seq2Seq

### 1. Embedding

- **임베딩 행렬의 구조**
	- 임베딩 행렬의 각 행은 vocab에 있는 각 단어 또는 문자에 해당한다. 예를 들어 vocab에 '0', '1', ..., 'EOS'가 있으면 임베딩 행렬의 첫번째 행은 '0', 두번째 행은 '1'에 대응되는 output_dim 길이의 벡터를 가진다.

- **벡터의 의미**
	- 초기의 임베딩 행렬의 값들은 무작위로 설정된다. 그러나 훈련 과정중에 모델은 이 벡터들을 조정하며 각 단어 또는 문자가 주어진 작업에 대해 가장 유익한 방식으로 표현되도록 한다.

- **의미적 유사성**
	- 훈련이 진행됨에 따라, 의미적으로 비슷한 단어나 문자는 임베딩 공간에서 유사한 벡터를 찾게된다. 즉 그들의 벡터 표현이 서로 유사해진다.

- 차원의 크기 output_dim = 5는 임베딩 벡터의 차원(열의 길이)가 5라는 것을 의미한다. 이것은 각 단어 또는 문자를 5개 실수 값으로 표편한다는 것을 의미한다. 이 5차원 벡터는 해당 단어 또는 문자의 의미와 관련된 정보를 포함하게 된다.

**예시)**
'킹' 이라는 단어의 임베딩 벡터가 [1, 2, 3] 이라고 가정 
'여왕' 이라는 단어의 임베딩 벡터가 [1.1, 2.1, 2.9] 라고 가정 
'자동차' 라는 단어의 임베딩 벡터가 [5, 5, 5] 라고 가정 

위의 예에서 '킹'과 '여왕'의 벡터는 임베딩 공간에서 서로 가깝다. 이 두 벡터의 거리(유클리디안 거리)가 작기 때문, 반면 '킹'과 '자동차' 사이의 거리는 더 멀다 

즉 임베딩은 의미상 비슷한 단어나 항목을 공간에서 서로 가까운 위치로 맵핑하려고 하며 이는 임베딩 레이어가 데이터에서 단어나 항목 간의 관계를 학습하는 동안 자연스럽게 발생한다.


``` python
embedding_matrix:
[[0.1, 0.2, 0.3],
[0.4, 0.5, 0.6],
[0.7, 0.8, 0.9]
...
]


input_data: [2, 0, 1]

  
output:

[[0.7, 0.8, 0.9], # embedding matrix 의 2번째 행
[0.1, 0.2, 0.3], # embedding matrix 의 0번째 행
[0.4, 0.5, 0.6]] # embedding matrix 의 1번째 행
```






## 설명을 위한 예제

``` text
inverted_vocab
{0: '0',
 1: '1',
 2: '2',
 3: '3',
 4: '4',
 5: '5',
 6: '6',
 7: '7',
 8: '8',
 9: '9',
 10: '+',
 11: '-',
 12: 'PAD',
 13: 'EOS'}


(문제) train_data_bow
[12 12 12 5 0 10 9 13] // 50 + 9

(문제) train_shift_answer_bow 
[12. 5. 9. 13. 12.]    // 59

(정답) train_answer_onehot
[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]    # 5
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]    # 9
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]    # EOS
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]    # PAD
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]   # PAD
```

---
### 2. Encoder

입력 : train_data_bow: [12, 12, 12, 5, 0 ,10, 9, 13] 이 값은 '50 + 9'를 나타낸다. 그러나 이 숫자 시퀀스 앞부분에는 'PAD' (값이 12인 토큰)가 3개 있다. 이는 모든 입력 시퀀스의 길이를 동일하게 하기 위한 패딩이다.

1. **embedding**: 이 숫자 시퀀스는 임베딩 레이어를 통해 각 숫자를 해당하는 밀집 벡터로 변환한다.
2. **GRU**: 임베딩 레이어의 출력은 GRU 레이어에 전달된다. GRU는 입력 시퀀스를 순차적으로 처리하고 마지막 시간 단계의 은닉상태를 출력한다. 이 은닝 상태는 Context vector(z) 로 간주되며, 전체 입력 시퀀스의 정보를 포함하고 있다.


---
### 3. Decoder 

입력: train_shift_answer_bow: [12. 5. 9. 13. 12.] 이 입력은 '59'를 나타내며, 'EOS' 토근(값이 13인 토큰)으로 끝나는데, 이는 시퀀스의 끝을 나타낸다. 

1. **embedding**: 디코더의 입력 시퀀스도 임베딩 레이어를 통해 밀집 벡터로 변환된다.
2. **GRU**: 임베딩 레이어의 출력과 인코더의 context vector(z)가 디코더의 GRU 레이어에 전달되며 context vector는 디코더의 초기 상태로 설정된다.
3. **softmax** : GRU의 출력은 Dense 레이어에 전달되어 각 가능한 출력 토큰에 대한 확률 분포를 생성한다.

#### Decoder의 time step별 동작 

train_shift_answer_bow: [12. 5. 9. 13. 12.]

이는 우리가 예측하고자 하는 정답인 '59'를 나타낸다. 여기서 중요한 점은 디코더에게 입력으로 주어지는 데이터가 실제 답안의 **"이전"** 토큰임을 이해하는 것이다. 따라서, 주어진 시퀀스는 실제로 'PAD59EOS'를 나타낸다.

1. Time Step 1:
	- 입력: 'PAD' (12)
	- 초기 상태: 인코더의 컨텍스트 벡터 z
	- GRU 출력: 'PAD'에 대한 은닉 상태 
	- softmax 출력 : '5'의 확률이 가장 높음

2. Time Step 2:
	- 입력: '5' (학습시에는 실제 값인 5가 주어짐)
	- 초기 상테: 타임 스텝 1에서 GRU의 출력 상태
	- GRU 출력: '5'에 대한 은닉 상태 
	- softmax 출력: '9'의 확률이 가장 높음 

3. Time Step 3:
	- 입력: '9' (학습 시에는 실제 값인 9가 주어짐)
	- 초기 상태: 타임 스텝 2에서 GRU의 출력 상태
	- GRU 출력 : '9'에 대한 은닉상태
	- softmax 출력: 'EOS'의 확률이 가장 높음 

4. Time Step 4:
	- 입력: 'EOS' (13)
	- 초기 상태: 타임 스텝 3에서 GRU의 출력 상태
	- GRU 출력: 'EOS'에 대한 은닉 상태 
	- softmax 출력 'PAD'의 확률이 가장 높음 

5. Time Step 5:
	- 입력: 'PAD' (12)
	- 초기 상태: 타임 스텝 4에서 GRU의 출력 상태
	- GRU출력: 'PAD'에 대한 은닉 상태 
	- softmax 출력: 다음 예측값


이렇게 디코더는 각 타임 스텝에서 이전 타임 스텝의 출력 (또는 실제 값)과 이전 상태를 사용하여 다음 토큰을 예측한다. 학습 중에는 실제 값이 디코더의 입력으로 제공되며, 이를 "Teacher Forcing" 이라고 한다.

학습이 끝나면, 예측 시에는 디코더는 자신의 출력을 다음 타음 스텝의 입력으로 사용한다.


---
### 4. 학습

1. 인코더는 train_data_bow를 입력으로 받아 Context vector를 생성한다.
2. 디코더는 생성된 Context vector와 train_shift_answer_bow를 입력으로 받아 예측 시퀀스를 생성한다.
3. 이 예측 시퀀스는 train_answer_onehot 과 비교되어 손실이 계산된다.
4. 이 손실을 사용하여 모델의 가중치를 업데이트한다.

즉, 인코더는 입력 시퀀스를 Context vector로 변환하며, Decoder는 이 Context vector와 자신의 입력을 사용하여 출력 시퀀스를 생성한다. 이 출력 시퀀스가 실제 타겟 시퀀스와 비교외어 모델을 업데이트 한다.


---
### 5. Model  Implementation

- return_sequences = True
	- keras의 순환층(예: LSTM, GRU)에 사용되는 매개변수로 이 매개변수의 값에 따라 순환층이 반환하는 출력의 형태가 달라진다.

	- 순환층은 각 타임스텝에 대한 히든 상태를 반환한다. 
	- 만약 입력 시퀀스의 길이가 t라면 순환층의 출력은 (batch, t, hidden_units)의 형태를 가진다. 즉 각 입력 시퀀스 타임스텝에 대한 출력이 순차적으로 반환된다.
	- 주로 시퀀스 출력이 필요한 경우나 다른 순환층과 연결될 때 사용된다.
	
	- Seq2Seq, 다층 순환 신경망(Stacked RNN), 양방향 순환 신경망(Bidirectional RNN), Attention Mechanism, etc...

- return_sequences = False (기본값)
	- 순환층은 마지막 타임스텝에 대한 히든 상태만 반환한다.
	- 출력의 형태는 (batch, hidden_units)
	- 주로 전체 시퀀스 정보를 요약하고자 할 때나 다음층이 순환층이 아닐 때 사용한다.

- 타임스텝(time step)
	- 시퀀스 데이터에서의 각 요소나 시점을 의미 
	- 시퀀스 데이터를 생각할 때, 각 데이터 포인트가 어떤 순서에 따라 발생하는 경우, 그 순서의 각 위치를 타임스탭이라고 한다. 

	  ex)
	  문장 : "안녕하세요"

	  타임스텝 1: "안"
	  타임스텝 2: "녕"
	  타임스텝 3: "하"
	  타임스텝 4: "세"
	  타임스텝 5: "요"


- initial_state = context_vector 
	- context_vector는 인코더의 마지막 히든 상태이다.

	- seq2seq 모델에서 인코더의 주요 목적은 입력 시퀀스를 하나의 고정된 크기의 벡터, 즉 context vector로 압축하는 것이다. 이 벡터는 입력 시퀀스의 전체 정보를 포함하게 된다.

	- 디코더는 이 context vector를 사용해서 원하는 출력 시퀀스를 생성한다. 디코더의 초기 상태로 인코더의 context vector를 설정함으로써, 인코더에서 추출한 정보가 디코더로 전달된다. 이렇게 해서 디코더는 입력 시퀀스에 기반한 적절한 출력 시퀀스를 생성할 수 있다.

	- 간단한 예로, 번역모델의 인코더는 영어 문장을 context vector로 표현하고, 디코더는 그 context vector를 사용해 해당 영어 문장의 한국어 번역을 생성한다. 여기서 context vector는 영어 문장의 전체적인 의미를 포함하며, 디코더는 그 정보를 활용해서 한국어로 번역된 문장을 생성한다.

	- 즉 seq2seq 모델에서 인코더의 출력 (context vector)을 디코더의 초기 상태로 설정해야 인코더에서 얻은 정보를 디코더로 전달해 줄 수 있다. initial_state = context_vector를 통해 디코더의 초기 상태로 설정하게 되면, 디코더는 해당 context_vector에 포함된 정보를 기반으로 출력 시퀀스를 생성하기 시작한다.

``` python
from keras.layers import Input, Embedding, GRU
from keras.models import Model 


def seq2seq():
	 
	# 모델용 encoder
	# [12 12 12 9 6 10 6 13] (train_data_bow)
	encoder_input = Input(shape = (8, ))
	embedding = Embedding(len(vocab), output_dim = 5) # (14, 5)
	x = embedding(encoder_input) # x = (8, 5)
	context_vector = GRU(units = 16)(x)

	encoder = Model(encoder_input, context_vector)
	
	# 모델용 decoder 
	# [12. 1. 0. 2. 13.] (train_shift_answer_bow)s
	decoder_input = Input(shape = (5, ))
	y = embedding(decoder_input)	
	gru = GRU(units=16, return_sequences=True) # 타임스텝 출력이 순차적으로 반환	
	y = gru(y, initial_state = context_vector)

	softmax = Dense(units = len(vocab), activation = 'sigmoid')
	y = softmax(y)

	# 디코더 내부 설정
	"""디코더의 각 타임스텝에서 입력될 하나의 토큰을 나타내는 입력"""
	# 자기 자신의 출력물을 다음 네트워크가 받을 수 있도록 하기 위한 변수 (ex 'PAD')
	next_decoder_input = Input(shape = (1, )) 
	# 토큰을 임베딩 변환(1, 5)
	next_decoder_embedded = embedding(next_decoder_input) 
	decoder_initial_state = Input(shape = (16, )) # context_vector shape

	decoder_gru_output = gru(next_decoder_embedded, initial_state = decoder_initial_state)
	decoder_softmax_output = softmax(decoder_gru_output)

	decoder = Model([next_decoder_input, decoder_initial_state], [decoder_softmax_output, decoder_gru_output])

	# 인코더 디코더 연결용 모델 (큰 틀)
	model = Model([encoder_input, decoder_input], y)
	model.complie(loss = 'categorical_crossentropy',
				  optimizer = 'adam',
				  metrics = ['accuracy'])



	return model, encoder, decoder
```

디코더의 입력:
1. next_decoder_input : 이는 디코더의 다음 타임스텝에 들어갈 입력값으로 seq2seq 모델에서의 디코더는 이전 타임스텝의 출력을 현재 타임스텝의 입력으로 사용하기 때문에 이 변수가 필요하다.

2. decoder_initial_state: 이는 디코더의 GRU초기 상태로 사용된다. 일반적으로 seq2seq 모델에서는 인코더의 마지막 상태, 즉 context_vector가 이 초기 상태로 사용된다. 그러나 훈련된 인코더-디코더 모델을 사용하여 새로운 시퀀스를 생성할 때는 디코더의 이전 타임스텝의 상태를 이 초기 상태로 사용하게 된다.


디코더의 출력:
1. decoder_softmax_output: 이는 디코더의 GRU출력을 softmax 활성화 함수를 통과시킨 결과이다. 이 출력은 각 단어의 확률 분포를 나타내므로 가장 확률이 높은 단어를 선택하여 다음 타임스텝의 입력으로 사용할 수 있다.

2. decoder_gru_output: 이는 디코더의 GRU층의 직접적인 출력이다. 이 값은 다음 타임스텝의 초기 상태로 사용될 수 있다.

요약: 디코더는 next_decoder_input과 decoder_initial_state를 입력으로 받아, 그에 해당하는 softmax 확률 분포와 다음상태를 출력한다.


