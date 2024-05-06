![](../../Data/Models/ObjectDetection/YoloV5/0.png)

![](../../Data/Models/ObjectDetection/YoloV5/1.jpg)

![](../../Data/Models/ObjectDetection/YoloV5/2.jpg)

![](../../Data/Models/ObjectDetection/YoloV5/3.jpg)

![](../../Data/Models/ObjectDetection/YoloV5/4.jpg)






## 조건부 Focal Loss 공식

수정된 Focal Loss 함수, $( \text{FL}_{\text{modified}}(y_{\text{true}}, p_t))$의 정의:

$$\text{FL}_{\text{modified}}(y_{\text{true}}, p_t) = \begin{cases}-\alpha_t (1 - p_t)^\gamma \log(p_t) & \text{만약 } (y_{\text{true}} = 1 \text{ 및 } p_t < \text{임계값}) \text{ 또는 } (y_{\text{true}} = 0 \text{ 및 } p_t \geq \text{임계값}) \\0 & \text{만약 } (y_{\text{true}} = 1 \text{ 및 } p_t \geq \text{임계값}) \text{ 또는 } (y_{\text{true}} = 0 \text{ 및 } p_t < \text{임계값}) \\\end{cases}$$

여기서:
- $p_t $는 $ $y_{\text{true}} = 1$일 때 예측 확률, $y_{\text{true}} = 0$일 때는 $1-\text{y_pred}$    
- \( \alpha_t \)는 클래스 가중치로, \( y_{\text{true}} = 1 \)일 때는 \( \alpha \), \( y_{\text{true}} = 0 \)일 때는 \( 1 - \alpha \)를 사용
- \( \gamma \)는 손실 함수의 스케일을 조정하는 매개변수

이 공식은 예측이 특정 임계값을 초과할 때 추가적인 손실을 부과하지 않음으로써 전통적인 Focal Loss 계산을 변경한다. 이 접근법은 모델이 고신뢰도 예측에 대해 과도하게 처벌받지 않도록 하여 오버피팅을 방지하고 모델의 일반화 능력을 향상시킬 수 있도록 돕는다.







Box Loss:
$$
L_{\text{box}} = \sum_{i}\begin{cases}
0.5 \times (y_{\text{true},i} - y_{\text{pred},i})^2 & \text{if } |y_{\text{true},i} - y_{\text{pred},i}| \leq \delta\\
|y_{\text{true},i} - y_{\text{pred},i}| - 0.5 \times \delta & \text{otherwise}
\end{cases}
$$

F1 Score Loss:
$$
\begin{align*}
\text{TP} &= \sum_i y_{\text{true},i} \times y_{\text{pred},i}\\
\text{FP} &= \sum_i (1 - y_{\text{true},i}) \times y_{\text{pred},i}\\
\text{FN} &= \sum_i y_{\text{true},i} \times (1 - y_{\text{pred},i})\\
\text{Precision} &= \frac{\text{TP}}{\text{TP} + \text{FP} + \epsilon}\\
\text{Recall} &= \frac{\text{TP}}{\text{TP} + \text{FN} + \epsilon}\\
\text{F1} &= \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}\\
L_{\text{f1}} &= 1 - \text{F1}
\end{align*}
$$

Classification Loss:
$$
L_{\text{cls}} = -\alpha_t \times (1 - p_t)^\gamma \times \log(p_t)
$$
where
$$
\alpha_t = \begin{cases}
\alpha & \text{if } y_{\text{true}} = 1\\
1 - \alpha & \text{otherwise}
\end{cases}
$$
and
$$
p_t = \begin{cases}
y_{\text{pred}} & \text{if } y_{\text{true}} = 1\\
1 - y_{\text{pred}} & \text{otherwise}
\end{cases}
$$

Final Loss:
$$
\begin{align*}
L_{\text{positive}} &= \frac{L_{\text{cls,pos}} + L_{\text{f1,pos}} + L_{\text{box,pos}}}{N_{\text{pos}}}\\
L_{\text{hard\_neg}} &= \frac{L_{\text{cls,hard\_neg}} + L_{\text{f1,hard\_neg}}}{N_{\text{hard\_neg}}}\\
L_{\text{cls}} &= \sqrt{L_{\text{cls,pos}} \times L_{\text{cls,hard\_neg}}}\\
L_{\text{f1}} &= \sqrt{L_{\text{f1,pos}} \times L_{\text{f1,hard\_neg}}}\\
L_{\text{combined}} &= \sqrt{L_{\text{cls}} \times L_{\text{f1}}}\\
L_{\text{final}} &= w_{\text{cls}} \times L_{\text{combined}} + w_{\text{box}} \times L_{\text{box,pos}}
\end{align*}
$$

- $y_{\text{true}}$: 실제 값 (박스 좌표 및 클래스 레이블)
- $y_{\text{pred}}$: 예측 값 (박스 좌표 및 클래스 확률)
- $\delta$: 박스 손실의 하이퍼파라미터
- $\epsilon$: F1 점수 계산에서 0으로 나누는 것을 방지하기 위한 작은 값
- $\alpha, \gamma$: 분류 손실의 하이퍼파라미터
- $N_{\text{pos}}$: 양성 샘플 수
- $N_{\text{hard\_neg}}$: Hard Negative 샘플 수
- $w_{\text{cls}}, w_{\text{box}}$: 분류 손실과 박스 손실의 가중치

이 손실 함수는 박스 좌표 예측, 클래스 분류, F1 점수를 모두 고려하며, Positive와 Hard Negative 샘플을 구분하여 다룬다. 또한 다양한 정규화와 가중치 기법을 사용하였다.