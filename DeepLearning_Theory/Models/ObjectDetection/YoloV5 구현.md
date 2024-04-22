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