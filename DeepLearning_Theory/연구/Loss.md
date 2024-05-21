


``` python
class BoxLoss(tf.losses.Loss):

def __init__(self, delta):

super(BoxLoss, self).__init__(reduction="none", name="BoxLoss")

self._delta = delta

  

def call(self, y_true, y_pred):

difference = y_true - y_pred

absolute_difference = tf.abs(difference)

squared_difference = difference ** 2

loss = tf.where(

tf.less(absolute_difference, self._delta),

0.5 * squared_difference,

absolute_difference - 0.5 * self._delta

)

loss = tf.reduce_sum(loss, axis=-1)

print("BoxLoss: ", loss)

return loss

  
  

class F1ScoreLoss(tf.losses.Loss):

def __init__(self, epsilon=1e-7):

super(F1ScoreLoss, self).__init__(reduction="none", name="F1ScoreLoss")

self._epsilon = epsilon

def call(self, y_true, y_pred):

y_true = tf.cast(y_true, dtype=tf.float32)

y_pred = tf.nn.sigmoid(y_pred)

  

tp = tf.reduce_sum(y_true * y_pred, axis=-1)

fp = tf.reduce_sum((1 - y_true) * y_pred, axis=-1)

fn = tf.reduce_sum(y_true * (1 - y_pred), axis=-1)

  

precision = tp / (tp + fp + self._epsilon)

recall = tp / (tp + fn + self._epsilon)

  

f1_score = 2 * precision * recall / (precision + recall + self._epsilon)

return 1 - f1_score

  
  

class FBetaLoss(tf.keras.losses.Loss):

def __init__(self, beta=2.0, threshold=0.5, **kwargs):

super(FBetaLoss, self).__init__(**kwargs)

self.beta = beta

self.threshold = threshold

  

def call(self, y_true, y_pred):

y_true = tf.cast(y_true, tf.float32)

y_pred = tf.nn.sigmoid(y_pred)

  

y_true_binary = tf.cast(tf.math.equal(y_true, 1.0), dtype=tf.float32)

y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)

  

tp = tf.reduce_sum(y_true_binary * y_pred_binary, axis=-1)

fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary, axis=-1)

fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary), axis=-1)

  

precision = tp / (tp + fp + tf.keras.backend.epsilon())

recall = tp / (tp + fn + tf.keras.backend.epsilon())

  

fbeta_score = (1 + self.beta ** 2) * (precision * recall) / (self.beta ** 2 * precision + recall + tf.keras.backend.epsilon())

return 1 - fbeta_score

  

# alpha : Recall에 영향 Beta: Precision에 영향

class ConfidenceLoss(tf.losses.Loss):

def __init__(self, alpha=3.0, beta=1.25):

super(ConfidenceLoss, self).__init__(reduction="none", name="ConfidenceLoss")

self._alpha = alpha

self._beta = beta

  

def call(self, y_true, y_pred):

# iou = y_true

confidence = tf.nn.sigmoid(y_pred)

positive_mask = tf.cast(tf.greater(y_true, 0.0), dtype=tf.float32)

negative_mask = 1.0 - positive_mask

positive_loss = -self._alpha * positive_mask * tf.math.log(confidence + 1e-15)

negative_loss = -self._beta * negative_mask * tf.math.log(1.0 - confidence + 1e-15)

loss = positive_loss + negative_loss

loss = tf.reduce_sum(loss, axis=-1)

return loss

class SquaredIoULoss(tf.losses.Loss):

def __init__(self, delta):

super(SquaredIoULoss, self).__init__(reduction="none", name="SquaredIoULoss")

self._delta = delta

  

def call(self, y_true, y_pred):

difference = y_true - y_pred

absolute_difference = tf.abs(difference)

squared_difference = difference ** 2

loss = tf.where(

tf.less(absolute_difference, self._delta),

0.5 * squared_difference,

absolute_difference - 0.5 * self._delta

)

loss = tf.reduce_sum(loss, axis=-1)

print("SquaredIoULoss: ", loss)

return loss

# Alpha 값을 높이면 Recall 증가

# Alpha 값을 낮추면 Precision 증가

# Gamma 값이 커질수록 잘못 분류된 예제에 더 큰 가중치 부여

class ClassificationLoss(tf.losses.Loss):

def __init__(self, alpha, gamma):

super(ClassificationLoss, self).__init__(reduction="none", name="ClassificationLoss")

self._alpha = alpha

self._gamma = gamma

  

def call(self, y_true, y_pred):

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

probs = tf.nn.sigmoid(y_pred)

alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))

pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)

loss = alpha * tf.math.pow(1.0 - pt + 1e-6, self._gamma) * cross_entropy

loss = tf.reduce_sum(loss, axis=-1)

print("ClassificationLoss: ", loss.shape)

return loss

  
  

class Loss(tf.losses.Loss):

def __init__(self, num_classes=1, alpha=0.5, gamma=4.0, delta=2.0, neg_pos_ratio=3.0, hard_pos_ratio=3.0):

super(Loss, self).__init__(reduction="auto", name="Loss")

self._cls_loss = ClassificationLoss(alpha, gamma)

self._box_loss = BoxLoss(delta)

self._confidence_loss = ConfidenceLoss()

self._score_loss = SquaredIoULoss(delta)

self._num_classes = num_classes

self._neg_pos_ratio = neg_pos_ratio

self._hard_pos_ratio = hard_pos_ratio

  

def call(self, y_true, y_pred):

y_pred = tf.cast(y_pred, dtype=tf.float32)

box_labels = y_true[:, :, :4]

box_predictions = y_pred[:, :, :4]

cls_labels = y_true[:, :, 4:5]

cls_predictions = y_pred[:, :, 4:5]

  

score_labels = y_true[:, :, 5:]

score_predictions = y_pred[:, :, 5:]

  

print("box_labels : ", box_labels.shape)

print("cls_predictions : ", cls_predictions.shape)

  

positive_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 1.0), dtype=tf.float32)

ignore_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 0.5), dtype=tf.float32)

negative_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 0.0), dtype=tf.float32)

  

print("positive_mask: ", positive_mask.shape)

print("ignore_mask: ", ignore_mask.shape)

print("negative_mask: ", negative_mask.shape)

  

cls_loss = self._cls_loss(cls_labels, cls_predictions)

confidence_loss = self._confidence_loss(cls_labels, cls_predictions)

score_loss = self._score_loss(score_labels, score_predictions)

box_loss = self._box_loss(box_labels, box_predictions)

  

# Hard Negative Mining

negative_cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, confidence_loss * negative_mask)

num_positives = tf.reduce_sum(positive_mask)

num_hard_negatives = tf.cast(self._neg_pos_ratio * num_positives, dtype=tf.int32)

negative_cls_loss = tf.reshape(negative_cls_loss, shape=(-1,))

top_k_negative_cls_loss, _ = tf.math.top_k(negative_cls_loss, k=num_hard_negatives)

hard_negative_mask = tf.cast(negative_cls_loss >= top_k_negative_cls_loss[-1], dtype=tf.float32)

hard_negative_mask = tf.reshape(hard_negative_mask, shape=tf.shape(positive_mask))

  

print("hard_negative_mask: ", hard_negative_mask.shape)

  

# Hard Positive Mining

positive_cls_loss = tf.where(tf.equal(positive_mask, 1.0), cls_loss, 0.0)

num_hard_positives = tf.cast(self._hard_pos_ratio * num_positives, dtype=tf.int32)

positive_cls_loss = tf.reshape(positive_cls_loss, shape=(-1,))

top_k_positive_cls_loss, _ = tf.math.top_k(positive_cls_loss, k=num_hard_positives)

hard_positive_mask = tf.cast(positive_cls_loss >= top_k_positive_cls_loss[-1], dtype=tf.float32)

hard_positive_mask = tf.reshape(hard_positive_mask, shape=tf.shape(positive_mask)) # Precision 증가

  

print("hard_positive_mask: ", hard_positive_mask.shape)

  

box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

  

normalizer_positive = tf.reduce_sum(positive_mask, axis=-1) + 1e-6

normalizer_negative = tf.reduce_sum(negative_mask, axis=-1) + 1e-6

normalizer_hard_negative = tf.reduce_sum(hard_negative_mask, axis=-1) + 1e-6

normalizer_hard_positive = tf.reduce_sum(hard_positive_mask, axis=-1) + 1e-6

  

cls_loss_positive = tf.math.divide_no_nan(tf.reduce_sum(cls_loss * (negative_mask + positive_mask), axis=-1), normalizer_negative + normalizer_positive)

confidence_loss_positive = tf.math.divide_no_nan(tf.reduce_sum(confidence_loss * (hard_negative_mask + hard_positive_mask), axis=-1), normalizer_hard_negative + normalizer_hard_positive)

score_loss = tf.math.divide_no_nan(tf.reduce_sum(score_loss * (positive_mask), axis=-1), normalizer_positive)

  

box_loss_positive = tf.math.divide_no_nan(tf.reduce_sum(box_loss * (positive_mask), axis=-1), normalizer_positive)

  

loss = box_loss_positive + cls_loss_positive + confidence_loss_positive + score_loss

return loss
```