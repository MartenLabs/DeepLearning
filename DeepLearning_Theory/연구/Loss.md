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

loss = tf.reduce_sum(loss + 1e-15, axis=-1)

print("BoxLoss: ", loss)

return loss

  
  

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

loss = positive_loss + negative_loss + 1e-15

loss = tf.reduce_sum(loss, axis=-1)

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

loss = alpha * tf.math.pow(1.0 - pt + 1e-4, self._gamma) * cross_entropy

loss = tf.reduce_sum(loss, axis=-1)

print("ClassificationLoss: ", loss.shape)

return loss

  

class Loss(tf.losses.Loss):

def __init__(self, num_classes=1, alpha=0.35, gamma=3.0, delta=2.0, neg_pos_ratio=5.0, hard_pos_ratio = 3.0):

super(Loss, self).__init__(reduction="auto", name="Loss")

# self._cls_loss = ClassificationLoss(alpha, gamma)

self._box_loss = BoxLoss(delta)

self._confidence_loss = ConfidenceLoss()

self._num_classes = num_classes

self._neg_pos_ratio = neg_pos_ratio

self._hard_pos_ratio = hard_pos_ratio

  

def call(self, y_true, y_pred):

y_pred = tf.cast(y_pred, dtype=tf.float32)

box_labels = y_true[:, :, :4]

box_predictions = y_pred[:, :, :4]

cls_labels = y_true[:, :, 4:]

cls_predictions = y_pred[:, :, 4:]

print("box_labels : ", box_labels.shape)

print("cls_predictions : ", cls_predictions.shape)

  

positive_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 1.0), dtype=tf.float32)

ignore_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 0.5), dtype=tf.float32)

negative_mask = tf.cast(tf.math.equal(y_true[:, :, 4], 0.0), dtype=tf.float32)

  

print("positive_mask: ", positive_mask.shape)

print("ignore_mask: ", ignore_mask.shape)

print("negative_mask: ", negative_mask.shape)

  

confidence_loss = self._confidence_loss(cls_labels, cls_predictions)

  

box_loss = self._box_loss(box_labels, box_predictions)

  

num_positives = tf.reduce_sum(positive_mask)

# Hard Positive Mining

positive_cls_loss = tf.where(tf.equal(positive_mask, 1.0), confidence_loss, 0.0)

num_hard_positives = tf.cast(self._hard_pos_ratio * num_positives, dtype=tf.int32)

positive_cls_loss = tf.reshape(positive_cls_loss, shape=(-1,))

top_k_positive_cls_loss, _ = tf.math.top_k(positive_cls_loss, k=num_hard_positives)

hard_positive_mask = tf.cast(positive_cls_loss >= top_k_positive_cls_loss[-1], dtype=tf.float32)

hard_positive_mask = tf.reshape(hard_positive_mask, shape=tf.shape(positive_mask)) # Precision 증가

print("hard_positive_mask: ", hard_positive_mask.shape)

  

confidence_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, confidence_loss)

box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

normalizer_positive = tf.reduce_sum(positive_mask, axis=-1) + 1e-4

normalizer_hard_positive = tf.reduce_sum(hard_positive_mask, axis=-1) + 1e-4

normalizer_hard_negative = tf.reduce_sum(negative_mask, axis=-1) + 1e-4

confidence_loss_positive = tf.math.divide_no_nan(tf.reduce_sum(confidence_loss * (hard_positive_mask + negative_mask), axis=-1), normalizer_hard_positive + normalizer_hard_negative)

box_loss_positive = tf.math.divide_no_nan(tf.reduce_sum(box_loss * positive_mask, axis=-1), normalizer_positive)

  

# box_loss_scaled = box_loss_positive / tf.reduce_mean(box_loss_positive)

# conf_loss_scaled = confidence_loss_positive / tf.reduce_mean(confidence_loss_positive)

  

loss = 0.1 * box_loss_positive + 1.0 * confidence_loss_positive

return loss