import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
mnist = input_data.read_data_sets('mnist', one_hot=True)
bach_size = 80
n_classes = 10
X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')
dropout = 0.8
keep_prob = tf.placeholder('float')

def convnet(x):
    weights = {'conv_1': tf.Variable(tf.random_normal([6, 6, 1, 32])),
               'conv_2': tf.Variable(tf.random_normal([6, 6, 32, 64])),
               'conv_3': tf.Variable(tf.random_normal([6, 6, 64, 128])),
               'w_fc': tf.Variable(tf.random_normal([4 * 4 * 128, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))
               }

    biases = {"conv_1": tf.Variable(tf.random_normal([32])),
              "conv_2": tf.Variable(tf.random_normal([64])),
              "conv_3": tf.Variable(tf.random_normal([128])),
              "w_fc": tf.Variable(tf.random_normal([1024])),
              "out": tf.Variable(tf.random_normal([n_classes]))
              }

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    l1 = tf.nn.conv2d(x, weights['conv_1'], strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.relu(l1 + biases["conv_1"])
    l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l2 = tf.nn.conv2d(l1, weights['conv_2'], strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(l2 + biases["conv_2"])
    l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.nn.conv2d(l2, weights['conv_3'], strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.relu(l3 + biases["conv_3"])
    l3 = tf.nn.max_pool(l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(l3)
    fc = tf.reshape(l3, [-1, 4 * 4 * 128])
    fc = tf.matmul(fc, weights['w_fc'])
    fc = tf.nn.relu(fc + biases["w_fc"])
    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['out']) + biases["out"]
    print(output)
    return output


cam = cv2.VideoCapture(0)
# image input function for real-time testing
def imagefile():
    rect, img = cam.read()
    cv2.rectangle(img, (150, 170), (300, 350), (255, 255, 255), 2)
    img_crop = img[170:350, 150:300]
    cv2.imshow('img', img)
    res = cv2.resize(img_crop, (28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('croped', img_crop)
    cv2.imshow('28*28', res)
    res1 = list(res)
    grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (1, 1), 0)
    (thres, im_bw) = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    tva = [(255 - h) * 1 / 255 for h in res1]
    cv2.imshow('b&w', im_bw)
    tva = np.reshape(tva, [-1, 784])
    cv2.waitKey(100)
    return np.reshape(im_bw, [-1, 784])


def train(X):
    output = convnet(X)
    cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)))
    optimiser = tf.train.AdamOptimizer().minimize(cost)
    hm_epoch = 10
    with tf.Session() as S:
        S.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / bach_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size=bach_size)
                # epoch_x = tf.reshape(epoch_x, shape=[-1, 28, 28, 1])
                # epoch_y = tf.reshape(epoch_y, shape=[-1, 28, 28, 1])
                _, c = S.run([optimiser, cost], feed_dict={X: epoch_x, y: epoch_y})
                epoch_loss += c
                print('Epoch', epoch, 'Complete out of', hm_epoch, 'loss= ', epoch_loss)
            corect = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(corect, 'float'))
            print('Accuracy:-')
            print('ACCURACY := ', accuracy.eval({X: mnist.test.images, y: mnist.test.labels}))

        while True:
            pred = tf.argmax(output, 1)
            value = pred.eval(feed_dict={X: imagefile(), keep_prob: dropout})
            print(value)


train(X)
