import sys
import os
import csv
import time
import cv2
import numpy as np
import tensorflow as tf
import cnn as nn

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', '/Users/junffy/Work/lf-autonomous-gimmic/detect_traffic_lights/train/train_data/train.txt', 'train')
flags.DEFINE_string('test', '/Users/junffy/Work/lf-autonomous-gimmic/detect_traffic_lights/train/test_data/test.txt', 'test')
flags.DEFINE_string('train_dir', '/Users/junffy/Work/lf-autonomous-gimmic/detect_traffic_lights/train/train.py', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 200, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')

if __name__ == '__main__':

    f = open(FLAGS.train, 'r')
    print(FLAGS.train)

    backup_dir = os.path.dirname(os.path.abspath(__file__)) + "/test_data/model"

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    train_image = []
    train_label = []

    for line in f:
        line = line.rstrip()
        l = line.split()

        print('loading '+l[0])
        img = cv2.imread(l[0])
        img = cv2.resize(img, (nn.IMAGE_SIZE, nn.IMAGE_SIZE))

        train_image.append(img.flatten().astype(np.float32)/255.0)

        tmp = np.zeros(nn.NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    print('---1---')
    print(train_image)
    print('---2---')
    print(train_label)
    print('---3---')
    f.close()

    f = open(FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (nn.IMAGE_SIZE, nn.IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(nn.NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    start = time.time()
    csvlist = []
    csvlist.append([])
    csvlist[0].append("step")
    csvlist[0].append("accuracy")
    csvlist[0].append("loss")

    with tf.Graph().as_default():

        images_placeholder = tf.placeholder("float", shape=(None, nn.IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, nn.NUM_CLASSES))
        keep_prob = tf.placeholder("float")

        logits = nn.inference(images_placeholder, keep_prob)
        loss_value = nn.loss(logits, labels_placeholder)
        train_op = nn.training(loss_value, FLAGS.learning_rate)
        acc = nn.accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # train
        for step in range(FLAGS.max_steps):

            #print("----- train max num------")
            print(len(train_image)/FLAGS.batch_size)
            for i in range(len(train_image)/FLAGS.batch_size):

                #print("----- train num------")
                #print(i)
                batch = FLAGS.batch_size*i
                sess.run(train_op, feed_dict={
                  images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                  labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                  keep_prob: 0.5})

            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            train_loss = sess.run(loss_value, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})

            print("step %d, training accuracy %g, loss %g"%(step, train_accuracy, train_loss))

            # add csv list for graph
            csvlist.append([])
            csvlist[step+1].append(step)
            csvlist[step+1].append(train_accuracy)
            csvlist[step+1].append(train_loss)

            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print("test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    # save model
    save_path = saver.save(sess, backup_dir + "/model.ckpt")

    # save graph data(csv)
    f = open('graph/graph.csv', 'w')
    dataWriter = csv.writer(f)
    dataWriter.writerows(csvlist)
    f.close()