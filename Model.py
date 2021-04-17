
import tensorflow as tf
import time
import tools
import os
import numpy as np

from os import listdir
from os.path import isfile, join
from PIL import Image


class classifier:




    def __init__(self,train_path='./dataset/archive/train images 13440x32x32/train/',batch=1,epoch=10):

        self.batch=batch
        self.epoch=epoch
        self.GPU0 = '0'
        self.train_mod_dir='./train_model/'

        self.path=train_path




    def batchs(self,btch):

        dataset_path = './dataset/archive/train images 13440x32x32/train/'
        files = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]

        # print(files)

        labels = []
        images = np.zeros((len(files), 32, 32))

        c = 0
        for file in files:
            id = file.split('.')[0].split('_')[1]
            label = file.split('.')[0].split('_')[-1]
            labels.append(int(label))
            image = Image.open(dataset_path + file)
            images[c, :, :] = (np.asarray(image))
            c = c + 1

        a = np.asarray(labels)
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1
        #btch = 100
        total = b.shape[0]

        self.num_btchs = int(total / btch)
        start = 0
        end = btch

        self.new_label_gt = np.zeros((self.num_btchs, btch, b.shape[1]))
        self.new_img_gt = np.zeros((self.num_btchs, btch, 32, 32))
        for n in range(self.num_btchs):
            self.new_label_gt[n, :, :] = b[start:end, :]
            self.new_img_gt[n, :, :, :] = images[start:end, :, :]
            start = end
            end = end + btch




    def classifier(self, X):

        with tf.device('/gpu:' + self.GPU0):

            X = tf.reshape(X, [-1, 32,32, 1])
            X = tf.identity(X, name="classifier_input")




            print(X.get_shape())

            c_e = [1, 32,64,128,256,64,29]

            s_e = [0, 1, 1, 1,1,1,1]
            layers_e = []
            layers_e.append(X)
            for i in range(1, 7, 1):

                if (i >= 5):
                    layer = tools.Ops.conv2d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i], name='e' + str(i))
                    layer = tools.Ops.maxpool2d(tools.Ops.xxlu(layer, label='lrelu'), k=2, s=2, pad='SAME')
                else:
                    layer = tools.Ops.conv2d(layers_e[-1], k=4, out_c=c_e[i], str=s_e[i], name='e' + str(i))
                    layer = tools.Ops.maxpool2d(tools.Ops.xxlu(layer, label='lrelu'), k=2, s=2, pad='SAME')
                print(layer)
                # layer = tf.layers.batch_normalization(layer)

                layers_e.append(layer)
            # exit(0)
            ### fc
            print(layers_e[-1].get_shape())

            [_, d1, d2, cc] = layers_e[-1].get_shape()
            d1 = int(d1);
            d2 = int(d2);

            cc = int(cc)
            lfc = tf.reshape(layers_e[-1], [-1, int(d1) * int(d2) *  int(cc)])

            Y_sig = tf.nn.softmax(lfc)
            print(Y_sig)
            #exit(0)
            Y_sig = tf.identity(Y_sig, name="Softmax_output")

            return Y_sig
    def dis(self,  Y):
        with tf.device('/gpu:'+self.GPU0):



            l1= tools.Ops.fc(Y,out_d=16,name='l1')
            l1=tools.Ops.xxlu(l1, label='relu')
            l2= tools.Ops.fc(l1, out_d=16, name='l2')
            l2=tools.Ops.xxlu(l2, label='relu')

            # print(l2)
            # exit(0)
        return tf.nn.sigmoid(l2)



    def build(self):

        self.X= tf.placeholder(shape=[None,32,32,1],dtype=tf.float32)
        self.Y=tf.placeholder(shape=[None,29],dtype=tf.float32)


        with tf.variable_scope('classifier'):
                self.Y_class = self.classifier(self.X)

        with tf.variable_scope('dis'):
            self.XY_real_pair = self.dis( self.Y)
        with tf.variable_scope('dis',reuse=True):
            self.XY_fake_pair = self.dis(self.Y_class)



        with tf.device('/gpu:' + self.GPU0):

            #################################classifier loss
            self.classifier_loss = tf.reduce_mean(
                -tf.reduce_sum(self.Y * tf.log(self.Y_class), reduction_indices=[1]))

            #################################  ae + gan loss

            self.gan_g_loss = -tf.reduce_mean(self.XY_fake_pair)
            self.gan_d_loss_no_gp = tf.reduce_mean(self.XY_fake_pair) - tf.reduce_mean(self.XY_real_pair)
            alpha = tf.random_uniform(shape=[tf.shape(self.X)[0], 29], minval=0.0, maxval=1.0)

            Y_pred_ = tf.reshape(self.Y_class, shape=[-1, 29])
            differences_ = Y_pred_ - self.Y
            interpolates = self.Y + alpha * differences_
            with tf.variable_scope('dis', reuse=True):
                XY_fake_intep = self.dis( interpolates)
            gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
            self.gan_d_loss_gp = self.gan_d_loss_no_gp + 10 * gradient_penalty

            gan_g_w = 20
            aeu_w = 100 - gan_g_w
            self.aeu_gan_g_loss = 1* self.classifier_loss + 0.001 * self.gan_g_loss

            ##################################


        with tf.device('/gpu:' + self.GPU0):
            class_var = [var for var in tf.trainable_variables() if var.name.startswith('classifier')]
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]

            self.aeu_g_optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8). \
                minimize(self.classifier_loss, var_list=class_var)

            self.dis_optim = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8). \
                minimize(self.gan_d_loss_gp, var_list=dis_var)


        print(tools.Ops.variable_count())
        self.sum_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = self.GPU0

        self.sess = tf.Session(config=config)


        path = self.train_mod_dir
        # path = './Model_released/'   # to retrain our released model
        if os.path.isfile(path + 'model.cptk.data-00000-of-00001'):
            print('restoring saved model')
            self.saver.restore(self.sess, path + 'model.cptk')
        else:
            print('initilizing model')
            self.sess.run(tf.global_variables_initializer())

    def train(self,batch):

        for epoch in range(0, 7):

            wrong=0
            correct=0
            total=0.



            self.batchs(batch)

            total_train_batch_num = self.num_btchs
            print('total_train_batch_num:', total_train_batch_num)
            for i in range(total_train_batch_num):





                #################### training
                X_train_batch=self.new_img_gt[i,:,:,:].reshape(-1,32,32,1)




                gt_class = self.new_label_gt[i,:,:].reshape(-1,29)


                print(gt_class.shape)
                print(X_train_batch.shape)
                #exit(0)
                self.sess.run(self.dis_optim, feed_dict={self.X: X_train_batch, self.Y: gt_class})
                self.sess.run(self.aeu_g_optim,feed_dict={self.X: X_train_batch, self.Y: gt_class})

                print(gt_class.shape)

                c_loss,y_class=self.sess.run([self.classifier_loss,self.Y_class],feed_dict={self.X: X_train_batch, self.Y: gt_class})



                print('ep:', epoch, 'i:', i, 'classifier loss:', c_loss)

                gt_v = np.where(gt_class[0,:]==np.max(gt_class[0,:]))
                y_v= np.where(y_class[0,:]==np.max(y_class[0,:]))




                gt_v=int(np.asarray(gt_v).reshape(1))
                y_v=int(np.asarray(y_v).reshape(1))
                if gt_v==y_v:
                    correct=correct+1

                else:
                    wrong=wrong+1

                total = total + 1.0

                print('wrong total :'+str(wrong))
                print('correct total: '+str(correct))



                print((gt_v))
                print(int(y_v))
                #exit(0)
                if (i % 600 == 0):
                    result= (correct)/total
                    result=result*100
                    print('correct: '+str(result)+'%')
                    result=(wrong)/total
                    result=result*100
                    print('wrong: '+str(result)+'%')
                    time.sleep(5)
                if i%600 == 0 :
                    self.saver.save(self.sess, save_path=self.train_mod_dir + 'model.cptk')
                    print ('ep:', epoch, 'i:', i, 'model saved!')








if __name__== '__main__':
    model = classifier()
    model.build()
    model.train(5)
