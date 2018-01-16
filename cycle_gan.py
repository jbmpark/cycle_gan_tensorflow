import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import random
import scipy
np.set_printoptions(threshold=np.nan)


GPU = '0'   
discriminator_type = 'patch_gan'    #'patch_gan' , 'dense'
normalization_style='batch_norm' # 'instance_norm' or 'batch_norm'


if normalization_style=='instance_norm':
    batch_size = 1
else:
    batch_size = 6
#################################
# parameters
width  = 256
height = 256
channel = 3
epoch = 1000
latent_len = 100
train_data_dir_A = '../horse2zebra/trainA'  #horse
train_data_dir_B = '../horse2zebra/trainB'  #zebra
test_data_dir_A = '../horse2zebra/testA'
test_data_dir_B = '../horse2zebra/testB'
sample_dir = './samples_{}'.format(GPU)
checkpoint_dir = './checkpoint_{}'.format(GPU)
conv_kernel_size = 3




kernel_init = tf.truncated_normal_initializer(stddev=0.02)
bn_momentum=0.9
bn_eps = 0.00001
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

from shutil import copyfile
copyfile(__file__, sample_dir+'/'+__file__)

################################# 
# input
#G_in = tf.placeholder(tf.float32, (None, height, width, channel), name='g_input')
X_A = tf.placeholder(tf.float32, (None, height, width, channel), name='input_A')
X_B = tf.placeholder(tf.float32, (None, height, width, channel), name='input_B')
PHASE = tf.placeholder(tf.bool, name='is_training')
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = tf.placeholder(tf.float32, shape=[])


# instance_norm is borrowed from 'https://github.com/xhujoy/CycleGAN-tensorflow'
def instance_norm(x, depth):
    #depth = x.get_shape()[3]
    #scale = tf.get_variable('instance_scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02))
    #offset = tf.get_variable('instance_offset', [depth], initializer=tf.constant_initializer(0.0))
    scale = tf.Variable(tf.random_normal([depth], mean=1.0, stddev=0.02))
    offset = tf.Variable(tf.zeros([depth]))
    mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    inv = tf.rsqrt(variance + 1e-5)
    normalized = (x - mean) * inv
    return scale * normalized + offset
#################################
# encode block (convolution)
def enc(in_layer, num_out_tensor, kernel_size=conv_kernel_size, strides=(2,2), padding='same' , use_bn=True, phase=True, activation=None):
    _layer = tf.layers.conv2d(in_layer, num_out_tensor, kernel_size, strides=strides, padding=padding, activation=None, kernel_initializer=kernel_init)
    if use_bn:
        if normalization_style=='instance_norm':
            _layer = instance_norm(_layer, num_out_tensor)
        else:
            _layer = tf.layers.batch_normalization(_layer, training=phase, momentum=bn_momentum, epsilon=bn_eps) 
            

    if activation=='relu':
        _layer = tf.nn.relu(_layer)
    elif activation=='lrelu':
        _layer = tf.nn.leaky_relu(_layer)
        
    return _layer

#################################
# decode block (deconv)
def dec(in_layer, num_out_tensor, kernel_size=conv_kernel_size, strides=(2,2), padding='same' , use_bn=True, phase=True, activation=None):
    _layer = tf.layers.conv2d_transpose(in_layer, num_out_tensor, kernel_size, strides=strides, padding=padding, activation=None, kernel_initializer=kernel_init)
    if use_bn:
        if normalization_style=='instance_norm':
            _layer = instance_norm(_layer, num_out_tensor)
        else:
            _layer = tf.layers.batch_normalization(_layer, training=phase, momentum=bn_momentum, epsilon=bn_eps) 
    

    if activation=='relu':
        _layer = tf.nn.relu(_layer)
    elif activation=='lrelu':
        _layer = tf.nn.leaky_relu(_layer)
        
   
    return _layer

def transform_resnet(in_layer, num_out_tensor, phase=True):
    t1 = enc(in_layer, num_out_tensor, kernel_size=3, strides=(1,1), phase=phase, activation=None)
    t2 = enc(in_layer, num_out_tensor, kernel_size=3, strides=(1,1), phase=phase, activation=None)
    return (in_layer+t2)

#################################
# generator
#with tf.name_scope('Generator'):
def Generator(input_image, is_training, variable_scope_name, reuse=False):
    #input image : 256x256x3
    with tf.variable_scope(variable_scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        # Encoder
        e1 = enc(input_image,   64,     kernel_size=7, strides=(1,1), phase=is_training, activation='relu')    #256x256x64
        e2 = enc(e1,            128,    kernel_size=3, strides=(2,2), phase=is_training, activation='relu')                  #128x128x128
        e3 = enc(e2,            256,    kernel_size=3, strides=(2,2), phase=is_training, activation='relu')                  #64x64x256

        # Transformer
        t1 = transform_resnet(e3,   256,    phase=is_training)
        t2 = transform_resnet(t1,   256,    phase=is_training)
        t3 = transform_resnet(t2,   256,    phase=is_training)
        t4 = transform_resnet(t3,   256,    phase=is_training)
        t5 = transform_resnet(t4,   256,    phase=is_training)
        t6 = transform_resnet(t5,   256,    phase=is_training)
        t7 = transform_resnet(t6,   256,    phase=is_training)
        t8 = transform_resnet(t7,   256,    phase=is_training)
        t9 = transform_resnet(t8,   256,    phase=is_training)

        # Decoder
        d1 = dec(t9, 128, kernel_size=3, strides=(2,2), phase=is_training, activation='relu')      #128x128x128
        d2 = dec(d1, 64,  kernel_size=3, strides=(2,2), phase=is_training, activation='relu')      #256x256x64
        d3 = enc(d2, 3,   kernel_size=7, strides=(1,1), phase=is_training, activation=None)    #256x256x3
        d4 = tf.nn.tanh(d3)
    return d4



#################################
# discriminator
#with tf.name_scope('Discriminator'):
def Discriminator(input_image, is_training, variable_scope_name, reuse=False):
    with tf.variable_scope(variable_scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        if discriminator_type=='patch_gan':
            l1 = enc(input_image, 64, phase=is_training, use_bn=False, activation='lrelu')#128x128x64
            l2 = enc(l1, 128, phase=is_training, activation='lrelu')         #64x64x128
            l3 = enc(l2, 256, phase=is_training, activation='lrelu')         #32x32x256
            l4 = enc(l3, 512, phase=is_training, activation='lrelu', strides=(1,1))         #32x32x512
            l5 = enc(l4, 1,  phase=is_training, use_bn=False, activation=None, strides=(1,1))          #32x32x1
            return l5
        elif discriminator_type=='dense':
            l1 = enc(input_image, 64, phase=is_training, use_bn=False, activation='lrelu')#128x128x64
            l2 = enc(l1, 128, phase=is_training, activation='lrelu')         #64x64x128
            l3 = enc(l2, 256, phase=is_training, activation='lrelu')         #32x32x256
            l4 = enc(l3, 512, phase=is_training, activation='lrelu')         #16x16x512
            l5 = enc(l4, 1024, phase=is_training, activation='lrelu')         #8x8x1024
            l6 = enc(l5, 2048, phase=is_training, activation='lrelu')         #4x5x2048
            l7 = tf.layers.flatten(l6)
            l8 = tf.layers.dense(l7,  1,    activation=None, use_bias=True, kernel_initializer=kernel_init)
            return l8
        else:
            print 'Please select proper discriminator_type'


#with tf.name_scope('Optimizer'):

variable_scope_name_G = 'G_Vraiables'
variable_scope_name_F = 'F_Vraiables'
variable_scope_name_D_A = 'D_A_Vraiables'
variable_scope_name_D_B = 'D_B_Vraiables'
with tf.device("/device:GPU:{}".format(GPU)):
    #################################
    # gan network
    G = Generator(X_A, PHASE, variable_scope_name_G)
    F = Generator(X_B, PHASE, variable_scope_name_F)
    D_A_real = Discriminator(X_B,   PHASE, variable_scope_name_D_A)
    D_A_fake = Discriminator(G,     PHASE, variable_scope_name_D_A, reuse=True)
    D_B_real = Discriminator(X_A,   PHASE, variable_scope_name_D_B)
    D_B_fake = Discriminator(F,     PHASE, variable_scope_name_D_B, reuse=True)
    
    A_cycle = Generator(G, PHASE, variable_scope_name_F, reuse=True)    
    B_cycle = Generator(F, PHASE, variable_scope_name_G, reuse=True)

    #################################

    # train step
    D_A_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_A_real, labels=tf.ones_like(D_A_real)))
    D_A_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_A_fake, labels=tf.zeros_like(D_A_fake)))
    D_B_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_B_real, labels=tf.ones_like(D_B_real)))
    D_B_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_B_fake, labels=tf.zeros_like(D_B_fake)))
    D_A_loss = (D_A_loss_real + D_A_loss_fake)/2
    D_B_loss = (D_B_loss_real + D_B_loss_fake)/2


    G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_A_fake, labels=tf.ones_like(D_A_fake)))
    F_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_B_fake, labels=tf.ones_like(D_B_fake)))
    G_loss_cycle = tf.reduce_mean(tf.abs(X_A-A_cycle))
    F_loss_cycle = tf.reduce_mean(tf.abs(X_B-B_cycle))
    loss_cycle =  G_loss_cycle+F_loss_cycle
     
    
    G_gan_loss_ratio = (1.0/11.0)
    G_loss = G_gan_loss_ratio*G_loss_gan + (1.0-G_gan_loss_ratio)*loss_cycle
    F_loss = G_gan_loss_ratio*F_loss_gan + (1.0-G_gan_loss_ratio)*loss_cycle

    D_A_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope_name_D_A)
    D_B_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope_name_D_B)
    G_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope_name_G)
    F_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=variable_scope_name_F)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    base_lr=0.0002
    with tf.control_dependencies(update_ops):
        D_A_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(D_A_loss, var_list=D_A_var_list, global_step = global_step)
        D_B_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(D_B_loss, var_list=D_B_var_list, global_step = global_step)
        G_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(G_loss, var_list=G_var_list, global_step = global_step)
        F_train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(F_loss, var_list=F_var_list, global_step = global_step)


#tensorboard
tf.summary.scalar('D_A_Loss', D_A_loss)
tf.summary.scalar('D_B_Loss', D_B_loss)
tf.summary.scalar('G_loss', G_loss)
tf.summary.scalar('F_loss', F_loss)

#################################
#misc

def get_batch_file_list(list, n):
    start = n*batch_size
    return list[start:(start+batch_size)]

def img_squeeze(img):   # 0~255 -->  -1 ~ 1
    return ((img*2.0)/256.0) -1.

def img_recover(img):
    img =((img+1.)*256.0)/2.0
    return img.astype(int)

def read_image(file, scale_w, scale_h):
    img = scipy.misc.imread(file, mode='RGB').astype(np.float)
    img = scipy.misc.imresize(img, [scale_w, scale_h])
    img = img_squeeze(img)
    return img
  

def read_batch(batch_file_list):
    return [read_image(file , width, height) for file in batch_file_list]

#################################
# training

# data 
train_data_file_list_A = glob.glob(os.path.join(train_data_dir_A, '*.jpg'))
train_data_file_list_B = glob.glob(os.path.join(train_data_dir_B, '*.jpg'))
train_num_data_A = len(train_data_file_list_A)
train_num_data_B = len(train_data_file_list_B)
train_num_data=train_num_data_A
if(train_num_data_A>train_num_data_B):
    train_num_data=train_num_data_B
train_num_batch = int(train_num_data/batch_size)


print('train_num_data_A  :  %d'%train_num_data_A)
print('train_num_data_B  :  %d'%train_num_data_B)
print('train_num_data    :  %d'%train_num_data)
print('batch_size        :  %d'%batch_size)
print('train_num_batch   :  %d'%train_num_batch)


test_size = 10  #num images to test
test_data_file_list_A = glob.glob(os.path.join(test_data_dir_A, '*.jpg'))
test_data_file_list_A = test_data_file_list_A[:test_size]
test_data_file_list_B = glob.glob(os.path.join(test_data_dir_B, '*.jpg'))
test_data_file_list_B = test_data_file_list_B[:test_size]
test_A = read_batch(test_data_file_list_A)
test_B = read_batch(test_data_file_list_B)
test_A_recon  = [(test_A[i]+1.0)/2.0 for i in range(test_size)]
test_B_recon  = [(test_B[i]+1.0)/2.0 for i in range(test_size)]
            



#session
gpu_options = tf.GPUOptions(allow_growth=True)  # Without this, the process occupies whole area of memory in the all GPUs.
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

#if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#    saver.restore(sess, ckpt.model_checkpoint_path)
#else:
#    sess.run(tf.global_variables_initializer())
sess.run(tf.global_variables_initializer())

#tensorboard
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs_{}'.format(GPU), sess.graph)

#epoch =1
#train_num_batch=100

for e in range(epoch):  
    random.shuffle(train_data_file_list_A)  #to avoid overfitting
    random.shuffle(train_data_file_list_B)  #to avoid overfitting
    if(e < 100) :
        lr = base_lr
    else:
        #lr = base_lr - base_lr*(e-100)/100
        lr = base_lr/10


    for b in range(train_num_batch):
        this_batch_A = get_batch_file_list(train_data_file_list_A, b)
        this_batch_B = get_batch_file_list(train_data_file_list_B, b)
        input_A = read_batch(this_batch_A)
        input_B = read_batch(this_batch_B)

        #train 
        is_training=True
        G_, G_batch_loss, F_, F_batch_loss = \
            sess.run([G_train, G_loss, F_train, F_loss], \
            feed_dict={X_A:input_A, X_B:input_B, PHASE:is_training, learning_rate:lr})

        D_A_, D_A_batch_loss, D_B_, D_B_batch_loss = \
            sess.run([D_A_train, D_A_loss, D_B_train, D_B_loss], \
            feed_dict={X_A:input_A, X_B:input_B, PHASE:is_training, learning_rate:lr})

         
        print("epoch: %04d"%e, "batch: %05d"%b, \
            "lr: {:.04}".format(lr), \
            "D_A_loss: {:.04}".format(D_A_batch_loss), \
            "D_B_loss: {:.04}".format(D_B_batch_loss), \
            "G_loss: {:.04}".format(G_batch_loss), \
            "F_loss: {:.04}".format(F_batch_loss) )

               
        # testing
        if not e%10:
            if b==(train_num_batch-1): 
                is_training=False
                samples_G = sess.run(G, feed_dict={X_A:test_A, X_B:test_B, PHASE:is_training})
                samples_F = sess.run(F, feed_dict={X_A:test_A, X_B:test_B, PHASE:is_training})
                samples_G = (samples_G+1.0)/2.0
                samples_F = (samples_F+1.0)/2.0
                fig, ax = plt.subplots(4, test_size, figsize=(test_size, 2), dpi=400)
                for k in range(test_size):
                    ax[0][k].set_axis_off()
                    ax[0][k].imshow(test_A_recon[k])
                    ax[1][k].set_axis_off()
                    ax[1][k].imshow(samples_G[k])
                    ax[2][k].set_axis_off()
                    ax[2][k].imshow(test_B_recon[k])
                    ax[3][k].set_axis_off()
                    ax[3][k].imshow(samples_F[k])
                plt.savefig(sample_dir+'/cycle_gan_{}'.format(str(e).zfill(3)) + '_{}.png'.format(str(b).zfill(5)), bbox_inches='tight')
                plt.close(fig)

                saver.save(sess, checkpoint_dir+'/cycle_gan.ckpt', global_step=global_step)
                #tensorboard
                is_training=False
                summary = sess.run(merged, feed_dict={X_A:test_A, X_B:test_B, PHASE:is_training})
                writer.add_summary(summary, global_step=sess.run(global_step))




