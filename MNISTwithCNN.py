
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


input_data = input_data.read_data_sets('MNIST_data/',one_hot= True)



# Helper Functions

    #init Weights
def init_weights(shape):
    init_random_distribution = tf.truncated_normal(shape=shape,stddev=0.1) 
    return tf.Variable(init_random_distribution)

    #init Bias
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

    #return a 2d convolution
def conv2d(x,W):
    # x --> Input tensor [batch ,Height,width,channels]
    # W --> Kernel [Filter Height,Filter WIdth , channels, channels_out]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

    #pooling helper function
def max_pooling_2by2(x):
    # x --> Input tensor [batch ,Height,width,channels]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # Convolutional Layer
def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W) + b )

    # Fully Connected Layer
def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W) + b


# In[4]:


# #Steps : 
# 0. Data Preprocessing
# 1. Placeholders
# 2. Layers
# 2. Variables
# 3. Graph operations
# 4. Loss function
# 5. Optimizer
# 6. Session



# Data Preprocessing





# Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])


# In[7]:


# Layers
x_image = tf.reshape(x,[-1,28,28,1]) # recreate the original image , 28 by 28 , one channel
convo_1  = convolutional_layer(x_image,shape=[5,5,1,32]) # 32 feautures for 5by5 batch , 1 is the nu of channels
convo_1_pooling = max_pooling_2by2(convo_1)
# we stack the layers , something worth mentioning , is that the 32 feautures as output of convo_1 are translated 
# to a 32 channel input on the second convo layer . Lastly , we produce a 64 feauture output
convo_2  = convolutional_layer(convo_1_pooling,shape = [5,5,32,64]) 
convo_2_pooling = max_pooling_2by2(convo_2)
# image size is 7 by 7 , 64 is the output of convo_2
convo_2_flat = tf.reshape(convo_2_pooling,shape=[-1,7*7*64])
# 1024 is the number of neurons we will create
normal_full_connected = normal_full_layer(convo_2_flat,1024)

full_layer_1  = tf.nn.relu(normal_full_connected)
  
    # Drop Out Layer
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_1,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,10)


# In[8]:


# Loss functions

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))


# In[9]:


# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate=0.001 )
train     = optimizer.minimize(cross_entropy)


# In[ ]:


# Session

init = tf.global_variables_initializer()
steps = 500

with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x , batch_y = input_data.train.next_batch(batch_size=5)
        sess.run(train,feed_dict = {x : batch_x,y_true : batch_y , hold_prob : 0.5})
        
        if i % 100 == 0 or i == steps - 1 :
            print("On step: {} " .format(i))
            print("Accuracy: ")
            matches = tf.equal(tf.argmax(y_true,1),tf.argmax(y_pred,1))
            print("1")
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            #dict_to_feed = {x:input_data.test.images[:50],y_true:input_data.test.labels[:50],hold_prob : 1.0}
            # code below will take hours to run , code above presents a sample    
            print(2)
            dict_to_feed = {x:input_data.test.images,y_true:input_data.test.labels,hold_prob : 1.0}

            print(sess.run(acc,feed_dict=dict_to_feed))
            print('\n')

