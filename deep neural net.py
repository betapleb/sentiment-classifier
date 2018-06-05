import tensorflow as tf
from create_sentiment_featuresets import create_feature_sets_and_labels
import pickle
import numpy as np
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

'''
input > weight > hidden layer 1 (activation function) > weights
> hidden layer 2 (activcation function) > weights > output layer

compare output to intended output > cost or loss function (ie. cross entropy)
optimazation function (ie. optimizer) > minimize that cost (ie. AdamOptimizer, stochastic gradient descent, AdaGrad)

backpropagation: go backwards to manipulate weights.

feed forward + backprop = epoch 
'''



n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10

# x isn't a specific value, it's a placeholder.
x = tf.placeholder('float')
y = tf.placeholder('float')

# biases allow some neurons to fire even if all inputs are 0.
# create a tensor or array of your data using random numbers. 
    
hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}
    

# nothing changes from mnist set
def neural_network_model(data):

    # activation function
    # (input_data * weights) + biases
    
    # matrix multiplication
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
             # tf.nn.relu: an activation function that computes rectified linear and returns a tensor.
             # use ReLU rather than sigmoid and tanh. Sigmoid range [0,1], ReL range [0,∞]. 

                          # layer2's input is whatever the activation function returns for layer1
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output




def train_neural_network(x):
    prediction = neural_network_model(x)   # pass in data
    cost = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )   # cost var measures how wrong. should minimize by manipulating weights.    

    # optimize cost function using AdamOptimizer. Other popular
    # optimizers are Stochastic gradient descent & AdaGrad.
    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        '''
        for each epoch and batch in data, run optimizer & cost against batch of data.
        to keep track of loss/cost at each step, add total cost per epoch up. 
        '''
        for epoch in range(hm_epochs):
            epoch_loss = 0

            i=0
            while i < len(train_x):     # using i to iterate through the data
                start = i
                end = 1 + batch_size    # range is wherever we are with i, to the i+batch_size value.
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
                

        # how many predictions made that matched the labels.
        # compare prediction to actual label.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x, y:test_y})) 



train_neural_network(x)

'''
This gives ~62% accuracy.
But testing with more layers, more nodes...etc leaves
minimal impact.
dataset sizes is important
'''



