#!/usr/bin/env python3

#importing libraries
import tensorflow as tf
import numpy as np

#data
N=1000
D=3
#inputs
x=np.random.random((N,D))
#weights
w=np.random.random((D,1))

y= x @ w + np.random.random((N,1))*0.20

tf.reset_default_graph()

features=tf.placeholder(tf.float32,(None,D))
target=tf.placeholder(tf.float32,(None,1))

weights=tf.get_variable("weights",shape=(D,1),dtype=tf.float32)

y_pred=features @ weights 

loss=tf.reduce_mean((y_pred-target)**2)

optimizer=tf.train.GradientDescentOptimizer(0.1)
step=optimizer.minimize(loss)

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	for i in range(300):
		_,curr_loss,curr_weights=s.run([step,loss,weights],feed_dict={features:x,target:y})
		if i%50 ==0:
			print(curr_loss)
