#!/usr/bin/env python3

#importing libraries
import tensorflow as tf
import numpy as np

#resetting graph
tf.reset_default_graph()

#defining matrices
a=tf.Variable(np.ones((2,2)),dtype=np.float32)
b=tf.constant(np.ones((2,2)),dtype=np.float32)
c=tf.placeholder(np.float32,(2,2))

#multiplying matrices
d=a@b
e=c@b

#Creating session
s=tf.InteractiveSession()
s.run(tf.global_variables_initializer())
print(s.run(d))
print(s.run(e,feed_dict={c:np.ones((2,2))}))


