#!/usr/bin/env python3

#importing libraries
import tensorflow as tf
import numpy as np

'''
#SIMPLE OPTIMIZATION
tf.reset_default_graph()
x=tf.get_variable("x",shape=(),dtype=tf.float32)
f=x**2


#defining optimizer with learning rate = 0.1
optimizer=tf.train.GradientDescentOptimizer(0.1)
step=optimizer.minimize(f,var_list=[x])


#creating session
s=tf.InteractiveSession()
s.run(tf.global_variables_initializer())

for i in range(10):
	_, curr_x, curr_f= s.run([step,x,f])
	print(curr_x,curr_f)

s.close()
'''

'''
#LOGGING WITH TENSORFLOW PRINT
tf.reset_default_graph()
x=tf.get_variable("x",shape=(),dtype=tf.float32)
f=x**2

f=tf.Print(f,[x,f],"x, f:")

optimizer=tf.train.GradientDescentOptimizer(0.1)
step=optimizer.minimize(f)

with tf.Session() as s:	#another method to declare session
	s.run(tf.global_variables_initializer())
	
	for i in range(10):
		s.run([step,f])
'''

#LOGGING USING TENSORBOARD
tf.reset_default_graph()
x=tf.get_variable("x",shape=(),dtype=tf.float32)
f=x**2

optimizer=tf.train.GradientDescentOptimizer(0.1)
step=optimizer.minimize(f)

#creating summary
tf.summary.scalar('curr_x',x)
tf.summary.scalar('curr_f',f)
summaries=tf.summary.merge_all()

s=tf.InteractiveSession()
summary_writer=tf.summary.FileWriter("logs/2",s.graph)
s.run(tf.global_variables_initializer())

for i in range(10):
	_, curr_summary=s.run([step,summaries])
	summary_writer.add_summary(curr_summary,i)
	summary_writer.flush()

