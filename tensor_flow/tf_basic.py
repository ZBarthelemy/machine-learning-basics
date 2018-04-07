import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#result = x1 * x2
result = tf.multiply(x1,x2)
print(result)

#no computation is done until instantiation
with tf.Session() as sesh:
	output = sesh.run(result)
	print(output)
print (output)