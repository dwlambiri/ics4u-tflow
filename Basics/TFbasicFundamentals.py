#This is our first file including all the simple pieces of code we tested to further our understanding
#about some of the easiest parts of tensorflow

#this is the file header that basically includes and initializes tensorflow 
import tensorflow as tf

#these are tensors, a central unit of tensorflow
#contains a data type, location on a computational graph, and a value
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

#prints the information contained in the tensor to screen
print(node1, node2)


#creates a session, which is how you run the graph, results in the printing of numeric values
sess = tf.Session()
print(sess.run([node1, node2]))



#adds the two tensors together, producing 7 as the value held at node 3
node3 = tf.add(node1, node2)

#prints the tensor info
print("node3:", node3)
#prints the value held at the tensor:: sess.run(node3)
print("sess.run(node3):", sess.run(node3))


#these are placeholders, something that can hold a different values at different times
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#this is a sort of mini function doing what tf.add() did before
adder_node = a + b  

#adder node adds other nodes like a function but doesn't HOLD a value
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


#this is a compound operation involving both addition and multiplication
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


#makes a line of data, using y=wx+b
W = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([-2], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b


#unknown
init = tf.global_variables_initializer()

#runs model and prints results of the input values being turned into the output values
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))


#the loss function running, calculates the difference between the ideal output versus the current output
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#these are the perfect parameters to produce the desired output, this is what our machine is trying to do
#the model is an exact match for the data
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))