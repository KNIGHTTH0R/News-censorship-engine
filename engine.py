from flask import Flask, redirect, url_for, request
import numpy as np
import re
import tensorflow as tf

app = Flask(__name__)

#################################################################################
the_classes = np.array(['sensitive', 'nonsensitive'])
num_classes = len(the_classes)

with open('np/positive.txt', 'r') as myfile:
    content_pos = myfile.readlines()
    content_pos = [x.strip() for x in content_pos] 

with open('np/negative.txt', 'r') as myfile:
    content_neg = myfile.readlines()
    content_neg = [x.strip() for x in content_neg] 

mergedlist = content_pos + content_neg
num_features = len(mergedlist)
          
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

feature_size_flat = num_features
x = tf.placeholder(tf.float32, shape=[None, feature_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
layer_fc1 = new_fc_layer(input= x,
                         num_inputs=num_features,
                         num_outputs=1000,
                         use_relu=True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=1000,
                         num_outputs=num_classes,
                         use_relu=False)
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

saver = tf.train.Saver()
session = tf.Session()
save_path = "the_model/opinion_model"
saver.restore(sess=session, save_path= save_path)
#######################################################################################

@app.route('/success/<name>')
def success(name):
	return 'kategori berita: %s' % name

@app.route('/classifier', methods=['POST', 'GET'])
def login():
	if request.method == 'POST':
		myfile = request.form['nm']
		wordList = re.sub("[^\w]", " ",  myfile).split()

		# mencocokkan kata
		feature = [0] * (len(mergedlist))
		for i in range (len(wordList)):
    			kata = wordList[i]
        
    			# mencari kata yang cocok dari tabel
    			for j in range (len(mergedlist)):
        			if mergedlist[j] == kata :
            				feature[j] += 1

		feature = np.asarray(feature)
		feature = feature.reshape(1, num_features)
		feed_dict = {x: feature}
		tmp = y_pred_cls.eval(feed_dict, session = session)
		hasil = the_classes[tmp]

		return redirect(url_for('success', name=hasil))
	else:
		user = request.args.get('nm')
		return redirect(url_for('success', name=user))

if __name__ == '__main__':
    app.run(debug=True)
