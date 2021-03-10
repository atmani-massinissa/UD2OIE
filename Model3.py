import tensorflow as tf
import numpy as np
def PositionAttentionLayer(lab,key,values,num_unit,seq_len,mask_seq_len,batch_size,dropout,scale_regularizer,name="PA"):
	with tf.name_scope('%s' % (name)):
		w_key = tf.keras.layers.Dense(num_unit,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale_regularizer),use_bias=False)
		w_values = tf.keras.layers.Dense(num_unit,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale_regularizer),use_bias=False)
		w_positions = tf.keras.layers.Dense(num_unit,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale_regularizer),use_bias=False)
		w_V = tf.keras.layers.Dense(1,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale_regularizer),use_bias=False)
		dum = tf.zeros_like(values)
		lab_b = tf.expand_dims(lab,axis=2)
		lab_b = tf.reshape(tf.tile(lab_b,tf.constant([1,1,num_unit], tf.int32)),[-1,seq_len,num_unit])
		cond = tf.math.greater(lab_b, tf.constant([2]))
		positions_v = tf.where(cond,values,dum)
		p = tf.reduce_sum(positions_v,1)
		A= tf.expand_dims(w_values(values),1)
		B = tf.expand_dims(tf.expand_dims(w_positions(p),1),1)
		C=A+B
		D=C+ tf.expand_dims(w_key(values),2)
		E =tf.nn.tanh(D)
		F = w_V(E)
		attention_weights = tf.nn.softmax(F, axis=1)
		attention_weights = tf.nn.dropout(attention_weights,dropout)
		#score = tf.squeeze(attention_weights,-1)
		mask = tf.cast(tf.sequence_mask(mask_seq_len,maxlen=seq_len),dtype=tf.float32)
		attention_weights = tf.expand_dims(tf.einsum('lij,lj->lij',  tf.squeeze(attention_weights,-1),mask),axis=3)
		context_vector = tf.einsum('lki,lkjh->lki',  values,attention_weights)
		#attention_weights = tf.expand_dims(score,axis=-1)
		#context_vector = attention_weights * tf.expand_dims(values,1)
		#context_vector = tf.reduce_sum(context_vector, axis=1)
	return context_vector, attention_weights

def argument(features, labels, mode, params):
	"""Model function for BBC."""
	import random
	learning_rate = params["learning_rate"] #0.001
	input_dim = params["input_dim"] #40
	batch_size = params["batch_size"] #5
	num_units = params["num_units"] #300 # the number of units in the LSTM cell
	gcn_units = params["gcn_units"] #300
	input_dim_word = params["input_dim_word"]
	word_vocab_size = params["word_vocab_size"]
	dep_vocab_size = params["dep_vocab_size"]
	regularizer = None
	scale_regularizer = 0.0
	input_keep_prob= 1.0
	output_keep_prob= 1.0
	state_keep_prob= 1.0
	embedding_keep_prob=1.0
	dropout_rate= 1.0
	pos_embedding_keep_prob=1.0
	dep_embedding_keep_prob=1.0
	arg_embedding_keep_prob=1.0
	if mode == tf.estimator.ModeKeys.TRAIN:
		scale_regularizer = params["scale_regularizer"] #0.001
		input_keep_prob=params["input_keep_prob"] #0.7
		output_keep_prob=params["output_keep_prob"] #0.6
		state_keep_prob= params["state_keep_prob"] #0.6
		embedding_keep_prob= 0.9#params["state_keep_prob"] #0.9
		dropout_rate= params["dropout_rate"] #0.5
		pos_embedding_keep_prob = 0.9
		dep_embedding_keep_prob = 0.9
		#arg_embedding_keep_prob = 0.9

	num_classes = params["num_classes"] #2 #pp.num_classes
	max_seq_len = params["max_seq_len"] #101 #np.max(pp.num_nodes_by_sentences)
	VOCAB_SIZE =  params["VOCAB_SIZE"] #19 #pp.POS_n_words
	labels_num = params["labels_num"] #40 #len(pp.DEP_indices)
		
	input_data  = features["arg_input"]
	input_data_mask  = features["arg_mask"]
	input_prop_mask  = features["prop_mask"]
	original_sequence_lengths  = features["arg_original_sequence_lengths"]
	input_dep = features["arg_dep"]
	with tf.variable_scope("Arguments"):
		with tf.variable_scope("pos"):
			embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE, input_dim], -1, 1),trainable=True)
			with tf.name_scope("pos_dropout"):
				embedding = tf.nn.dropout(embedding, keep_prob=pos_embedding_keep_prob, noise_shape=[VOCAB_SIZE,1])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
		with tf.variable_scope("dep"):
			dep_embedding = tf.Variable(tf.random_uniform([dep_vocab_size, input_dim], -1, 1),trainable=True)
			with tf.name_scope("dep_dropout"):
				dep_embedding = tf.nn.dropout(dep_embedding, keep_prob=dep_embedding_keep_prob, noise_shape=[dep_vocab_size,1])
			inputs_dep = tf.nn.embedding_lookup(dep_embedding, input_dep)
		with tf.variable_scope("arg_mask"):
			predicate_embedding = tf.Variable(tf.random_uniform([4, input_dim], -1, 1),trainable=True)
			#with tf.name_scope("arg_dropout"):
				#dep_embedding = tf.nn.dropout(predicate_embedding, keep_prob=arg_embedding_keep_prob, noise_shape=[4,1])
			predicate_mask = tf.nn.embedding_lookup(predicate_embedding, input_data_mask)
		with tf.variable_scope("prop_mask"):
			prop_mask_embedding = tf.Variable(tf.random_uniform([4, input_dim], -1, 1),trainable=True)
			prop_mask = tf.nn.embedding_lookup(prop_mask_embedding, input_prop_mask)
		inputs = tf.concat([inputs,inputs_dep], axis=-1)
		inputs_m = tf.concat([predicate_mask,prop_mask], axis=-1)
		inputs = tf.concat([inputs,inputs_m], axis=-1)
		reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['vocab_path'],delimiter="\t")
		with tf.name_scope("BiLSTM"):
				with tf.variable_scope('forward'):
						lstm_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=1.0)
						lstm_fw_cell_dropout  = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_fw_cell,input_size=inputs.shape[-1],
																			   input_keep_prob=input_keep_prob,
																			   output_keep_prob=output_keep_prob,
																			   state_keep_prob=state_keep_prob,
																			 variational_recurrent=True,dtype=tf.float32)
				with tf.variable_scope('backward'):
						lstm_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=1.0)
						lstm_bw_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_bw_cell,input_size=inputs.shape[-1],
																			input_keep_prob=input_keep_prob,
																			   output_keep_prob=output_keep_prob,
																			   state_keep_prob=state_keep_prob,
																			   variational_recurrent=True,dtype=tf.float32)
				(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell_dropout, 
																					 cell_bw=lstm_bw_cell_dropout,
																					 inputs=inputs,
																					 sequence_length=original_sequence_lengths, 
																					 dtype=tf.float32,
																					 scope="BiLSTM")
		outputs = tf.reduce_mean(tf.concat([tf.expand_dims(output_fw, axis=3), tf.expand_dims(output_bw, axis=3)], axis=-1),axis=-1)

		with tf.variable_scope("proj_1"):
				W_1 = tf.get_variable("W_1", dtype=tf.float32,shape=[1*num_units, num_classes], initializer=tf.initializers.orthogonal(gain=0.6),regularizer = tf.contrib.layers.l2_regularizer(scale=scale_regularizer))
				b_1 = tf.get_variable("b_1", shape=[num_classes],dtype=tf.float32, initializer=tf.zeros_initializer())
				proj_outputs = tf.reshape(outputs, [-1, 1*num_units])
				pred_label = tf.matmul(proj_outputs, W_1) + b_1
				pred_label = tf.nn.dropout(pred_label,dropout_rate)
				logits_label = tf.reshape(pred_label, [-1, max_seq_len, num_classes])
		with tf.variable_scope("CRF"):
				crf_params = tf.get_variable("transition", [num_classes, num_classes], dtype=tf.float32)
				preds_label, scores_label = tf.contrib.crf.crf_decode(logits_label, crf_params, original_sequence_lengths)

		predictions = {
				# Generate predictions (for PREDICT and EVAL mode)
				"classes": preds_label,
				"tags":  reverse_vocab_tags.lookup(tf.to_int64(preds_label)) ,
				"probabilities":   tf.identity(scores_label, name="viterbi"),
				"sequence_lenghts": original_sequence_lengths
			}
		# PREDICT mode
		if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(
					mode=mode,
					predictions=predictions,
					export_outputs={
						'predict': tf.estimator.export.PredictOutput(predictions)
					})
					
		labels_tag = labels

		weights = tf.to_float(tf.sequence_mask(original_sequence_lengths,maxlen=max_seq_len))
		with tf.name_scope("CRF"):
			log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits_label, labels_tag, original_sequence_lengths, crf_params)
			loss = tf.reduce_mean(-log_likelihood/tf.cast(original_sequence_lengths,tf.float32))

		if(scale_regularizer!=0.0):
			loss +=  tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=scale_regularizer), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
				optimizer = tf.train.AdamOptimizer(learning_rate)
				train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
				return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	indices = params["indices"]
	 
	return tf.estimator.EstimatorSpec(mode=mode,loss=loss)

def predicate(features, labels, mode, params):
	"""Model function for BBC."""
	print("strat")
	learning_rate = params["learning_rate"] #0.001
	input_dim = params["input_dim"] #40
	batch_size = params["batch_size"] #5
	num_units = params["num_units"] #300 # the number of units in the LSTM cell
	gcn_units = params["gcn_units"] #300
	input_dim_word = params["input_dim_word"]
	word_vocab_size = params["word_vocab_size"]
	dep_vocab_size = params["dep_vocab_size"]
	
	regularizer = None
	scale_regularizer = 0.0
	input_keep_prob= 1.0
	output_keep_prob= 1.0
	state_keep_prob= 1.0
	embedding_keep_prob=1.0
	dropout_rate= 1.0
	if mode == tf.estimator.ModeKeys.TRAIN:
		scale_regularizer = params["scale_regularizer"] #0.001
		input_keep_prob=params["input_keep_prob"] #0.7
		output_keep_prob=params["output_keep_prob"] #0.6
		state_keep_prob= params["state_keep_prob"] #0.6
		embedding_keep_prob= 1.0#params["state_keep_prob"] #0.9
		dropout_rate= params["dropout_rate"] #0.5
	num_classes = params["num_classes"] #6 #pp.num_classes
	max_seq_len = params["max_seq_len"] #101 #np.max(pp.num_nodes_by_sentences)
	VOCAB_SIZE =  params["VOCAB_SIZE"] #19 #pp.POS_n_words
	labels_num = params["labels_num"] #40 #len(pp.DEP_indices)
	#print(params)
	# Input Layer
	input_data  = features["pred_input"]
	input_data_mask  = features["pred_mask"]
	original_sequence_lengths  = features["pred_original_sequence_lengths"]
	input_dep = features["pred_dep"]
	with tf.variable_scope("Predicates"):
		with tf.variable_scope("pos"):
			embedding = tf.Variable(tf.random_uniform([VOCAB_SIZE, input_dim], -1, 1),trainable=True)
			with tf.name_scope("pos_dropout"):
				embedding = tf.nn.dropout(embedding, keep_prob=embedding_keep_prob, noise_shape=[VOCAB_SIZE,1])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
		with tf.variable_scope("dep"):
			dep_embedding = tf.Variable(tf.random_uniform([dep_vocab_size, input_dim], -1, 1),trainable=True)
			with tf.name_scope("dep_dropout"):
				dep_embedding = tf.nn.dropout(dep_embedding, keep_prob=embedding_keep_prob, noise_shape=[dep_vocab_size,1])
			inputs_dep = tf.nn.embedding_lookup(dep_embedding, input_dep)
		with tf.variable_scope("mask",reuse=tf.AUTO_REUSE):
			predicate_embedding = tf.Variable(tf.random_uniform([3, input_dim], -1, 1),trainable=True)
			predicate_mask = tf.nn.embedding_lookup(predicate_embedding, input_data_mask)
		inputs = tf.concat([inputs,inputs_dep], axis=-1)
		inputs = tf.concat([inputs,predicate_mask], axis=-1)
		reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['vocab_path'],delimiter="\t")

		with tf.name_scope("BiLSTM"):
				with tf.variable_scope('forward',reuse=tf.AUTO_REUSE):
						lstm_fw_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=1.0)
						lstm_fw_cell_dropout  = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_fw_cell,input_size=inputs.shape[-1],
																			   input_keep_prob=input_keep_prob,
																			   output_keep_prob=output_keep_prob,
																			   state_keep_prob=state_keep_prob,
																			 variational_recurrent=True,dtype=tf.float32)
				with tf.variable_scope('backward',reuse=tf.AUTO_REUSE):
						lstm_bw_cell = tf.contrib.rnn.LSTMBlockCell(num_units, forget_bias=1.0)
						lstm_bw_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(cell = lstm_bw_cell,input_size=inputs.shape[-1],
																			input_keep_prob=input_keep_prob,
																			   output_keep_prob=output_keep_prob,
																			   state_keep_prob=state_keep_prob,
																			   variational_recurrent=True,dtype=tf.float32)
				(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell_dropout, 
																					 cell_bw=lstm_bw_cell_dropout,
																					 inputs=inputs,
																					 sequence_length=original_sequence_lengths, 
																					 dtype=tf.float32,
																					 scope="BiLSTM")
		outputs = tf.reduce_mean(tf.concat([tf.expand_dims(output_fw, axis=3), tf.expand_dims(output_bw, axis=3)], axis=-1),axis=-1)


		with tf.variable_scope("proj",reuse=tf.AUTO_REUSE):
				W = tf.get_variable("W", dtype=tf.float32,shape=[num_units, num_classes], initializer=tf.initializers.orthogonal(gain=0.6),regularizer = tf.contrib.layers.l2_regularizer(scale=scale_regularizer))
				b = tf.get_variable("b", shape=[num_classes],dtype=tf.float32, initializer=tf.zeros_initializer())
				proj_outputs = tf.reshape(outputs, [-1, num_units])
				pred = tf.matmul(proj_outputs, W) + b
				pred = tf.nn.dropout(pred,dropout_rate)
				logits = tf.reshape(pred, [-1, max_seq_len, num_classes])
		with tf.variable_scope("CRF",reuse=tf.AUTO_REUSE):
				#probabilities = tf.cast(tf.nn.softmax(logits), tf.float32)
				#preds = tf.argmax(probabilities, -1)
				crf_params = tf.get_variable("transition", [num_classes, num_classes], dtype=tf.float32)
				preds, scores = tf.contrib.crf.crf_decode(logits, crf_params, original_sequence_lengths)
				
		arc = tf.reshape(preds, [-1, max_seq_len, 1])
		with tf.variable_scope("pred_arc",reuse=tf.AUTO_REUSE):
			pred_arc_embedding = tf.Variable(tf.random_uniform([num_classes, input_dim], -1, 1),trainable=True)
			pred_arc_mask = tf.squeeze(tf.nn.embedding_lookup(pred_arc_embedding, tf.cast(arc, tf.int32)),[2])

		#lab_b = tf.expand_dims(preds,axis=2)
		#lab_b = tf.reshape(tf.tile(lab_b,tf.constant([1,1,num_units], tf.int32)),[-1,max_seq_len,num_units])
		#cond = tf.math.greater(lab_b, tf.constant([2]))
		#dum = tf.zeros_like(outputs)
		#positions_v = tf.where(cond,outputs,dum)
		#p = tf.reshape(tf.reduce_sum(positions_v,1),[-1,1,num_units])
		#p = tf.reshape(tf.tile(p,tf.constant([1,max_seq_len,1], tf.int32)),[-1,max_seq_len,num_units])
		proj_outputs = tf.concat([outputs,pred_arc_mask], axis=-1)
		#proj_outputs = tf.concat([proj_outputs,p], axis=-1)
		proj_outputs = tf.reshape(proj_outputs, [-1, max_seq_len, 1*num_units+input_dim])
		#proj_outputs,_= PositionAttentionLayer(preds,proj_outputs,proj_outputs,proj_outputs.shape[-1],max_seq_len,original_sequence_lengths,batch_size,dropout_rate,scale_regularizer,name="PA")
		
		with tf.variable_scope("proj_2",reuse=tf.AUTO_REUSE):
				W_2 = tf.get_variable("W_2", dtype=tf.float32,shape=[1*num_units+input_dim, 2], initializer=tf.initializers.orthogonal(gain=0.6),regularizer = tf.contrib.layers.l2_regularizer(scale=scale_regularizer))
				b_2 = tf.get_variable("b_2", shape=[2],dtype=tf.float32, initializer=tf.zeros_initializer())
				proj_outputs = tf.reshape(proj_outputs, [-1, 1*num_units+input_dim])
				pred_arc = tf.matmul(proj_outputs, W_2) + b_2
				pred_arc = tf.nn.dropout(pred_arc,dropout_rate)
				logits_arc = tf.reshape(pred_arc, [-1, max_seq_len, 2])
		with tf.variable_scope("CRF_2",reuse=tf.AUTO_REUSE):
				crf_params_2 = tf.get_variable("transition_2", [2, 2], dtype=tf.float32)
				preds_arc, scores_arc = tf.contrib.crf.crf_decode(logits_arc, crf_params_2, original_sequence_lengths)
				
		predictions = {
				# Generate predictions (for PREDICT and EVAL mode)
				"classes": preds,
				"arc":preds_arc,
				"outputs" : outputs,
				"tags":  reverse_vocab_tags.lookup(tf.to_int64(preds)) ,
				"probabilities":   tf.identity(scores_arc, name="viterbi"),
				"sequence_lenghts": original_sequence_lengths
			}
		# PREDICT mode
		if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(
					mode=mode,
					predictions=predictions,
					export_outputs={
						'predict': tf.estimator.export.PredictOutput(predictions)
					})
		weights = tf.sequence_mask(original_sequence_lengths,maxlen=max_seq_len,dtype=tf.float32)
		with tf.name_scope("CRF"):
			log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, original_sequence_lengths, crf_params)
			loss = tf.reduce_mean(-log_likelihood/tf.cast(original_sequence_lengths,tf.float32))
			#pr_mask = tf.math.logical_not(tf.math.equal(preds_arc, 0))
			#pr_mask = tf.math.logical_or(tf.math.greater(labels, tf.constant([2])),tf.math.greater(tf.to_int32(preds), tf.constant([2])))
			#pr_mask = tf.math.greater(labels, tf.constant([2]))
			#pr_mask = tf.cast(pr_mask, dtype=tf.float32)
			#weights = tf.multiply(weights,pr_mask)
			#print(weights.shape)
			#labels_one =  tf.reshape(tf.one_hot(labels, num_classes), [-1, max_seq_len, num_classes])
			#loss_pr = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels_one,logits=logits, label_smoothing=0,weights=weights)
			#loss_pr = tf.reduce_mean(loss_pr/tf.cast(tf.reduce_sum(weights),tf.float32))
			#print(loss_pr.shape)
		labels_arc = features["labels_arc"]
		with tf.name_scope("CRF_2"):
			log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits_arc, labels_arc, original_sequence_lengths, crf_params_2)
			loss_arc = tf.reduce_mean(-log_likelihood/tf.cast(original_sequence_lengths,tf.float32))

		loss =  0.5*loss + 0.5*loss_arc

		if(scale_regularizer!=0.0):
			loss +=  tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=scale_regularizer), tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		# Configure the Training Op (for TRAIN mode)
		if mode == tf.estimator.ModeKeys.TRAIN:
				optimizer = tf.train.AdamOptimizer(learning_rate)
				train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
				return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	indices = params["indices"]
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, predictions=predictions,
				export_outputs={
					'predict': tf.estimator.export.PredictOutput(predictions)
				})
								
max_seq_len = 101

def serving_input_pred():
	"""
	This is used to define inputs to serve the model.
	:return: ServingInputReciever
	"""
	reciever_tensors = {
		'pred_input' : tf.placeholder(shape=[None, None], dtype=tf.int32, name='pred_input'),
		'pred_mask' : tf.placeholder(shape=[None, None], dtype=tf.int32, name='pred_mask'),
		'pred_original_sequence_lengths' : tf.placeholder(shape=[None], dtype=tf.int32, name='pred_original_sequence_lengths'),
		'pred_dep' : tf.placeholder(tf.int32, [None, None], name="pred_dep"),
		'labels_arc' : tf.placeholder(tf.int32, [None, None], name="labels_arc")
	}

	# Convert give inputs to adjust to the model.
	features = {
		'pred_input': reciever_tensors['pred_input'],
		'pred_mask': reciever_tensors['pred_mask'],
		'pred_original_sequence_lengths': reciever_tensors['pred_original_sequence_lengths'],
		'pred_dep': reciever_tensors['pred_dep'],
		'labels_arc':reciever_tensors['labels_arc']
	}
	#TensorServingInputReceiver
	#ServingInputReceiver  
	return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,features=features)

def serving_input_arg():
	"""
	This is used to define inputs to serve the model.
	:return: ServingInputReciever
	"""
	reciever_tensors = {
		'arg_input' : tf.placeholder(shape=[None, None], dtype=tf.int32, name='arg_input'),
		'arg_mask' : tf.placeholder(shape=[None, None], dtype=tf.int32, name='arg_mask'),
		'arg_original_sequence_lengths' : tf.placeholder(shape=[None], dtype=tf.int32, name='arg_original_sequence_lengths'),
		'arg_dep' : tf.placeholder(tf.int32, [None, None], name="arg_dep"),
		'prop_mask' : tf.placeholder(tf.int32, [None, None], name="prop_mask")
	}

	# Convert give inputs to adjust to the model.
	features = {
		'arg_input': reciever_tensors['arg_input'],
		'arg_mask': reciever_tensors['arg_mask'],
		'arg_original_sequence_lengths': reciever_tensors['arg_original_sequence_lengths'],
		'arg_dep': reciever_tensors['arg_dep'],
		'prop_mask':reciever_tensors['prop_mask']
	}
	#TensorServingInputReceiver
	#ServingInputReceiver  
	return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,features=features)
