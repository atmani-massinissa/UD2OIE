import subprocess
from subprocess import *
import tensorflow as tf
import numpy as np
import math
import random
from sparse import COO
import warnings
import os
import logging
warnings.simplefilter(action='ignore')
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import uuid
import time
import stanza
import difflib
import pickle
from pathlib import Path
from tensorflow.contrib import predictor
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
#logging.getLogger('tensorflow').setLevel(tf.compat.v1.logging.INFO)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
#tf.autograph.set_verbosity(1)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class PreProcessing :
    def __init__(self):
        self.edges_indexes_collection = list()
        self.edges_type_collection = list()
        self.nodes_type_collection = list()
        self.nodes_indexes_collection = list()
        self.edges_type_indexes_collection = list()
        self.edges_embd_collection = list()
        self.edges_embd_indexes_collection = list()
        self.num_nodes_by_sentences = np.array([],dtype=np.int32)
        self.position_predicates_by_sentence = np.array([],dtype=np.int32)
        self.symbols = {0: 'PAD',1: 'UNK',2:'SOS'}
        self.vocab = []
        self.words_type_collection = list()
        self.words_indexes_collection = list()
        self.WORD_word2index = {"PAD":0,"UNK":1,"SOS":2}
        self.WORD_index2word = {0:"PAD",1:"UNK",2:"SOS"}
        self.WORD_word2count = {"PAD":0,"UNK":0,"SOS":0}
        self.WORD_indices = [0,1,2]
        self.WORD_n_words = len(self.WORD_word2index)
        self.POS_word2index = {"PAD":0,"SOS":1}
        self.POS_index2word = {0:"PAD",1:"SOS"}
        self.POS_word2count = {"PAD":0,"SOS":0}
        self.POS_indices = [0,1]
        self.POS_n_words = len(self.POS_word2index)
        self.DEP_word2index = {}
        self.DEP_index2word = {}
        self.DEP_word2count = {}
        self.DEP_indices = list()
        self.DEP_n_words = len(self.DEP_word2index)
        self.DEP_embd_word2index = {"PAD":0,"SOS":1}
        self.DEP_embd_index2word = {0:"PAD",1:"SOS"}
        self.DEP_embd_word2count = {"PAD":0,"SOS":0}
        self.DEP_embd_indices = [0,1]
        self.DEP_embd_n_words = len(self.DEP_embd_word2index)
        self.seq2seq_labels = list()
        self.predicate = list()
        self.arg = list()
        self.arc = list()
        self.mask = list()
        self.pred = list()
        self.pred_list = np.array([])
        self.sequence = np.array([])
        self.tag_sequence = np.array([])
        self.edges = list()
        self.coref = list()
        self.sentences = np.array([])

    def preprocess(self):
        self.create_structures("data/web_STANZA.txt","utf-8")
        self.create_structures("data/nyt_STANZA.txt","utf-8")
        self.create_structures("data/wiki_STANZA.txt","utf-8")
        self.create_structures("data/DEV1_STANZA.txt","utf-8")
        #self.create_structures("data/french_STANZA.txt")
        self.create_structures("data/Re-OIE2016_STANZA.txt","utf-8")
        self.create_structures("data/missing_STANZA.txt","utf-8")
        self.create_structures("data/french_STANZA.txt","utf-8")
        self.create_structures("data/spanish_STANZA.txt","iso-8859-1")
        self.create_structures("data/portuguese_STANZA.txt","iso-8859-1")
        self.create_structures("data/expert2_STANZA.txt","utf-8")
        
        self.create_ground_truth("data/web_seq2seq_.txt")
        self.create_ground_truth("data/nyt_seq2seq_.txt")
        self.create_ground_truth("data/wiki_seq2seq_.txt")
        self.create_ground_truth("data/DEV1_seq2seq_.txt")
        #self.create_ground_truth("data/french_seq2seq_.txt")
        self.create_ground_truth("data/Re-OIE2016_seq2seq_.txt")
        self.create_ground_truth("data/missing_seq2seq_.txt")
        self.create_ground_truth("data/french_seq2seq_.txt")
        self.create_ground_truth("data/expert_seq2seq_.txt")
        
        self.create_predicate("data/web_pred.txt")
        self.create_predicate("data/nyt_pred.txt")
        self.create_predicate("data/wiki_pred.txt")
        self.create_predicate("data/DEV1_pred.txt")
        #self.create_predicate("data/french_pred.txt")
        self.create_predicate("data/Re-OIE2016_pred.txt")
        self.create_predicate("data/missing_pred.txt")
        self.create_predicate("data/french_pred.txt")
        self.create_predicate("data/expert_pred.txt")
        
        self.create_arg("data/web_arg.txt")
        self.create_arg("data/nyt_arg.txt")
        self.create_arg("data/wiki_arg.txt")
        self.create_arg("data/DEV1_arg.txt")
        #self.create_arg("data/french_arg.txt")
        self.create_arg("data/Re-OIE2016_arg.txt")
        self.create_arg("data/missing_arg.txt")
        self.create_arg("data/french_arg.txt")
        self.create_arg("data/expert_arg.txt")

        self.create_arc("data/web_arc.txt")
        self.create_arc("data/nyt_arc.txt")
        self.create_arc("data/wiki_arc.txt")
        self.create_arc("data/DEV1_arc.txt")
        #self.create_arg("data/french_arc.txt")
        self.create_arc("data/Re-OIE2016_arc.txt")
        self.create_arc("data/missing_arc.txt")
        self.create_arc("data/french_arc.txt")
        self.create_arc("data/expert_arc.txt")

        self.create_mask("data/web_bin.txt")
        self.create_mask("data/nyt_bin.txt")
        self.create_mask("data/wiki_bin.txt")
        self.create_mask("data/DEV1_bin.txt")
        #self.create_mask("data/french_bin.txt")
        self.create_mask("data/Re-OIE2016_bin.txt")
        self.create_mask("data/missing_bin.txt")
        self.create_mask("data/french_bin.txt")
        self.create_mask("data/expert_bin.txt")
        
        self.create_predicate_positions("data/web_pa_.txt")
        self.create_predicate_positions("data/nyt_pa_.txt")
        self.create_predicate_positions("data/wiki_pa_.txt")
        self.create_predicate_positions("data/DEV1_pa_.txt")
        #self.create_predicate_positions("data/french_pa_.txt")
        self.create_predicate_positions("data/Re-OIE2016_pa_.txt")
        self.create_predicate_positions("data/missing_pa_.txt")
        self.create_predicate_positions("data/french_pa_.txt")
        self.create_predicate_positions("data/spanish_pa_.txt")
        self.create_predicate_positions("data/portuguese_pa_.txt")
        self.create_predicate_positions("data/expert_pa_.txt")
        
        self.map_to_numerical_labels_with_Tockenizer()
        self.create_POS_Vocabulary()
        self.create_DEP_Vocabulary()
        self.create_DEP_embd_Vocabulary()
        self.map_structures_to_indexes()
        self.predicate_embedding(np.max(self.num_nodes_by_sentences))
        self.write_sentences("data/carb_s.txt","utf-8")
        self.write_sentences("data/french_s.txt","utf-8")
        self.write_sentences("data/spanish_sent_pos_.txt","iso-8859-1")
        self.write_sentences("data/portuguese_sent_pos_.txt","iso-8859-1")
        self.write_sentences("data/expert_sent.txt","utf-8")
        self.write_coref("data/coref.txt")
        
        train_size= 2761+945
        test_size = 2761+945+1922+1783+1498+1503
        self.split_train_test_set(train_size,train_size,test_size)
        self.sequence = self.pad_data_with_tf(self.nodes_indexes_collection,np.max(self.num_nodes_by_sentences))
        self.tag_sequence = self.pad_data_with_tf(self.numerical_labels,np.max(self.num_nodes_by_sentences))
        self.dep_sequence = self.pad_data_with_tf(self.edges_embd_indexes_collection,np.max(self.num_nodes_by_sentences))
        self.arg_sequence = self.pad_data_with_tf(self.arg_seq,np.max(self.num_nodes_by_sentences))
        self.arc_sequence = self.pad_data_with_tf(self.arc,np.max(self.num_nodes_by_sentences))
        self.pred_sequence = self.pad_data_with_tf(self.pred_seq,np.max(self.num_nodes_by_sentences))
        self.mask_sequence = self.pad_data_with_tf(self.mask,np.max(self.num_nodes_by_sentences))
        self.predicate_mask = self.pad_data_with_tf(self.predicate_mask,np.max(self.num_nodes_by_sentences))
        self.word_sequence = self.pad_data_with_tf(self.words_indexes_collection,np.max(self.num_nodes_by_sentences))
        self.propagated_mask = self.predicate_embedding_propagate(np.array(self.mask)[:7411],np.max(self.num_nodes_by_sentences),np.array(self.edges_indexes_collection)[:7411],np.array(self.num_nodes_by_sentences)[:7411])
        self.propagated_mask = self.pad_data_with_tf(self.propagated_mask,np.max(self.num_nodes_by_sentences))
        
        pickle.dump(self.words_tokenizer, open("words_tokenizer.p","wb"))
        pickle.dump(self.pred_tokenizer, open("pref_tokenizer.p","wb"))
        pickle.dump(self.arg_tokenizer, open("arg_tokenizer.p","wb"))
        #pickle.dump(self.bin_tokenizer, open("bin_tokenizer.p","wb"))
        pickle.dump(self.POS_word2index, open( "POS_word2index.p", "wb" ) )
        pickle.dump(self.DEP_word2index, open( "DEP_word2index.p", "wb" ) )
        pickle.dump(self.index_tag, open( "index_tag.p", "wb" ) )
        pickle.dump(self.DEP_embd_word2index, open( "DEP_embd_word2index.p", "wb" ) )
        pickle.dump(self.WORD_word2index, open( "WORD_word2index.p", "wb" ) )       
    def train(self):
        from Model3 import predicate,argument,serving_input_pred,serving_input_arg
        tf.app.flags.DEFINE_string('pred_dir', os.path.join("UD2OIE_pred","ckpt"), "Dir to save a model and checkpoints")
        tf.app.flags.DEFINE_string('saved_pred_dir', os.path.join("UD2OIE_pred","pb"), "Dir to save a model for TF serving")
        tf.app.flags.DEFINE_string('arg_dir', os.path.join("UD2OIE_Arg","ckpt"), "Dir to save a model and checkpoints")
        tf.app.flags.DEFINE_string('saved_arg_dir', os.path.join("UD2OIE_Arg","pb"), "Dir to save a model for TF serving")
        FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_string('f', '', 'kernel')
        vocab_path = "index_tag.txt"
        with  open(vocab_path,"w",encoding="utf-8") as f:
            for k,v in self.index_tag.items():
                f.write(str(k)+"\t"+v+"\n")
        arg_params= {
        'pp' : self,
        'learning_rate' : 0.001,
        'input_dim' : 20,
        'num_units' : 200,
        'gcn_units' : 128,
        'indices': [v for k,v in self.tag_index.items() if (k!="__PADDING__"   and k!="SOS")],
        'batch_size': 5,
        'vocab_path': vocab_path,
        'scale_regularizer': 0.001,
        'input_keep_prob': 0.7,
        'output_keep_prob': 0.7,
        'state_keep_prob': 0.7,
        'dropout_rate': 0.7,
        'num_classes' : self.num_arg,
        'max_seq_len' : np.max(self.num_nodes_by_sentences),
        'VOCAB_SIZE' : self.POS_n_words,
        'word_vocab_size': self.WORD_n_words,
        'dep_vocab_size': self.DEP_embd_n_words,
        'input_dim_word': 50,
        'labels_num' : len(self.DEP_indices)
		
        }
        pred_params= {
        'pp' : self,
        'learning_rate' : 0.001,
        'input_dim' : 20,
        'num_units' : 128,
        'gcn_units' : 128,
        'indices': [v for k,v in self.tag_index.items() if (k!="__PADDING__"   and k!="SOS")],
        'batch_size': 5,
        'vocab_path': vocab_path,
        'scale_regularizer': 0.0,
        'input_keep_prob': 0.7,
        'output_keep_prob': 0.7,
        'state_keep_prob': 0.6,
        'dropout_rate': 0.7,
        'num_classes' : self.num_pred,
        'max_seq_len' : np.max(self.num_nodes_by_sentences),
        'VOCAB_SIZE' : self.POS_n_words,
        'word_vocab_size': self.WORD_n_words,
        'dep_vocab_size': self.DEP_embd_n_words,
        'input_dim_word': 50,
        'labels_num' : len(self.DEP_indices)
        }
        tf.reset_default_graph()
        # Create the Estimator
        training_pred_config = tf.estimator.RunConfig(
            model_dir=FLAGS.pred_dir,
            save_summary_steps=742,
            save_checkpoints_steps=742)
        training_arg_config = tf.estimator.RunConfig(
            model_dir=FLAGS.arg_dir,
            save_summary_steps=742,
            save_checkpoints_steps=742)

        classifier_predicate = tf.estimator.Estimator(
            model_fn=predicate,
            model_dir=FLAGS.pred_dir,
            config=training_pred_config,
            params=pred_params)

        classifier_argument = tf.estimator.Estimator(
            model_fn=argument,
            model_dir=FLAGS.arg_dir,
            config=training_arg_config,
            params=arg_params)

        # Train the model
        train_pred_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'pred_input': self.sequence[self.train_index.astype(int)].astype(int),
                'pred_original_sequence_lengths': self.num_nodes_by_train_sentences.astype(int),
                'pred_mask':self.predicate_mask[self.train_index.astype(int)].astype(int),
                'pred_dep':self.dep_sequence[self.train_index.astype(int)],
                'labels_arc':self.arc_sequence[self.train_index.astype(int)].astype(int)
            },
            y=self.pred_sequence[self.train_index.astype(int)].astype(int),
            batch_size=5,
            num_epochs=1,
            shuffle=True
        )
        
        #self.propagated_mask = self.predicate_embedding_propagate(self.mask,np.max(self.num_nodes_by_sentences),self.edges_indexes_collection,self.num_nodes_by_sentences)
        #self.propagated_mask = self.pad_data_with_tf(self.propagated_mask,np.max(self.num_nodes_by_sentences))
        #print(self.propagated_mask.shape)
        #self.pred_list[self.train_index.astype(int)].astype(int)
        new_mask = copy.deepcopy(self.mask)
        for k in self.train_index:
            for l in range(len(new_mask[k])):
                if(self.arc_sequence[self.train_index.astype(int)][k][l]==1
				 and new_mask[k][l]!=2
				  and new_mask[k][l]!=0):
                    new_mask[k][l]=3
        new_mask = self.pad_data_with_tf(new_mask,np.max(self.num_nodes_by_sentences))[self.train_index.astype(int)].astype(int)
        train_arg_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'arg_input': self.sequence[self.train_index.astype(int)].astype(int),
                'arg_original_sequence_lengths': self.num_nodes_by_train_sentences.astype(int),
                'arg_mask':new_mask.astype(int),
				'prop_mask':self.propagated_mask[self.train_index.astype(int)].astype(int),
                'arg_dep':self.dep_sequence[self.train_index.astype(int)]
            },
            y=self.arg_sequence[self.train_index.astype(int)].astype(int),
            batch_size=5,
            num_epochs=1,
            shuffle=True
        )
        for k in range(0,41):
            if(k<8):
                classifier_predicate.train(
                    input_fn=train_pred_fn,
                    steps=None)

                classifier_argument.train(
                    input_fn=train_arg_fn,
                    steps=None)

            else:
                #if(k>20):
                     #decay_rate = 0.001 / 40
                     #arg_params['learning_rate'] *= (1. / (1. + decay_rate * k))#0.0005
                     #classifier_argument = tf.estimator.Estimator(
					 #      model_fn=argument,
					 #      model_dir=FLAGS.arg_dir,
					 #      config=training_arg_config,
					 #      params=arg_params)
                #print(arg_params)
                #self.restore(os.path.join("Multi_en_Arg/pb"),os.path.join("Multi_en_Pred_0505_hyper/pb"),k-1)
                classifier_predicate.train(input_fn=train_pred_fn,steps=None)
                export_pred_dir = classifier_predicate.export_saved_model(os.path.join(FLAGS.saved_pred_dir,str(k)),serving_input_receiver_fn=serving_input_pred)
                classifier_argument.train(input_fn=train_arg_fn,steps=None)
                export_arg_dir = classifier_argument.export_saved_model(os.path.join(FLAGS.saved_arg_dir,str(k)),serving_input_receiver_fn=serving_input_arg)
                #self.evaluate(os.path.join("UD2OIE_Arg/pb"),os.path.join("UD2OIE_pred/pb"),k)
                #self.restore(os.path.join("UD2OIE_Arg/pb"),os.path.join("UD2OIE_pred/pb"),k)

    def reset(self):
        self.edges_indexes_collection = list()
        self.edges_type_collection = list()
        self.nodes_type_collection = list()
        self.nodes_indexes_collection = list()
        self.words_type_collection = list()
        self.words_indexes_collection = list()
        self.edges_type_indexes_collection = list()
        self.edges_embd_collection = list()
        self.edges_embd_indexes_collection = list()
        self.num_nodes_by_sentences = np.array([],dtype=np.int32)
        self.position_predicates_by_sentence = np.array([],dtype=np.int32)
        self.candidate = list()
        self.predicate_mask  = np.array([])
        self.arg = list()
        self.arc = list()
        self.mask = list()
        self.pred = list()
        self.coref = list()
        self.sentences = np.array([])

    def create_POS_Vocabulary(self):
        for s in (self.nodes_type_collection) :
            self.addSentence(s,"POS")
            
    def load_POS_Vocabulary(self,file):
        self.POS_word2index = pickle.load( open( file, "rb" ))
        self.POS_index2word = {i:w for w,i in self.POS_word2index.items()}
    
    def create_DEP_Vocabulary(self):
        for s in (self.edges_type_collection) :
            self.addSentence(s,"DEP")
    
    def load_DEP_Vocabulary(self,file):
        self.DEP_word2index = pickle.load( open( file, "rb" ))
        self.DEP_index2word = {i:w for w,i in self.DEP_word2index.items()}
     
    def create_DEP_embd_Vocabulary(self):
        for s in (self.edges_embd_collection) :
            self.addSentence(s,"DEP_embd")
    
    def load_DEP_embd_Vocabulary(self,file):
        self.DEP_embd_word2index = pickle.load( open( file, "rb" ))
        self.DEP_embd_index2word = {i:w for w,i in self.DEP_embd_word2index.items()}
        
    def create_WORD_Vocabulary(self,train_offset):
        for s in (self.words_type_collection[:train_offset]) :
            self.addSentence(s,"WORD")
    
    def load_WORD_Vocabulary(self,file):
        self.WORD_word2index = pickle.load( open( file, "rb" ))
        self.WORD_index2word = {i:w for w,i in self.WORD_word2index.items()}
        
    def load_tag_Vocabulary(self,file):
        self.index_tag = pickle.load( open( file, "rb" ))
        self.tag_index = {i:w for w,i in self.index_tag.items()}
        self.num_classes = len(self.tag_index)
        
    def load_Vocabulary(self,file):
        self.vocab = pickle.load( open( file, "rb" ))      
    def load_tockenizer(self,file):
        self.words_tokenizer = pickle.load( open( file, "rb" ))
    def load_pred_tockenizer(self,file):
        self.pred_tokenizer = pickle.load( open( file, "rb" ))
    def load_arg_tokenizer(self,file):
        self.arg_tokenizer = pickle.load( open( file, "rb" ))
    def load_bin_tokenizer(self,file):
        self.bin_tokenizer = pickle.load( open( file, "rb" ))
        
    def addSentence(self, sentence_array, gr):
        for word in sentence_array:
            self.addWord(word,gr)

    def addWord(self, word, gr):
        if(gr=="WORD"):
            if word not in self.vocab:
                self.vocab.append(word.lower())
            if word not in self.WORD_word2index:
                self.WORD_word2index[word] = self.WORD_n_words
                self.WORD_word2count[word] = 1
                self.WORD_index2word[self.WORD_n_words] = word
                self.WORD_indices.append(self.WORD_n_words)
                self.WORD_n_words += 1
            else:
                self.WORD_word2count[word] += 1
        elif(gr=="POS"):
            if word not in self.POS_word2index:
                self.POS_word2index[word] = self.POS_n_words
                self.POS_word2count[word] = 1
                self.POS_index2word[self.POS_n_words] = word
                self.POS_indices.append(self.POS_n_words)
                self.POS_n_words += 1
            else:
                self.POS_word2count[word] += 1
        elif(gr=="DEP"):
            if word not in self.DEP_word2index:
                self.DEP_word2index[word] = self.DEP_n_words
                self.DEP_word2count[word] = 1
                self.DEP_index2word[self.DEP_n_words] = word
                self.DEP_indices.append(self.DEP_n_words)
                self.DEP_n_words += 1
            else:
                self.DEP_word2count[word] += 1
        else:
            if word not in self.DEP_embd_word2index:
                self.DEP_embd_word2index[word] = self.DEP_embd_n_words
                self.DEP_embd_word2count[word] = 1
                self.DEP_embd_index2word[self.DEP_embd_n_words] = word
                self.DEP_embd_indices.append(self.DEP_embd_n_words)
                self.DEP_embd_n_words += 1
            else:
                self.DEP_embd_word2count[word] += 1
    def addOne_hot(self,depth,gr):
        if(gr=="POS"):
            self.POS_one_hot = tf.one_hot(self.POS_indices, depth)
        else:
            self.DEP_one_hot = tf.one_hot(self.DEP_indices, depth)
            
    def generate_candidate(self, nodes_array, edges_type_array, edges_target_array, edges_source_array):
        candidate = list()
        candidate.extend([i for i, x in enumerate(nodes_array) if (x == "VERB" or x == "AUX")])
                
        for edge_type,edge_source in zip(edges_type_array, edges_source_array):
            if(edge_type == "nsubj"):
                print(edge_source)
                candidate.extend([int(edge_source-1)])# will be incremented in the predicate generation function
                
        candidate = list(set(candidate))
        candidate = list(filter( lambda x: (x-1 not in candidate) and (x-2 not in candidate) , candidate ))
        for edge_type,edge_target in zip(edges_type_array, edges_target_array):
            if(edge_type == "appos" and nodes_array[edge_target-2]=="PUNCT"): 
                candidate.extend([int(edge_target-2)])# will be incremented in the predicate generation function
            elif(edge_type == "appos" and nodes_array[edge_target-3]=="PUNCT"):
                candidate.extend([int(edge_target-3)])# will be incremented in the predicate generation function
            elif(edge_type == "appos" and nodes_array[edge_target-4]=="PUNCT"):
                candidate.extend([int(edge_target-4)])# will be incremented in the predicate generation function
            elif(edge_type == "appos" and nodes_array[edge_target-5]=="PUNCT"):
                candidate.extend([int(edge_target-5)])# will be incremented in the predicate generation function
                
        candidate = list(set(candidate))
        candidate = list(filter( lambda x: (x-1 not in candidate) and (x-2 not in candidate) , candidate ))
        
        return candidate
        
    def stanford_parse(self,sentences,lang):
        self.candidate = list()
        nlp_en = stanfordnlp.Pipeline(lang='en',use_gpu=True)
        nlp_fr = stanfordnlp.Pipeline(lang='fr',use_gpu=True)
        for sent_list,lg in zip(sentences,lang):
            if(lg=="en"): 
                doc = nlp_en(sent_list)
            else:
                doc = nlp_fr(sent_list)
            for sent in doc.sentences:
                    print(sent)
                    edges_source_array = list()
                    edges_target_array = list()
                    edges_type_array = list()
                    nodes_array = list()
                    words_array = list()
                    edges_array = list()
                    candidate = list()
                    print(sent)

                    for word in sent.words:
                        if(int(word.governor)>0):
                            edges_source_array.append(int(word.governor))
                            edges_target_array.append(int(word.index))
                            edges_type_array.append(word.dependency_relation.strip())
                        nodes_array.append(word.upos.strip())
                        words_array.append(word.text.strip())
                        edges_array.append(word.dependency_relation.strip())

                    c = self.generate_candidate(nodes_array, edges_type_array, edges_target_array, edges_source_array )
                    candidate.append(len(c))
                    self.candidate.extend(c)
            for c in candidate:
                for _ in range(c):
                    self.edges_indexes_collection.append([edges_source_array,edges_target_array])
                    self.edges_type_collection.append(edges_type_array)
                    self.edges_embd_collection.append(np.append("SOS",edges_array))
                    self.nodes_type_collection.append(np.append("SOS",nodes_array))
                    self.words_type_collection.append(np.append("SOS",words_array))
                            
    def create_structures(self,file,encoding):
        edges_source_array = list()
        edges_target_array = list()
        edges_type_array = list()
        nodes_array = list()
        words_array = list()
        edges_array = list()
        f = open(file,encoding=encoding)
        for line in f:
                if( not line.strip()):
                    if(len(edges_source_array)>0):
                        self.edges_indexes_collection.append([edges_source_array,edges_target_array])
                        self.edges_type_collection.append(edges_type_array)

                    if(len(nodes_array)>0):
                        self.nodes_type_collection.append(np.append("SOS",nodes_array))
                        self.words_type_collection.append(np.append("SOS",words_array))
                        self.edges_embd_collection.append(np.append("SOS",edges_array))
                    edges_source_array = list()
                    edges_target_array = list()
                    edges_type_array = list()
                    nodes_array = list()
                    words_array = list()
                    edges_array = list()
                    morph_array = list()
                else :
                    syntactic_analysis = line.strip().split("\t")
                    if(int(syntactic_analysis[6].strip())>0):
                        edges_source_array.append(int(syntactic_analysis[6].strip()))
                        edges_target_array.append(int(syntactic_analysis[0].strip()))
                        edges_type_array.append(syntactic_analysis[7].strip())
                    morph = None
                    if(syntactic_analysis[5].strip()!="None"):
                        morph = syntactic_analysis[5].strip().split("|") if ("|" in syntactic_analysis[5].strip()) else [syntactic_analysis[5].strip()]
                    pos = syntactic_analysis[3].strip()
                    
                    if (morph is not None):
                        for m in morph:
                            #if("Mood=" not in m and "Poss=" not in m and "Person=" not in m 
                            #   and "Reflex=" not in m and "Gender=" not in m  and "Foreign=" not in m):
                            if( "PronType=" in m):
                                if("," in m.split("=")[1].strip()):
                                     pos = pos+"_"+(m.split("=")[1].strip().split(",")[1].strip())
                                else:
                                    if( m.split("=")[1].strip() not in ("Neg","Ind","Tot","Emp")):
                                        pos = pos+"_"+m.split("=")[1].strip()
                                    elif( pos=="DET" and m.split("=")[1].strip() in ("Neg")):
                                        pos = pos+"_"+m.split("=")[1].strip()
                                if(pos=="DET_Rel"):
                                    pos="DET"
                                    #print("DET_Rel",self.words_type_collection[-1],words_array)
                                if(pos=="PROPN_Art"):
                                    pos="PROPN"
                                    #print("PROPN_Art",self.words_type_collection[-1],words_array)
                    nodes_array.append(pos)
                    words_array.append(syntactic_analysis[1].strip())
                    edges_array.append(syntactic_analysis[7].strip())

        self.edges_indexes_collection.append([edges_source_array,edges_target_array])
        self.edges_type_collection.append(edges_type_array)
        self.edges_embd_collection.append(np.append("SOS",edges_array))
        self.nodes_type_collection.append(np.append("SOS",nodes_array))
        self.words_type_collection.append(np.append("SOS",words_array))

    def map_structures_to_indexes(self):
        for l in self.nodes_type_collection:
            temp_nodes = list()
            for n in l:
                temp_nodes.append(self.POS_word2index[n])
            self.nodes_indexes_collection.append(temp_nodes)
        for l in self.edges_type_collection:
            temp_edges = list()
            for n in l:
                if(n in self.DEP_word2index):
                    temp_edges.append(self.DEP_word2index[n])
                else:
                    print("not here",n,n.split(":")[0])
                    if(n.split(":")[0] in self.DEP_word2index):
                        temp_edges.append(self.DEP_word2index[n.split(":")[0]])
            self.edges_type_indexes_collection.append(temp_edges)
        for l in self.edges_embd_collection:
            embd_edges = list()
            for n in l:
                embd_edges.append(self.DEP_embd_word2index[n])
            self.edges_embd_indexes_collection.append(embd_edges)
        self.num_nodes_by_sentences = np.append(self.num_nodes_by_sentences,[len(s) for s in self.nodes_indexes_collection])

    def pad_data(self):
        data = list(itertools.chain(*self.nodes_indexes_collection))
        max_seq_len    = np.max(self.num_nodes_by_sentences)
        self.data_pad  = np.zeros([len(self.num_nodes_by_sentences), max_seq_len], dtype=np.int32)
        self.data_mask = np.zeros([len(self.num_nodes_by_sentences), max_seq_len], dtype=np.float32)
        offset = 0
        for i in range(len(self.num_nodes_by_sentences)):
            self.data_pad [i, :self.num_nodes_by_sentences[i]] = data[offset: offset + self.num_nodes_by_sentences[i]]
            self.data_mask[i, :self.num_nodes_by_sentences[i]] = 1
            offset += self.num_nodes_by_sentences[i]
    def pad_data_with_tf(self,data,dim):
        return  tf.keras.preprocessing.sequence.pad_sequences(data,padding='post',maxlen=dim)
    def create_ground_truth(self,file):
        f = open(file,encoding="utf-8")
        for line in f:
            if (";" in line):
                l = (line.strip().split(";")[1]).split(" ")
            else:
                l = line.strip().split(" ")
            self.seq2seq_labels.append(["SOS"]+l)
            
    def write_sentences(self,file,encoding):
        with open(file,encoding=encoding) as fp:  
            for cnt, line in enumerate(fp):
                self.sentences = np.append(self.sentences,line.strip())
    def write_coref(self,file):
        with open(file,encoding="utf-8") as fp:  
            for cnt, line in enumerate(fp):
                self.coref.append(line.strip())
        
    def create_predicate(self,file):
        f = open(file,encoding="utf-8")
        for line in f:
            if (";" in line):
                l = (line.strip().split(";")[1]).split(" ")
            else:
                l = line.strip().split(" ")
            self.pred.append(["SOS"]+l)
            
    def create_arg(self,file):
        f = open(file,encoding="utf-8")
        for line in f:
            if (";" in line):
                l = (line.strip().split(";")[1]).split(" ")
            else:
                l = line.strip().split(" ")
            self.arg.append(["SOS"]+l)
	
    def create_arc(self,file):
        f = open(file,encoding="utf-8")
        for line in f:
            if (";" in line):
                l = (line.strip().split(";")[1]).split(" ")
            else:
                l = line.strip().split(" ")
            self.arc.append([0]+l)
            
    def create_mask(self,file):
        f = open(file,encoding="utf-8")
        temp = list()
        for line in f:
            if (";" in line):
                l = (line.strip().split(";")[1]).split(" ")
                for i in range(len(l)):
                    if(l[i]=="2"):
                        temp.append(i+1)
            else:
                l = list(map(int, line.strip().split(" ")))
                for i in range(len(l)):
                    if(l[i]=="2"):
                        temp.append(i+1)
            self.pred_list = np.append(self.pred_list,temp)
            temp = list()
            self.mask.append([1]+l)
            
    def create_predicate_positions(self,file):
        f = open(file,encoding="utf-8")#self.candidate#open(file,encoding="utf-8")
        for l in f:
                v = max(0,int(l))
                self.position_predicates_by_sentence = np.append(self.position_predicates_by_sentence,v+1)
    def map_to_numerical_labels(self): 
        label_encoder = preprocessing.LabelEncoder()
        tags = label_encoder.fit(list(itertools.chain(*self.seq2seq_labels)))
        self.num_classes = len(tags.classes_)
        self.numerical_labels = [label_encoder.transform(s) for s in self.seq2seq_labels]
    def map_to_numerical_labels_with_Tockenizer(self): 
        words_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',lower=False)
        words_tokenizer.fit_on_texts(map(lambda s: ' '.join(s),self.seq2seq_labels))
        pred_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',lower=False)
        pred_tokenizer.fit_on_texts(map(lambda s: ' '.join(s),self.pred))
        arg_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',lower=False)
        arg_tokenizer.fit_on_texts(map(lambda s: ' '.join(s),self.arg))
        mask_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',lower=False)
        mask_tokenizer.fit_on_texts(map(lambda s: ' '.join(s),self.arg))
        
        self.numerical_labels = words_tokenizer.texts_to_sequences(map(lambda s: ' '.join(s),self.seq2seq_labels))
        self.pred_seq = pred_tokenizer.texts_to_sequences(map(lambda s: ' '.join(s),self.pred))
        self.arg_seq = arg_tokenizer.texts_to_sequences(map(lambda s: ' '.join(s),self.arg))

        self.tag_index = words_tokenizer.word_index
        self.words_tokenizer = words_tokenizer
        self.tag_index['__PADDING__'] = 0
        self.index_tag = {i:w for w, i in self.tag_index.items()}
        self.num_classes = len(self.tag_index)
        
        self.pred_tag_index = pred_tokenizer.word_index
        self.pred_tokenizer = pred_tokenizer
        self.pred_tag_index['__PADDING__'] = 0
        self.pred_index_tag = {i:w for w, i in self.pred_tag_index.items()}
        self.num_pred = len(self.pred_tag_index)
        
        self.arg_tag_index = arg_tokenizer.word_index
        self.arg_tokenizer = arg_tokenizer
        self.arg_tag_index['__PADDING__'] = 0
        self.arg_index_tag = {i:w for w, i in self.arg_tag_index.items()}
        self.num_arg = len(self.arg_tag_index)
     
    def map_to_bin_with_Tockenizer(self): 
        bin_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',lower=False)
        self.bin = bin_tokenizer.texts_to_sequences(map(lambda s: ' '.join(s),self.mask))
        self.bin_tag_index = bin_tokenizer.word_index
        self.bin_tokenizer = bin_tokenizer
        #self.bin_tag_index['__PADDING__'] = 0
        self.bin_index_tag = {i:w for w, i in self.bin_tag_index.items()}
        self.num_bin = len(self.bin_tag_index)
        
    def split_train_test_set(self,train_size,test_start,test_size):
        #2214#5196#3705
        train_index,test_index,val_index,val_index2 = np.arange(0,train_size),np.arange(test_start,test_size),np.arange(test_size,test_size+115),np.arange(test_size-3001,test_size+115-3001)
        self.train_index, self.test_index = np.asarray(random.sample(list(train_index),len(list(train_index)))),np.sort(np.asarray(random.sample(list(test_index),len(list(test_index)))))
        self.val_index = np.sort(np.asarray(random.sample(list(val_index),len(list(val_index)))))
        self.val_index2 = np.sort(np.asarray(random.sample(list(val_index2),len(list(val_index2)))))
        self.num_nodes_by_test_sentences = np.array([len(e) for e in np.array(self.nodes_indexes_collection)[self.test_index.astype(int)]])
        self.num_nodes_by_train_sentences = np.array([len(e) for e in np.array(self.nodes_indexes_collection)[self.train_index.astype(int)]])
        self.num_nodes_by_val_sentences = np.array([len(e) for e in np.array(self.nodes_indexes_collection)[self.val_index.astype(int)]])
            
    def batch_1(self,data,embd,labels,sequence_lengths, batch_size,arg_data,pred_data,predicate_data_mask,propag_mask):
        n_batch = int(math.ceil(len(data) / batch_size))
        index = 0
        for _ in range(n_batch):
            if(index + batch_size>len(data)):
                batch_sequence_lengths = np.array(sequence_lengths[index: len(data)])
                batch_length = np.array(data.shape[1]) # max length in batch
                batch_data =  data[index: len(data)] # pad data
                batch_labels = labels[index: len(data)] # pad labels
                batch_embd = embd[index: len(data)]
                predicate_mask = predicate_data_mask[index:  len(data)]
                prop_mask = propag_mask[index:  len(data)]
                pred = pred_data[index:  len(data)]
                arg = arg_data[index:  len(data)]
                index += batch_size
            else :
                batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
                batch_length = np.array(data.shape[1]) # max length in batch
                batch_data =  data[index: index + batch_size] # pad data
                batch_labels = labels[index: index + batch_size] # pad labels
                batch_embd = embd[index: index+ batch_size]
                predicate_mask = predicate_data_mask[index:  index + batch_size]
                prop_mask = propag_mask[index:  index + batch_size]
                pred = pred_data[index:  index + batch_size]
                arg = arg_data[index:  index + batch_size]
                index += batch_size

            yield batch_data,batch_embd, batch_labels, batch_length, batch_sequence_lengths,arg,pred,predicate_mask,prop_mask

    def positional_embedding(self,pos, model_size):
        PE = np.zeros((1, model_size))
        for i in range(model_size):
            if i % 2 == 0:
                PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
            else:
                PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
        return PE
    def predicate_embedding(self,dim):
        self.predicate_mask  = list()
        count = 0
        for p,e,l in zip(self.position_predicates_by_sentence,self.edges_indexes_collection,self.num_nodes_by_sentences) :
            mask = np.ones(l)
            mask[p] = 2
            count+=1
            self.predicate_mask.append(mask)
            
    def predicate_embedding2(self,dim):
        self.predicate_mask  = list()
        count = 0
        for p,e,l in zip(self.position_predicates_by_sentence,self.edges_indexes_collection,self.num_nodes_by_sentences) :
            mask = np.zeros(l)#np.ones(l)
            mask[p] = 1#2
            count+=1
            for s,t in zip(e[0],e[1]):
                    if(int(p)==int(s) and mask[t]!=2 and int(s)!=0 and int(t)!=0):
                        mask[t] = 1
                        #print("e:p",p,e)
                    elif(int(p)==int(t)and mask[s]!=2 and int(s)!=0 and int(t)!=0):
                        mask[s] = 1
            self.predicate_mask.append(mask)

    def predicate_embedding3(self,dim):
        self.predicate_mask  = list()
        count = 0
        for p,e,l in zip(self.position_predicates_by_sentence,self.edges_indexes_collection,self.num_nodes_by_sentences) :
            mask = np.ones(l)#np.ones(l)
            mask[p] = 2
            count+=1
            for s,t in zip(e[0],e[1]):
                    if(int(p)==int(s) and mask[t]!=2 and int(s)!=0 and int(t)!=0):
                        mask[t] = 2
                        #print("e:p",p,e)
                    elif(int(p)==int(t)and mask[s]!=2 and int(s)!=0 and int(t)!=0):
                        mask[s] = 2
            self.predicate_mask.append(mask)
        
    def predicate_embedding_propagate(self,infered_predicates,dim,edges_indexes_collection,num_nodes_by_sentences):
        propagate_mask  = list()
        for pred,e,l in zip(infered_predicates,edges_indexes_collection,num_nodes_by_sentences) :
            mask = np.ones(l).astype(int)
            pp = np.where(np.array(pred)==2)[0]
            for p in pp:
                mask[p] = 2
                for s,t in zip(e[0],e[1]):
                    if(int(p)==int(s) and mask[t]!=2 and int(s)!=0 and int(t)!=0):
                        mask[t] = 3
                    elif(int(p)==int(t)and mask[s]!=2 and int(s)!=0 and int(t)!=0):
                        mask[s] = 3
            propagate_mask.append(mask)
        return propagate_mask
        
    def predicate_word_embedding(self,dim):
        self.predicate_word  = np.array([])
        for i,p in zip(range(len(pp.words_indexes_collection)), self.position_predicates_by_sentence) :
            word_embedding = np.full((dim), pp.words_indexes_collection[i][p], dtype=int)
            self.predicate_word = np.append(self.predicate_word,word_embedding)
        self.predicate_word = self.predicate_word.reshape(-1,dim)
        
    def get_pos(self,sen,pos,pred):
        def get_overlap(s1, s2, next_s2, prec_s2):
            next_match = None
            prec_match = prec_s2
            if(next_s2!=None):
                s_next = difflib.SequenceMatcher(None, s1, next_s2)
                for match in s_next.get_matching_blocks():
                    if (match[-1] == len(next_s2) ):
                        next_match = match
                        break
            boundaries = list()
            s = difflib.SequenceMatcher(None, s1, s2)
            for match in s.get_matching_blocks():
                if (match[-1] == len(s2) ):
                    boundaries.append(match)
            if(len(boundaries)>1 and next_match!=None):
                        #print(boundaries)
                        return min(boundaries, key=lambda x:abs(x[0]-next_match[0]))
            if(len(boundaries)>1 and prec_match!=None):
                        #print(boundaries)
                        return min(boundaries, key=lambda x:abs(x[0]-prec_match[0]))
            else :
                return boundaries[0] if boundaries else [0,0,0]
        s,_,l = get_overlap(sen,pred.split(" "),None,None)
        return pos[s:s+l]

    def merge(self,pred_seq,arg_seq):
        pred_seq = [e.split(" ") for e in pred_seq]
        arg_seq = [e.split(" ") for e in arg_seq]
        seq = arg_seq
        for i,(pred,arg) in enumerate(zip(pred_seq,arg_seq)):
            for j,v in (enumerate(pred)):
                if(v!="SOS" and v!="O"):
                    #print(len(seq[i]),i,j,len(pred))
                    seq[i][j]=v
        return seq

    def sequence_to_relation(self,pred_seq,seq_len,pred_score,index):
        def rindex(mylist, myvalue):
            return len(mylist) - mylist[::-1].index(myvalue) - 1
        tags = list()
        relations = list()
        #pp.words_tokenizer.sequences_to_texts(pred_seq)
        for a,d,c,score in zip(pred_seq,seq_len,np.array(self.words_type_collection)[index],pred_score):
                relation = dict()
                relation["arg0_index"] = []
                relation["pred_index"] = []
                relation["arg1_index"] = []
                relation["arg2_index"] = []
                relation["arg3_index"] = []
                relation["loc_index"] =  []
                relation["temp_index"] = []

                relation["arg0"] = []
                relation["pred"] = []
                relation["arg1"] = []
                relation["arg2"] = []
                relation["arg3"] = []
                relation["loc"] =  []
                relation["temp"] = []
                relation["score"] = []
                tags.append(a.split(" ")[:int(d)])
                s = a.split(" ")[:int(d)]
                if("P_S P_I" in " ".join(s)):
                    s = " ".join(s).strip().replace("P_S P_I","P_B P_I").split(" ")
                if("P_S O P_I" in " ".join(s)):
                    s = " ".join(s).strip().replace("P_S O P_I","P_B P_I P_I").split(" ")
                if("P_I O P_I" in " ".join(s)):
                    s = " ".join(s).strip().replace("P_I O P_I","P_I P_I P_I").split(" ")
                arg0 = np.array(s)
                l = list()
                indx = list()
                if("A_0_S") in s:
                    for b,k in zip(np.where(arg0 == "A_0_S")[0],range(len(np.where(arg0 == "A_0_S")[0]))):
                        l.append([int(b),int(b+1)])
                        indx.append(" ".join(c[b: b+1]))
                if("A_0_B") in s:
                    for b,e,k in zip(np.where(arg0 == "A_0_B")[0],np.where(arg0 == "A_0_E")[0],range(len(np.where(arg0 == "A_0_B")[0]))):
                        l.append([int(b),int(e)])
                        indx.append(" ".join(c[b: e+1]))
                relation["arg0_index"] = l
                relation["arg0"] = indx

                if("P_S") in s:
                    relation["pred_index"].append([s.index("P_S"), rindex(s,"P_S")])
                    relation["pred"].append(" ".join(c[s.index("P_S"): rindex(s,"P_S")+1]))
                if("P_B") in s:
                    if("P_E" in s ):
                        relation["pred_index"].append([s.index("P_B"), rindex(s,"P_E")])
                        relation["pred"].append(" ".join(c[s.index("P_B"): rindex(s,"P_E")+1]))
                    else:
                        relation["pred_index"].append([s.index("P_B"), s.count("P_I")])
                        relation["pred"].append(" ".join(c[s.index("P_B"): s.count("P_I")+1]))

                arg1 = np.array(s)
                l = list()
                indx = list()
                if("A_1_S") in s:
                    for b,k in zip(np.where(arg1 == "A_1_S")[0],range(len(np.where(arg1 == "A_1_S")[0]))):
                        l.append([int(b),int(b+1)])
                        indx.append(" ".join(c[b: b+1]))
                if("A_1_B") in s:
                    for b,e,k in zip(np.where(arg1 == "A_1_B")[0],np.where(arg1 == "A_1_E")[0],range(len(np.where(arg1 == "A_1_B")[0]))):
                        l.append([int(b),int(e)])
                        indx.append(" ".join(c[b: e+1]))
                relation["arg1_index"] = l
                relation["arg1"] = indx

                arg2 = np.array(s)
                l = list()
                indx = list()
                arg3 = np.array(s)
                if("A_2_S") in s:
                    for b,k in zip(np.where(arg2 == "A_2_S")[0],range(len(np.where(arg2 == "A_2_S")[0]))):
                        if(k==0):
                            relation["arg2_index"].append([int(b),int(b+1)])
                            relation["arg2"].append(" ".join(c[b: b+1]))
                        elif(k==1):
                            relation["loc_index"].append([int(b),int(b+1)])
                            relation["loc"].append(" ".join(c[b: b+1]))
                        elif(k==2):
                            relation["temp_index"].append([int(b),int(b+1)])
                            relation["temp"].append(" ".join(c[b: b+1]))
                        #print(b,e,k)
                if("A_2_B") in s:
                    for b,e,k in zip(np.where(arg2 == "A_2_B")[0],np.where(arg2 == "A_2_E")[0],range(len(np.where(arg2 == "A_2_B")[0]))):
                        if(k==0):
                            relation["arg2_index"].append([int(b),int(e)])
                            relation["arg2"].append(" ".join(c[b: e+1]))
                        elif(k==1):
                            relation["loc_index"].append([int(b),int(e)])
                            relation["loc"].append(" ".join(c[b: e+1]))
                        elif(k==2):
                            relation["temp_index"].append([int(b),int(e)])
                            relation["temp"].append(" ".join(c[b: e+1]))

                relation["score"]= score/float(d)
                relations.append(relation)
        return relations
    
    def generate(self,relations,sentences):
        rel_dict = dict()
        for r,current_sentence in zip(relations,sentences):

            if(current_sentence.strip() not in rel_dict.keys()):
                r["context"] = current_sentence.strip()
                rel_dict[current_sentence.strip()] = np.array([r])
            else:
                r["context"] = current_sentence.strip()
                duplicate=False
                for re in rel_dict[current_sentence.strip()]:
                    if((" ".join(r["arg0"])).strip()==(" ".join(re["arg0"])).strip()
                       and (" ".join(r["pred"])).strip()==(" ".join(re["pred"])).strip()
                       and (" ".join(r["arg1"])).strip()==(" ".join(re["arg1"])).strip()
                      and (" ".join(r["arg2"])).strip()==(" ".join(re["arg2"])).strip()):
                        duplicate=True
                        break
                if(duplicate==True):
                    r["arg0"] = []
                    r["arg1"] = []
                    r["pred"] = []
                    r["arg2"] = []
                value = np.concatenate((rel_dict[current_sentence.strip()], r), axis=None)
                rel_dict.update({current_sentence.strip(): value})
        return rel_dict
    
    def evaluate(self,Arg_path,Pred_path):
        print("#########################################################################################")
        print("Validation on Expert benchmark")
        print("#########################################################################################")
        for k in range(10,15):
            print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(k)))
            self.validation(os.path.join(Arg_path,str(29)),os.path.join(Pred_path,str(k)))
            subprocess.call(["python","carb/carb.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--out=carb/res.out"])
            #subprocess.call(["python","carb/oie16.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--out=carb/res.out"])
    def evaluate(self,Arg_path,Pred_path,k):
        print("#########################################################################################")
        print("Validation on Expert benchmark")
        print("#########################################################################################")
        print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(10)))
        self.validation(os.path.join(Arg_path,str(k)),os.path.join(Pred_path,str(10)))
        subprocess.call(["python","carb/carb.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--out=carb/res.out"])
        subprocess.call(["python","carb/carb.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--single_match","--out=carb/res.out"])

            
    def predict(self,Arg_path,Pred_path):
        print("\n")
        print("#########################################################################################")
        print("Multilingual Evaluation using the CARB Evaluation Strategy")
        print("#########################################################################################")
        for k in range(8,15):
            print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(k)))
            self.inference(os.path.join(Arg_path,str(29)),os.path.join(Pred_path,str(k)),k)
            print("English")
            subprocess.call(["python","carb/carb.py","--gold=carb/extractions.tsv","--tabbed=carb/english_predictions_"+str(k)+".tsv","--out=carb/res.out"])
            print("French")
            subprocess.call(["python","carb/carb.py","--gold=carb/french_carb.tsv","--tabbed=carb/french_predictions_"+str(k)+".tsv","--out=carb/res.out"])
            print("Spanish")
            subprocess.call(["python","carb/carb.py","--gold=carb/spanish_carb.tsv","--tabbed=carb/spanish_predictions_"+str(k)+".tsv","--out=carb/res.out"])
            print("Portuguese")
            subprocess.call(["python","carb/carb.py","--gold=carb/portuguese_carb.tsv","--tabbed=carb/portuguese_predictions_"+str(k)+".tsv","--out=carb/res.out"])

    def vald_and_test(self,Arg_path,Pred_path):
        for j in range(10,16):
            for k in range(21,30):
                print("\n")
                print("#########################################################################################")
                print("Validation on Expert benchmark")
                print("#########################################################################################")
                print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(j)))
                self.validation(os.path.join(Arg_path,str(k)),os.path.join(Pred_path,str(j)))
                subprocess.call(["python","carb/carb.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--out=carb/res.out"])
                subprocess.call(["python","carb/carb.py","--gold=carb/expert.tsv","--tabbed=carb/validation.tsv","--single_match","--out=carb/res.out"])
                print("\n")
                print("#########################################################################################")
                print("Multilingual Evaluation using the CARB Evaluation Strategy")
                print("#########################################################################################")
                print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(k)))
                self.inference(os.path.join(Arg_path,str(k)),os.path.join(Pred_path,str(j)),k)
                print("English")
                subprocess.call(["python","carb/carb.py","--gold=carb/extractions.tsv","--tabbed=carb/english_predictions_"+str(k)+".tsv","--out=carb/res.out"])
                print("French")
                subprocess.call(["python","carb/carb.py","--gold=carb/french_carb.tsv","--tabbed=carb/french_predictions_"+str(k)+".tsv","--out=carb/res.out"])
                print("Spanish")
                subprocess.call(["python","carb/carb.py","--gold=carb/spanish_carb.tsv","--tabbed=carb/spanish_predictions_"+str(k)+".tsv","--out=carb/res.out"])
                print("Portuguese")
                subprocess.call(["python","carb/carb.py","--gold=carb/portuguese_carb.tsv","--tabbed=carb/portuguese_predictions_"+str(k)+".tsv","--out=carb/res.out"])

    def restore(self,Arg_path,Pred_path,k):
        print("#########################################################################################")
        print("Multilingual Evaluation using the CARB Evaluation Strategy")
        print("#########################################################################################")
        print("Argument Model :",os.path.join(Arg_path,str(k)),"Predicate Model :",os.path.join(Pred_path,str(k)))
        self.inference(os.path.join(Arg_path,str(k)),os.path.join(Pred_path,str(k)),k)
        print("English")
        subprocess.call(["python","carb/carb.py","--gold=carb/extractions.tsv","--tabbed=carb/english_predictions_"+str(k)+".tsv","--out=carb/res.out"])
        print("French")
        subprocess.call(["python","carb/carb.py","--gold=carb/french_carb.tsv","--tabbed=carb/french_predictions_"+str(k)+".tsv","--out=carb/res.out"])
        print("Spanish")
        subprocess.call(["python","carb/carb.py","--gold=carb/spanish_carb.tsv","--tabbed=carb/spanish_predictions_"+str(k)+".tsv","--out=carb/res.out"])
        print("Portuguese")
        subprocess.call(["python","carb/carb.py","--gold=carb/portuguese_carb.tsv","--tabbed=carb/portuguese_predictions_"+str(k)+".tsv","--out=carb/res.out"])
        
    def inference(self,Arg_path,Pred_path,arg):
        #os.path.join("Pred128/pb",str(14))
        #os.path.join("Arg128/pb",str(14))
        subdirs = [x for x in Path(Pred_path).iterdir() if x.is_dir() and 'temp' not in str(x)]
        p = str(sorted(subdirs)[-1])
        predict_pred = predictor.from_saved_model(p)
        subdirs = [x for x in Path(Arg_path).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
        a = str(sorted(subdirs)[-1])
        predict_arg = predictor.from_saved_model(a)

        pred_seq = np.array([])
        pred_lab = np.array([])
        pred_len = np.array([])
        pred_score = np.array([])
        arg_seq = np.array([])
        arg_lab = np.array([])
        arg_len = np.array([])
        arc = np.array([])
        arg_score = np.array([])
        batch_size=200
        for batch_data,batch_embd, batch_labels, batch_seq_len, batch_sequence_lengths,arguments,predicate,pred_mask,prop_embed in self.batch_1(
                        self.sequence[self.test_index.astype(int)],
                        self.dep_sequence[self.test_index.astype(int)],
                        self.dep_sequence[self.test_index.astype(int)],
                        self.num_nodes_by_test_sentences, batch_size,
                        self.sequence[self.test_index.astype(int)],
                        self.sequence[self.test_index.astype(int)],
                        self.predicate_mask[self.test_index.astype(int)],
                        self.predicate_mask[self.test_index.astype(int)]
                        ):
            predictions = predict_pred(
                {
               'pred_input': batch_data.astype(int),
               'pred_original_sequence_lengths': batch_sequence_lengths.astype(int),
               'pred_mask':pred_mask.astype(int),
               'pred_dep':batch_embd,
            })

            pred_seq = np.append(pred_seq,predictions["classes"]).reshape(-1,batch_seq_len)
            arc = np.append(arc,predictions["arc"]).reshape(-1,batch_seq_len)
            pred_lab = np.append(pred_lab,predicate).reshape(-1,batch_seq_len)
            pred_score = np.append(pred_score,predictions["probabilities"]).reshape(-1,1)
            pred_len = np.append(pred_len,predictions["sequence_lenghts"]).reshape(-1,1)

        inf_mask,prop_mask = copy.deepcopy(self.pred_tokenizer.sequences_to_texts(pred_seq)), copy.deepcopy(self.pred_tokenizer.sequences_to_texts(pred_seq))
        pred_list = np.array([])
        for i in range(len(inf_mask)):
            inf_mask[i] = inf_mask[i].replace('SOS','1')
            inf_mask[i] = inf_mask[i].replace('O','1')
            inf_mask[i] = inf_mask[i].replace('P_B','2')
            inf_mask[i] = inf_mask[i].replace('P_I','2')
            inf_mask[i] = inf_mask[i].replace('P_E','2')
            inf_mask[i] = inf_mask[i].replace('P_S','2')
            inf_mask[i] = np.array(list(map(int, inf_mask[i].split(" "))))
            prop_mask[i] = prop_mask[i].replace('SOS','1')
            prop_mask[i] = prop_mask[i].replace('O','1')
            prop_mask[i] = prop_mask[i].replace('P_B','2')
            prop_mask[i] = prop_mask[i].replace('P_I','2')
            prop_mask[i] = prop_mask[i].replace('P_E','2')
            prop_mask[i] = prop_mask[i].replace('P_S','2')
            prop_mask[i] = np.array(list(map(int, prop_mask[i].split(" "))))
            pred_list = np.append(pred_list,np.where(np.array(inf_mask[i])==2)[0].tolist())
            for l in range(len(inf_mask[i])):
                if(arc[i][l]==1 and inf_mask[i][l]!=2 and inf_mask[i][l]!=0):
                    inf_mask[i][l]=3

        prop_mask = self.predicate_embedding_propagate(prop_mask,np.max(self.num_nodes_by_sentences),np.array(self.edges_indexes_collection)[self.test_index.astype(int)],self.num_nodes_by_test_sentences)
        prop_mask = self.pad_data_with_tf(prop_mask,np.max(self.num_nodes_by_sentences))
        inf_mask = self.pad_data_with_tf(inf_mask,np.max(self.num_nodes_by_sentences))
        propagated_mask_ = inf_mask
        for batch_data,batch_embd, batch_labels, batch_seq_len, batch_sequence_lengths,arguments,p_mask,pred_mask,prop_embed in self.batch_1(
                        self.sequence[self.test_index.astype(int)],
                        self.dep_sequence[self.test_index.astype(int)],
                        self.dep_sequence[self.test_index.astype(int)],
                        self.num_nodes_by_test_sentences, batch_size,
                        self.sequence[self.test_index.astype(int)],
                        prop_mask,
                        self.predicate_mask[self.test_index.astype(int)],
                        propagated_mask_
                        ):

            predictions_arg = predict_arg(
                {
               'arg_input': batch_data.astype(int),
               'arg_original_sequence_lengths': batch_sequence_lengths.astype(int),
               'arg_mask':prop_embed,
               'arg_dep':batch_embd,
			   'prop_mask':p_mask,
            })

            arg_seq = np.append(arg_seq,predictions_arg["classes"]).reshape(-1,batch_seq_len)
            arg_lab = np.append(arg_lab,arguments).reshape(-1,batch_seq_len)
            arg_score = np.append(arg_score,predictions_arg["probabilities"]).reshape(-1,1)
            arg_len = np.append(arg_len,predictions_arg["sequence_lenghts"]).reshape(-1,1)

        predicates = self.pred_tokenizer.sequences_to_texts(pred_seq)
        args = self.arg_tokenizer.sequences_to_texts(arg_seq)
        seq = self.merge(predicates,args)
        relations = self.sequence_to_relation([" ".join(e) for e in seq[1922:3705]],arg_len[1922:3705],arg_score[1922:3705],self.test_index[1922:3705].astype(int))
        rels = self.generate(relations,copy.deepcopy(self.sentences[1922:3705]))
        r_enhance=list()  
        r_enhance = self.enhance(relations)
        tuples = self.extractions_fr(copy.deepcopy(r_enhance),self.sentences[1922:3705],arg_len[1922:3705],None,self.test_index[1922:3705].astype(int))
        with open("carb/french_predictions_"+str(arg)+".tsv","w",encoding="utf-8") as output_file:
            for t in tuples:
                output_file.writelines(t+"\n")
		
        relations = self.sequence_to_relation([" ".join(e) for e in seq[0:1922]],arg_len[0:1922],arg_score[0:1922],self.test_index[0:1922].astype(int))
        rels = self.generate(relations,copy.deepcopy(self.sentences[0:1922]))
        r_enhance=list()
        r_enhance = relations#self.enhance(relations)
        tuples = self.extractions(copy.deepcopy(r_enhance),self.sentences[0:1922],arg_len[0:1922],self.coref[0:1922],self.test_index[0:1922].astype(int))
        with open("carb/english_predictions_"+str(arg)+".tsv","w",encoding="utf-8") as output_file:
            for t in tuples:
                output_file.writelines(t+"\n")
		
        relations = self.sequence_to_relation([" ".join(e) for e in seq[3705:5203]],arg_len[3705:5203],arg_score[3705:5203],self.test_index[3705:5203].astype(int))
        rels = self.generate(relations,copy.deepcopy(self.sentences[3705:5203]))
        r_enhance=list()  
        r_enhance = self.enhance(relations)
        tuples = self.extractions_es_pt(copy.deepcopy(r_enhance),self.sentences[3705:5203],arg_len[3705:5203],None,self.test_index[3705:5203].astype(int))
        with open("carb/spanish_predictions_"+str(arg)+".tsv","w",encoding="iso-8859-1") as output_file:
            for t in tuples:
                output_file.writelines(t+"\n")
                
        relations = self.sequence_to_relation([" ".join(e) for e in seq[5203:6706]],arg_len[5203:6706],arg_score[5203:6706],self.test_index[5203:6706].astype(int))
        rels = self.generate(relations,copy.deepcopy(self.sentences[5203:6706]))
        r_enhance=list()  
        r_enhance = self.enhance(relations)
        tuples = self.extractions_es_pt(copy.deepcopy(r_enhance),self.sentences[5203:6706],arg_len[5203:6706],None,self.test_index[5203:6706].astype(int))
        with open("carb/portuguese_predictions_"+str(arg)+".tsv","w",encoding="iso-8859-1") as output_file:
            for t in tuples:
                output_file.writelines(t+"\n")

    def validation(self,Arg_path,Pred_path):
        #os.path.join("Pred128/pb",str(14))
        #os.path.join("Arg128/pb",str(14))
        subdirs = [x for x in Path(Pred_path).iterdir() if x.is_dir() and 'temp' not in str(x)]
        p = str(sorted(subdirs)[-1])
        predict_pred = predictor.from_saved_model(p)
        subdirs = [x for x in Path(Arg_path).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
        a = str(sorted(subdirs)[-1])
        predict_arg = predictor.from_saved_model(a)

        pred_seq = np.array([])
        pred_lab = np.array([])
        pred_len = np.array([])
        pred_score = np.array([])
        arg_seq = np.array([])
        arg_lab = np.array([])
        arc = np.array([])
        arg_len = np.array([])
        arg_score = np.array([])
        batch_size=200
        for batch_data,batch_embd, batch_labels, batch_seq_len, batch_sequence_lengths,arguments,predicate,pred_mask,prop_embed in self.batch_1(
                        self.sequence[self.val_index.astype(int)],
                        self.dep_sequence[self.val_index.astype(int)],
                        self.dep_sequence[self.val_index.astype(int)],
                        self.num_nodes_by_val_sentences, batch_size,
                        self.dep_sequence[self.val_index.astype(int)],
                        self.dep_sequence[self.val_index.astype(int)],
                        self.predicate_mask[self.val_index.astype(int)],
                        self.predicate_mask[self.val_index.astype(int)]
                        ):
            predictions = predict_pred(
                {
               'pred_input': batch_data.astype(int),
               'pred_original_sequence_lengths': batch_sequence_lengths.astype(int),
               'pred_mask':pred_mask.astype(int),
               'pred_dep':batch_embd,
            })
            pred_seq = np.append(pred_seq,predictions["classes"]).reshape(-1,batch_seq_len)
            arc = np.append(arc,predictions["arc"]).reshape(-1,batch_seq_len)
            pred_lab = np.append(pred_lab,predicate).reshape(-1,batch_seq_len)
            pred_score = np.append(pred_score,predictions["probabilities"]).reshape(-1,1)
            pred_len = np.append(pred_len,predictions["sequence_lenghts"]).reshape(-1,1)

        inf_mask,prop_mask = copy.deepcopy(self.pred_tokenizer.sequences_to_texts(pred_seq)), copy.deepcopy(self.pred_tokenizer.sequences_to_texts(pred_seq))
        pred_list = np.array([])
        for i in range(len(inf_mask)):
            inf_mask[i] = inf_mask[i].replace('SOS','1')
            inf_mask[i] = inf_mask[i].replace('O','1')
            inf_mask[i] = inf_mask[i].replace('P_B','2')
            inf_mask[i] = inf_mask[i].replace('P_I','2')
            inf_mask[i] = inf_mask[i].replace('P_E','2')
            inf_mask[i] = inf_mask[i].replace('P_S','2')
            inf_mask[i] = np.array(list(map(int, inf_mask[i].split(" "))))
            prop_mask[i] = prop_mask[i].replace('SOS','1')
            prop_mask[i] = prop_mask[i].replace('O','1')
            prop_mask[i] = prop_mask[i].replace('P_B','2')
            prop_mask[i] = prop_mask[i].replace('P_I','2')
            prop_mask[i] = prop_mask[i].replace('P_E','2')
            prop_mask[i] = prop_mask[i].replace('P_S','2')
            prop_mask[i] = np.array(list(map(int, prop_mask[i].split(" "))))
            pred_list = np.append(pred_list,np.where(np.array(inf_mask[i])==2)[0].tolist())
            for l in range(len(inf_mask[i])):
                if(arc[i][l]==1 and inf_mask[i][l]!=2 and inf_mask[i][l]!=0):
                    inf_mask[i][l]=3

        prop_mask = self.predicate_embedding_propagate(prop_mask,np.max(self.num_nodes_by_sentences),np.array(self.edges_indexes_collection)[self.val_index.astype(int)],self.num_nodes_by_val_sentences)
        prop_mask = self.pad_data_with_tf(prop_mask,np.max(self.num_nodes_by_sentences))
        inf_mask = self.pad_data_with_tf(inf_mask,np.max(self.num_nodes_by_sentences))
        propagated_mask_ = inf_mask

        for batch_data,batch_embd, batch_labels, batch_seq_len, batch_sequence_lengths,arguments,p_mask,pred_mask,prop_embed in self.batch_1(
                        self.sequence[self.val_index.astype(int)],
                        self.dep_sequence[self.val_index.astype(int)],
                        self.dep_sequence[self.val_index.astype(int)],
                        self.num_nodes_by_val_sentences, batch_size,
                        self.dep_sequence[self.val_index.astype(int)],
                        prop_mask,
                        self.predicate_mask[self.val_index.astype(int)],
                        propagated_mask_
                        ):

            predictions_arg = predict_arg(
                {
               'arg_input': batch_data.astype(int),
               'arg_original_sequence_lengths': batch_sequence_lengths.astype(int),
               'arg_mask':prop_embed.astype(int),
               'arg_dep':batch_embd,
			   'prop_mask':p_mask
            })

            arg_seq = np.append(arg_seq,predictions_arg["classes"]).reshape(-1,batch_seq_len)
            arg_lab = np.append(arg_lab,arguments).reshape(-1,batch_seq_len)
            arg_score = np.append(arg_score,predictions_arg["probabilities"]).reshape(-1,1)
            arg_len = np.append(arg_len,predictions_arg["sequence_lenghts"]).reshape(-1,1)

        predicates = self.pred_tokenizer.sequences_to_texts(pred_seq)
        args = self.arg_tokenizer.sequences_to_texts(arg_seq)
        seq = self.merge(predicates,args)
        relations = self.sequence_to_relation([" ".join(e) for e in seq],arg_len,arg_score,self.val_index.astype(int))
        rels = self.generate(relations,copy.deepcopy(self.sentences[6706:]))#1922
        r_enhance=list()
        r_enhance = self.enhance(copy.deepcopy(relations))
        tuples = self.extractions(copy.deepcopy(relations),self.sentences[6706:],arg_len,None,self.val_index.astype(int))
        with open("carb/validation.tsv","w",encoding="utf-8") as output_file:
            for t in tuples:
                output_file.writelines(t+"\n")
              
    def enhance(self,r):
        ignore ={"over","on","in","at"}
        for i in range(0,len(r)):
                    for j in range(0,len(r)):
                        if(i!=j and r[i]["context"]==r[j]["context"]):
                            if(len(r[i]["arg0"])==0  and len(r[i]["pred"])>0 and len(r[j]["arg0"])>0 
                               and len(r[i]["arg1"])>0
                              and r[i]["pred"][0].lower() not in ("in",",","at","on","by")):
                                if((" ".join(r[i]["pred"]).strip()+" "+" ".join(r[i]["arg1"]).strip()
                                    +" "+" ".join(r[i]["arg2"]).strip()).strip()  
                                   in (" ".join(r[j]["arg2"]).strip()).strip()):
                                    if((" ".join(r[j]["arg2"]).strip()).strip().startswith("on")==True
                                      or (" ".join(r[j]["arg2"]).strip()).strip().startswith("under")==True
                                       or (" ".join(r[j]["arg2"]).strip()).strip().startswith("alongside")==True
                                     or len((" ".join(r[j]["arg2"]).strip()).strip().split(" "))<3):
                                        r[i]["arg0"] = r[j]["arg1"]
                                    elif(" or " not in (" ".join(r[j]["arg2"]).strip()).strip()
                                        and " and "  not in (" ".join(r[j]["arg2"]).strip()).strip()):
                                        r[i]["arg0"] = r[j]["arg0"]
                                    else:
                                        r[i]["arg0"] = r[j]["arg1"]
                                elif((" ".join(r[i]["pred"]).strip()+" "+" ".join(r[i]["arg1"]).strip()
                                    +" "+" ".join(r[i]["arg2"]).strip()).strip() 
                                    in (" ".join(r[j]["arg1"]).strip()).strip() and (len(r[j]["arg1"])>1)):
                                    r[i]["arg0"] = r[j]["arg1"]
                                elif((" ".join(r[i]["pred"]).strip()+" "+" ".join(r[i]["arg1"]).strip()
                                    +" "+" ".join(r[i]["arg2"]).strip()).strip() 
                                    in (" ".join(r[j]["arg1"]).strip()).strip()
                                   and r[i]["pred"][0].strip().split(" ")[-1] in ignore):
                                    r[i]["arg0"] = r[j]["arg0"]

                            elif(len(r[i]["arg0"])>0 and len(r[j]["pred"])>0 
                                 and len(r[j]["arg0"])==0 and len(r[j]["arg1"])>0
                                 and r[j]["pred"][0].lower() not in ("in",",","at","on","by")):
                                if((" ".join(r[j]["pred"]).strip()+" "+" ".join(r[j]["arg1"]).strip()
                                    +" "+" ".join(r[j]["arg2"]).strip()) in (" ".join(r[i]["arg2"]).strip()) ):
                                    if((" ".join(r[i]["arg2"]).strip()).strip().startswith("on")==True
                                      or (" ".join(r[i]["arg2"]).strip()).strip().startswith("under")==True
                                       or (" ".join(r[i]["arg2"]).strip()).strip().startswith("alongside")==True
                                      or len((" ".join(r[i]["arg2"]).strip()).strip().split(" "))<3):
                                        r[j]["arg0"] = r[i]["arg1"]
                                    elif(" or " not in (" ".join(r[i]["arg2"]).strip()).strip()
                                        and " and "  not in (" ".join(r[i]["arg2"]).strip()).strip()):
                                        r[j]["arg0"] = r[i]["arg0"]
                                    else:
                                        r[j]["arg0"] = r[i]["arg1"]
                                elif((" ".join(r[j]["pred"]).strip()+" "+" ".join(r[j]["arg1"]).strip()
                                        +" "+" ".join(r[j]["arg2"]).strip()) 
                                     in (" ".join(r[i]["arg1"]).strip()) and (len(r[i]["arg1"])>1)):
                                    r[j]["arg0"] = r[i]["arg1"]
        return r

    def extractions(self,relations,sentences,seq_len,coref,index):
        def intersection(lst1, lst2): 
            lst3 = [value for value in lst1 if value in lst2] 
            return lst3
        tuples = list()
        prep = ("at","in","on","of","over","to","from","until","by","as","with")
        anaph = ("we","he","she","they","them","him","his","her","it","its","their","themselves","itself","me","you","i",
                 "this","himself","herself","We","He","She","They","Them","Him","His","Her","It","Its","Their","Themselves",
                 "Itself","Me","You","I","This","Himself","Herself")
        counter = -1
        anaphora,coref = (False,seq_len) if coref is None else (True,coref)
        for r,current_sentence,l,ref in zip(relations,sentences,seq_len,coref):
            preds = []
            counter+=1
            subj = r["arg0"]
            obj = r["arg1"]
            if(len(r["arg0"])==1): 
                if(" such as " in r["arg0"][0]):
                    subj = r["arg0"][0].split(" such as ")
                if(" and " in r["arg0"][0] and len(r["arg0"][0].strip().split(" "))<4):
                    subj = r["arg0"][0].split(" and ")
            if(len(r["arg0"])>1):
                subj = []
                for s in r["arg0"]:
                    if(" and " in s 
                       and ((len(s.strip().split(" "))<5) or (len(s.strip().split(" "))<6 and s.startswith("a ")))):
                        conjuncts = s.split(" and ")
                        subj.append(conjuncts[0])
                        subj.append(conjuncts[1])
                    else:
                        subj.append(s)
            if(len(r["arg1"])>1):
                obj = []
                for o in r["arg1"]:
                    if(" and " in o 
                       and ((len(o.strip().split(" "))<5) or (len(o.strip().split(" "))<6 and o.startswith("a ")))):
                        conjuncts = o.split(" and ")
                        obj.append(conjuncts[0])
                        obj.append(conjuncts[1])
                    else:
                        obj.append(o)               

            if(len(r["pred"])>0):
                preds = [" ".join(r["pred"]).replace(",","is").strip()]
                if(" and " in preds):
                    pred = preds[0].split(" and ")
                    preds[0] = preds[0]+" "+(" ".join(preds[1].split(" ")[1:]))
                    preds[1] = preds[0].split(" ")[0]+" "+preds[1]
            for p_i in range(len(preds)):
                pred = preds[p_i]
                #if(len(r["arg0"])==0 and len(r["arg2"])>0):
                    #print(pred)
                    #if(pred.strip().split(" ")[-1] in ("on")):
                    #    print(r)
                    #    subj = obj
                    #    obj = r["arg2"]
                    #    pred = " ".join(pred.strip().split(" ")[:-1])
                binary=False
                        
                pos = self.get_pos(
                            np.array(self.words_type_collection)[index][counter],
                            np.array(self.nodes_type_collection)[index][counter],
                            pred)
                if(pred.strip  in (",",":","(",";","--"'"')):
                    pred = "is" 
                if(pred.startswith("'s")):
                    pred = "has "+pred.strip().split(pred[0:2].strip())[1].strip()
                if(pred.startswith("'d")):
                    pred = "had "+pred.strip().split(pred[0:2].strip())[1].strip()

                if(len(obj)>0):
                    if(all([p not in("AUX","VERB") for p in pos]) and len(pos)>0):
                        if(pred.strip().split(" ")[0] not in ("has","had")):
                            pred= "is "+pred
                else:       
                    if(all([p in("AUX","PART","VERB") for p in pos]) and len(subj)>1):
                        if(subj[1].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                        elif(subj[0].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                        elif(pred+" "+subj[0].strip() 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],pred,[subj[0]]
                        else:
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                    elif(all([p not in("AUX","VERB") for p in pos]) and len(subj)>1):
                        if(subj[1].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],"is "+pred,[subj[1]]
                        elif(subj[0].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],"is "+pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],"is "+pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],"is "+pred,[subj[1]]

                    elif(any([p in("PROPN") for p in pos]) and len(pos)>0):
                        obj = [str(pred)]
                        pred= "is"
                    elif(all([p in("ADJ") for p in pos]) and len(pos)>0):
                        pred= "is "+pred
                        binary=True
                    elif (all([p not in("AUX","PART","VERB") for p in pos]) 
                          and pred.strip().split(" ")[-1] not in prep
                         and len(subj)>0
                         and subj[0].strip() not in anaph
                         and (len(subj[0].strip().split(" "))>1 or len(pred.strip().split(" "))>1)):
                        obj = [str(pred)]
                        pred= "is"

                if(pred.lower()=="'s" or pred.startswith("'s' ")):
                    pred="has"
                elif(pred.lower()=="a" or pred.startswith("a ")):
                    pred="is a"

                if(len(obj)==0 and len(r["arg2"])>0):
                    obj=r["arg2"]
                    r["arg2"] = r["arg3"]

                for i in range(0,len(subj)):
                        if(ref!="None" and anaphora==True and ref.strip().split(";")[0].lower()!=ref.strip().split(";")[1].lower()):
                            if(subj[i].startswith(anaph) 
                                and (len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[0].strip().split()))
                                 or len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                               and any([e  not in anaph for e in ref.strip().split(";")])):
                                ref_ = ref.split(";")
                                ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                core_,rep = (ref_[0],ref_[1]) if subj[i].startswith(ref_[1]) else (ref_[1],ref_[0])
                                if(all([core_ not in  e for e in obj])):
                                        core_ = core_.split(",")[0] if "," in core_ else core_
                                        subj[i] =  subj[i].replace(rep,core_,1)

                        if(len(obj)>0):
                            for j in range(0,len(obj)):
                                if(ref!="None" and anaphora==True):
                                    if(obj[j].startswith(anaph)
                                     and (len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split()))
                                         or len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                                       and any([e  not in anaph for e in ref.strip().split(";")])
                                      ):
                                        ref_ = ref.split(";")
                                        ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                        ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                        core_,rep = (ref_[0],ref_[1]) if obj[j].startswith(ref_[1]) else (ref_[1],ref_[0])
                                        if(core_ not in subj[i]):
                                            core_ = core_.split(",")[0] if "," in core_ else core_
                                            obj[j] =  obj[j].replace(rep,core_, 1)

                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"+obj[j]+"\t"
                                if(len(r["arg2"])>0):
                                    triple+=("\t".join(r["arg2"]))
                                if(len(r["loc"])>0):
                                    triple+="\t"+("".join(r["loc"]))
                                if(len(r["temp"])>0):
                                    triple+="\t"+("".join(r["temp"]))

                                tuples.append(triple.strip())
                        elif(binary==True):
                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"
                                tuples.append(triple.strip())

        return tuples
	
    def extractions_fr(self,relations,sentences,seq_len,coref,index):
        def intersection(lst1, lst2): 
            lst3 = [value for value in lst1 if value in lst2] 
            return lst3
        tuples = list()
        prep = ("at","in","on","of","over","to","from","until","by","as","with")
        anaph = ("we","he","she","they","them","him","his","her","it","its","their","themselves","itself","me","you","i",
                 "this","himself","herself","We","He","She","They","Them","Him","His","Her","It","Its","Their","Themselves",
                 "Itself","Me","You","I","This","Himself","Herself")
        counter = -1
        anaphora,coref = (False,seq_len) if coref is None else (True,coref)
        for r,current_sentence,l,ref in zip(relations,sentences,seq_len,coref):
            preds = []
            counter+=1
            subj = r["arg0"]
            obj = r["arg1"]
            if(len(r["arg0"])==1): 
                if(" comme " in r["arg0"][0]):
                    subj = r["arg0"][0].split(" comme ")
                if(" et " in r["arg0"][0] and len(r["arg0"][0].strip().split(" "))<4):
                    subj = r["arg0"][0].split(" et ")
            if(len(r["arg0"])>1):
                subj = []
                for s in r["arg0"]:
                    if(" et " in s 
                       and ((len(s.strip().split(" "))<5) or (len(s.strip().split(" "))<6))):
                        conjuncts = s.split(" et ")
                        subj.append(conjuncts[0])
                        subj.append(conjuncts[1])
                    else:
                        subj.append(s)
            if(len(r["arg1"])>1):
                obj = []
                for o in r["arg1"]:
                    if(" et " in o 
                       and ((len(o.strip().split(" "))<5) or (len(o.strip().split(" "))<6 ))):
                        conjuncts = o.split(" et ")
                        obj.append(conjuncts[0])
                        obj.append(conjuncts[1])
                    else:
                        obj.append(o)               

            if(len(r["pred"])>0):
                preds = [" ".join(r["pred"]).strip()]#[" ".join(r["pred"]).replace(",","est").strip()]
                if(" et " in preds):
                    pred = preds[0].split(" et ")
                    preds[0] = preds[0]+" "+(" ".join(preds[1].split(" ")[1:]))
                    preds[1] = preds[0].split(" ")[0]+" "+preds[1]
            for p_i in range(len(preds)):
                pred = preds[p_i]
                #if(len(r["arg0"])==0 and len(r["arg2"])>0):
                    #print(pred)
                    #if(pred.strip().split(" ")[-1] in ("on")):
                    #    print(r)
                    #    subj = obj
                    #    obj = r["arg2"]
                    #    pred = " ".join(pred.strip().split(" ")[:-1])
                binary=False
                #if(pred.strip  in (",",":","(",";","--"'"')):
                #    pred = "is"         
                pos = self.get_pos(
                            np.array(self.words_type_collection)[index][counter],
                            np.array(self.nodes_type_collection)[index][counter],
                            pred)
                #if(pred.startswith("'s")):
                #    pred = "has "+pred.strip().split(pred[0:2].strip())[1].strip()
                #if(pred.startswith("'d")):
                #    pred = "had "+pred.strip().split(pred[0:2].strip())[1].strip()

                #if(len(obj)>0):
                #    if(all([p not in("AUX","VERB") for p in pos]) and len(pos)>0):
                #        if(pred.strip().split(" ")[0] not in ("has","had")):
                #            pred= "is "+pred
                #if(len(obj)==0):#else:       
                #    if(all([p in("AUX","PART","VERB") for p in pos]) and len(subj)>1):
                #        if(subj[1].strip()+" "+pred 
                #           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                #            subj,pred,obj = [subj[0]],pred,[subj[1]]
                #        elif(subj[0].strip()+" "+pred 
                #           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                #            subj,pred,obj = [subj[1]],pred,[subj[0]]
                #        elif(pred+" "+subj[1].strip()
                #           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                #            subj,pred,obj = [subj[0]],pred,[subj[1]]
                #        elif(pred+" "+subj[0].strip() 
                #           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                #            subj,pred,obj = [subj[1]],pred,[subj[0]]
                #        else:
                #            subj,pred,obj = [subj[0]],pred,[subj[1]]
                    #elif(all([p not in("AUX","VERB") for p in pos]) and len(subj)>1):
                    #    if(subj[1].strip()+" "+pred 
                    #       in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                    #        subj,pred,obj = [subj[0]],"is "+pred,[subj[1]]
                    #    elif(subj[0].strip()+" "+pred 
                    #       in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                    #        subj,pred,obj = [subj[1]],"is "+pred,[subj[0]]
                    #    elif(pred+" "+subj[1].strip()
                    #       in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                    #        subj,pred,obj = [subj[1]],"is "+pred,[subj[0]]
                    #    elif(pred+" "+subj[1].strip()
                    #       in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                    #        subj,pred,obj = [subj[0]],"is "+pred,[subj[1]]

                    #elif(any([p in("PROPN") for p in pos]) and len(pos)>0):
                    #    obj = [str(pred)]
                    #    pred= "is"
                    #elif(all([p in("ADJ") for p in pos]) and len(pos)>0):
                    #    pred= "is "+pred
                    #    binary=True
                    #elif (all([p not in("AUX","PART","VERB") for p in pos]) 
                    #      and pred.strip().split(" ")[-1] not in prep
                    #     and len(subj)>0
                    #     and subj[0].strip() not in anaph
                    #     and (len(subj[0].strip().split(" "))>1 or len(pred.strip().split(" "))>1)):
                    #    obj = [str(pred)]
                    #    pred= "is"

                #if(pred.lower()=="'s" or pred.startswith("'s' ")):
                #    pred="has"
                #elif(pred.lower()=="a" or pred.startswith("a ")):
                #    pred="is a"

                if(len(obj)==0 and len(r["arg2"])>0):
                    obj=r["arg2"]
                    r["arg2"] = r["arg3"]

                for i in range(0,len(subj)):
                        if(ref!="None" and anaphora==True and ref.strip().split(";")[0].lower()!=ref.strip().split(";")[1].lower()):
                            if(subj[i].startswith(anaph) 
                                and (len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[0].strip().split()))
                                 or len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                               and any([e  not in anaph for e in ref.strip().split(";")])):
                                ref_ = ref.split(";")
                                ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                core_,rep = (ref_[0],ref_[1]) if subj[i].startswith(ref_[1]) else (ref_[1],ref_[0])
                                if(all([core_ not in  e for e in obj])):
                                        core_ = core_.split(",")[0] if "," in core_ else core_
                                        subj[i] =  subj[i].replace(rep,core_,1)

                        if(len(obj)>0):
                            for j in range(0,len(obj)):
                                if(ref!="None" and anaphora==True):
                                    if(obj[j].startswith(anaph)
                                     and (len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split()))
                                         or len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                                       and any([e  not in anaph for e in ref.strip().split(";")])
                                      ):
                                        ref_ = ref.split(";")
                                        ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                        ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                        core_,rep = (ref_[0],ref_[1]) if obj[j].startswith(ref_[1]) else (ref_[1],ref_[0])
                                        if(core_ not in subj[i]):
                                            core_ = core_.split(",")[0] if "," in core_ else core_
                                            obj[j] =  obj[j].replace(rep,core_, 1)

                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"+obj[j]+"\t"
                                if(len(r["arg2"])>0):
                                    triple+=("\t".join(r["arg2"]))
                                if(len(r["loc"])>0):
                                    triple+="\t"+("".join(r["loc"]))
                                if(len(r["temp"])>0):
                                    triple+="\t"+("".join(r["temp"]))

                                tuples.append(triple.strip())
                        elif(binary==True):
                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"
                                tuples.append(triple.strip())

        return tuples
		
    def extractions_es_pt(self,relations,sentences,seq_len,coref,index):
        def intersection(lst1, lst2): 
            lst3 = [value for value in lst1 if value in lst2] 
            return lst3
        tuples = list()
        prep = ("at","in","on","of","over","to","from","until","by","as","with")
        anaph = ("we","he","she","they","them","him","his","her","it","its","their","themselves","itself","me","you","i",
                 "this","himself","herself","We","He","She","They","Them","Him","His","Her","It","Its","Their","Themselves",
                 "Itself","Me","You","I","This","Himself","Herself")
        counter = -1
        anaphora,coref = (False,seq_len) if coref is None else (True,coref)
        for r,current_sentence,l,ref in zip(relations,sentences,seq_len,coref):
            preds = []
            counter+=1
            subj = r["arg0"]
            obj = r["arg1"]
            #if(len(r["arg0"])==1): 
            #    if(" such as " in r["arg0"][0]):
            #        subj = r["arg0"][0].split(" tel que ")
            #    if(" et " in r["arg0"][0] and len(r["arg0"][0].strip().split(" "))<4):
            #        subj = r["arg0"][0].split(" et ")
            #if(len(r["arg0"])>1):
            #    subj = []
            #    for s in r["arg0"]:
            #        if(" et " in s 
            #           and ((len(s.strip().split(" "))<5) or (len(s.strip().split(" "))<6))):
            #            conjuncts = s.split(" et ")
            #            subj.append(conjuncts[0])
            #            subj.append(conjuncts[1])
            #        else:
            #            subj.append(s)
            #if(len(r["arg1"])>1):
            #    obj = []
            #    for o in r["arg1"]:
            #        if(" and " in o 
            #           and ((len(o.strip().split(" "))<5) or (len(o.strip().split(" "))<6 ))):
            #            conjuncts = o.split(" et ")
            #            obj.append(conjuncts[0])
            #            obj.append(conjuncts[1])
            #        else:
            #            obj.append(o)               

            if(len(r["pred"])>0):
                preds = [" ".join(r["pred"]).strip()]#[" ".join(r["pred"]).replace(",","est").strip()]
                #if(" et " in preds):
                #    pred = preds[0].split(" et ")
                #    preds[0] = preds[0]+" "+(" ".join(preds[1].split(" ")[1:]))
                #    preds[1] = preds[0].split(" ")[0]+" "+preds[1]
            for p_i in range(len(preds)):
                pred = preds[p_i]
                #if(len(r["arg0"])==0 and len(r["arg2"])>0):
                    #print(pred)
                    #if(pred.strip().split(" ")[-1] in ("on")):
                    #    print(r)
                    #    subj = obj
                    #    obj = r["arg2"]
                    #    pred = " ".join(pred.strip().split(" ")[:-1])
                binary=False
        
                pos = self.get_pos(
                            np.array(self.words_type_collection)[index][counter],
                            np.array(self.nodes_type_collection)[index][counter],
                            pred)
                if(pred.strip  in (",",":","(",";","--"'"')):
                    pred = "<Ser>" 
                #if(pred.startswith("'s")):
                #    pred = "has "+pred.strip().split(pred[0:2].strip())[1].strip()
                #if(pred.startswith("'d")):
                #    pred = "had "+pred.strip().split(pred[0:2].strip())[1].strip()

                if(len(obj)>0):
                    if(all([p not in("AUX","VERB") for p in pos]) and len(pos)>0):
                        if(pred.strip().split(" ")[0] not in ("has","had")):
                            pred= "<Ser> "+pred
                if(len(obj)==0):#else:       
                    if(all([p in("AUX","PART","VERB") for p in pos]) and len(subj)>1):
                        if(subj[1].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                        elif(subj[0].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                        elif(pred+" "+subj[0].strip() 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],pred,[subj[0]]
                        else:
                            subj,pred,obj = [subj[0]],pred,[subj[1]]
                    elif(all([p not in("AUX","VERB") for p in pos]) and len(subj)>1):
                        if(subj[1].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],"<Ser> "+pred,[subj[1]]
                        elif(subj[0].strip()+" "+pred 
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],"<Ser> "+pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[1]],"<Ser> "+pred,[subj[0]]
                        elif(pred+" "+subj[1].strip()
                           in " ".join( np.array(self.words_type_collection)[self.test_index.astype(int)][counter])):
                            subj,pred,obj = [subj[0]],"<Ser> "+pred,[subj[1]]

                    elif(any([p in("PROPN") for p in pos]) and len(pos)>0):
                        obj = [str(pred)]
                        pred= "<Ser>"
                    elif(all([p in("ADJ") for p in pos]) and len(pos)>0):
                        pred= "<Ser> "+pred
                        binary=True
                    elif (all([p not in("AUX","PART","VERB") for p in pos]) 
                          and pred.strip().split(" ")[-1] not in prep
                         and len(subj)>0
                         and subj[0].strip() not in anaph
                         and (len(subj[0].strip().split(" "))>1 or len(pred.strip().split(" "))>1)):
                        obj = [str(pred)]
                        pred= "<Ser>"

                #if(pred.lower()=="'s" or pred.startswith("'s' ")):
                #    pred="has"
                #elif(pred.lower()=="a" or pred.startswith("a ")):
                #    pred="is a"

                if(len(obj)==0 and len(r["arg2"])>0):
                    obj=r["arg2"]
                    r["arg2"] = r["arg3"]

                for i in range(0,len(subj)):
                        if(ref!="None" and anaphora==True and ref.strip().split(";")[0].lower()!=ref.strip().split(";")[1].lower()):
                            if(subj[i].startswith(anaph) 
                                and (len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[0].strip().split()))
                                 or len(intersection(subj[i].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                               and any([e  not in anaph for e in ref.strip().split(";")])):
                                ref_ = ref.split(";")
                                ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                core_,rep = (ref_[0],ref_[1]) if subj[i].startswith(ref_[1]) else (ref_[1],ref_[0])
                                if(all([core_ not in  e for e in obj])):
                                        core_ = core_.split(",")[0] if "," in core_ else core_
                                        subj[i] =  subj[i].replace(rep,core_,1)

                        if(len(obj)>0):
                            for j in range(0,len(obj)):
                                if(ref!="None" and anaphora==True):
                                    if(obj[j].startswith(anaph)
                                     and (len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split()))
                                         or len(intersection(obj[j].lower().strip().split(" ") , 
                                                     ref.strip().split(";")[1].strip().split())))
                                       and any([e  not in anaph for e in ref.strip().split(";")])
                                      ):
                                        ref_ = ref.split(";")
                                        ref_[0] =  ref_[0].split()[0] if "," in ref_[0] else ref_[0]
                                        ref_[1] =  ref_[1].split()[0] if "," in ref_[1] else ref_[1]
                                        core_,rep = (ref_[0],ref_[1]) if obj[j].startswith(ref_[1]) else (ref_[1],ref_[0])
                                        if(core_ not in subj[i]):
                                            core_ = core_.split(",")[0] if "," in core_ else core_
                                            obj[j] =  obj[j].replace(rep,core_, 1)

                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"+obj[j]+"\t"
                                if(len(r["arg2"])>0):
                                    triple+=("\t".join(r["arg2"]))
                                if(len(r["loc"])>0):
                                    triple+="\t"+("".join(r["loc"]))
                                if(len(r["temp"])>0):
                                    triple+="\t"+("".join(r["temp"]))

                                tuples.append(triple.strip())
                        elif(binary==True):
                                score = r["score"][0]
                                triple = r["context"].replace(",,",";")+"\t"+str(score)+"\t"
                                triple+=pred+"\t"+subj[i]+"\t"
                                tuples.append(triple.strip())

        return tuples