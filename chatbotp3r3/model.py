""" 
Bot Model using seq2seq neural net 
ICS4U Winter 2017

Uses the seq2seq model that was originally
invented for language translation
The chat model "translates" from english
to english.

Works with python2.7 and python3.5

Based on:
Sequence to sequence model by Cho et al.(2014)
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

"""

import time
import tensorflow as tf

import config
import os

class ChatBotModel(object):
    
    def __init__(self, forwardOnly, batchSize):
        """
        parameters: @forwardOnly: if true - do no construct backpropagation, else do
                    @batchSize - the list of batches
        """
        print('Initialize new model')
        self.forwardNetworkOnly = forwardOnly
        self.batchSize = batchSize
        setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
        setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)
        try:
            self.vocabSize = sum(1 for _ in open(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE)))
        except OSError:
            print("error: Vocabulary not found!")
    
    def _createPlaceholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create TF placeholders for enc and dec objects')
        self.encoderInputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoderInputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoderMasks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = self.decoderInputs[1:]
        
    def _inference(self):
        print('Create a sampled softmax function')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < self.vocabSize:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, self.vocabSize])
            b = tf.get_variable('proj_b', [self.vocabSize])
            self.outputProjection = (w, b)

        def sampledLoss(labels=None, logits=None):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, labels, logits, 
                                              config.NUM_SAMPLES, self.vocabSize)
        self.softmaxLossFunction = sampledLoss

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS)

    def _createLoss(self):
        print('Creating a loss function...')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=self.vocabSize,
                    num_decoder_symbols=self.vocabSize,
                    embedding_size=config.HIDDEN_SIZE,
                    output_projection=self.outputProjection,
                    feed_previous=do_decode)

        if self.forwardNetworkOnly:
            print('fw is true')
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoderInputs, 
                                        self.decoderInputs, 
                                        self.targets,
                                        self.decoderMasks, 
                                        config.BUCKETS, 
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmaxLossFunction)
            # If we use output projection, we need to project outputs for decoding.
            if self.outputProjection:
                for bucket in range(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.outputProjection[0]) + self.outputProjection[1]
                                            for output in self.outputs[bucket]]
        else:
            print('fw is false')
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoderInputs, 
                                        self.decoderInputs, 
                                        self.targets,
                                        self.decoderMasks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmaxLossFunction)
        print('Time: {:3.3f} seconds'.format(time.time() - start))

    def _createOptimizer(self):
        print('Creating optimizer function (one per bucket)...')
        with tf.variable_scope('training') as scope:
            self.globalStep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.forwardNetworkOnly:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradientNorms = []
                self.trainOps = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):
                    
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradientNorms.append(norm)
                    self.trainOps.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.globalStep))
                    print('Created optimized function for bucket {} in {:3.3f} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _createSummary(self):
        pass

    def buildGraph(self):
        self._createPlaceholders()
        self._inference()
        self._createLoss()
        self._createOptimizer()
        self._createSummary()
