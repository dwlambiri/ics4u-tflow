""" 
A neural chatbot using seq2seq model with
attentional decoder. 

This file runs the chatbot.
It has 2 modes: trainTheBot and chatWithBot

It runs in python 2.7 and python 3.5

Based on:
Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel
import config
import data

def _getRandomBucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def _assertLengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    """ Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths """
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_masks), decoder_size))

def runStep(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chatWithBot mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assertLengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def _getBuckets():
    """ 
    Load the dataset into buckets based on their lengths.
    train_buckets_scale is the inverval that'll help us 
    choose a random bucket later on.
    """
    test_buckets = data.loadData(config.TESTFILE+config.IDS+config.ENCODER, config.TESTFILE+config.IDS+config.DECODER)
    data_buckets = data.loadData(config.TRAINFILE+config.IDS+config.ENCODER, config.TRAINFILE+config.IDS+config.DECODER)
    train_bucket_sizes = [len(data_buckets[b]) for b in range(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale

def _getSkipStep(iteration):
    """ How many steps should the model trainTheBot before it saves all the weights. """
    if iteration < config.CHECKPOINTSTEP:
        return config.CHECKPOINTSMALL
    return config.CHECKPOINTSTEP

def _checkRestoreParameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/' + config.CHECKPT_FILE))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading checkpointed parameters for Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("State reloaded. Restarting computation....")
    else:
        print("Starting without a checkpoint. Fresh Chatbot")

def _evalTestSet(sess, model, test_buckets):
    """ Evaluate on the test set. """
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print("  Test: empty bucket %d" % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = data.getBatch(test_buckets[bucket_id], 
                                                                        bucket_id,
                                                                        batchSize=config.BATCH_SIZE)
        _, step_loss, _ = runStep(sess, model, encoder_inputs, decoder_inputs, 
                                   decoder_masks, bucket_id, True)
        print('Test bucket[{}] =( loss {:3.3f}: time {:3.3f} s )'.format(bucket_id, step_loss, time.time() - start))

def trainTheBot():
    """ Train the bot """
    print("This is a training session....")
    test_buckets, data_buckets, train_buckets_scale = _getBuckets()
    # in trainTheBot mode, we need to create the backward path, so forwrad_only is False
    model = ChatBotModel(False, config.BATCH_SIZE)
    model.buildGraph()

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True)) as sess:
        print('Start training session')
        sess.run(tf.global_variables_initializer())
        _checkRestoreParameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        start = time.time()
        while True:
            try:
                skip_step = _getSkipStep(iteration)
                bucket_id = _getRandomBucket(train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = data.getBatch(data_buckets[bucket_id], 
                                                                               bucket_id,
                                                                               batchSize=config.BATCH_SIZE)               
                _, step_loss, _ = runStep(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
                total_loss += step_loss
                iteration += 1
    
                if iteration % skip_step == 0:
                    print('Iter [{}] = ( Average Step Loss {:3.3f}: Average Step Time {:3.3f} s )'.format(iteration, total_loss/skip_step, (time.time() - start)/skip_step))
                    start = time.time()
                    total_loss = 0
                    saver.save(sess, os.path.join(config.CPT_PATH, 'chatbot'), global_step=model.global_step)
                    if iteration % (config.CHECKTESTMULT * skip_step) == 0:
                        # Run evals on development set and print their loss
                        _evalTestSet(sess, model, test_buckets)
                        start = time.time()
                    sys.stdout.flush()
            except KeyboardInterrupt:
                print("Keyboard interrupt received... Exiting")
                return

def _getUserInput():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

def _findRightBucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])
    
def _isQuestion(word):
    question = {'how': True,'where': True,'who': True, 'why': True, 'what': True, 'are': True, 'do' : True}
    try:
        return question[word]
    except KeyError:
        return False

def _constructResponse(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB
    
    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    #print(output_logits[0])

    outputs = []
    factor = 1.5
    for logit in output_logits:
        nonstop = np.argmax(logit[0][config.END_ID+1:])+config.END_ID+1
        #if iter == 0:
        #   outputs.append(int(nonstop))
        #else:
        if logit[0][config.END_ID] > factor* logit[0][nonstop]:
            outputs.append(config.END_ID)
        else:
            print("EOS={} Token={}".format(logit[0][config.END_ID], logit[0][nonstop]))
            outputs.append(int(nonstop))
        factor = factor**0.4

    #outputs = [int(np.argmax(logit[0][config.END_ID:])+config.END_ID) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    print("Outputs{}".format(outputs))
    for output in outputs:
        print("OToken={}".format(inv_dec_vocab[output]))
    if config.END_ID in outputs:
        outputs = outputs[:outputs.index(config.END_ID)]
    # Print out sentence corresponding to outputs.
    try:
        first = outputs[0]
        if _isQuestion(inv_dec_vocab[first]):
            eol = " ?"
        else:
            eol = " ."
    except IndexError:
        eol = " ???"
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs]) + eol

def chatWithBot():
    """ 
    in test mode, we don't to create the backward path
    """
    _, enc_vocab = data.loadVocabulary(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE))
    inv_dec_vocab, _ = data.loadVocabulary(os.path.join(config.PROCESSED_PATH, config.VOCAB_FILE))

    model = ChatBotModel(True, batch_size=1)
    model.buildGraph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _checkRestoreParameters(sess, saver)
        output_file = open(os.path.join(config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard input.
        max_length = config.BUCKETS[-1][0]
        print('Welcome to TensorChat! Press enter on an empty line to exit. Max length is', max_length)
        while True:
            line = _getUserInput()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = data.sentence2ID(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                line = _getUserInput()
                continue
            # Which bucket does it belong to?
            bucket_id = _findRightBucket(len(token_ids))
            print("BucketID {} Token Ids {}".format(bucket_id, token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = data.getBatch([(token_ids, [])], 
                                                                            bucket_id,
                                                                            batchSize=1)
            # Get output logits for the sentence.
            _, _, output_logits = runStep(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _constructResponse(output_logits, inv_dec_vocab)
            print('\n'+'BOT ++++ ' + response + '\n')
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the trainTheBot mode")
    args = parser.parse_args()

    if not os.path.isdir(config.PROCESSED_PATH):
        data.prepareRawData()
        data.processData()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    data.makeOutputDirectory(config.CPT_PATH)

    if args.mode == 'train':
        trainTheBot()
    elif args.mode == 'chat':
        chatWithBot()

if __name__ == '__main__':
    main()
