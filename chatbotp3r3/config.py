""" 
Neural chatbot configuration
ICU4U Winter 2017 project

Based on:
Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""

# parameters for processing the dataset
DATA_PATH = 'cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
VOCAB_FILE  = 'vocab.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'
CHECKPT_FILE = 'checkpoint'
TRAINFILE = 'train'
TESTFILE  = 'test'
ENCODER   = '.enc'
DECODER   = '.dec'
IDS       = '_ids'


# vocabulary parameters
PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3

SPECIAL_SEQ = ['<pad>', '<unk>', '<s>', '<\s>']
SPECIAL_ID = [PAD_ID, UNK_ID, START_ID, END_ID]

"""
if a word shows up in the data less than THRESHOLD
it is *not* written to the dictionary
all said words will point to <unk> in the text
this limits the dictionary size and makes
the training faster
"""
THRESHOLD = 3

"""
only dialog pairs that are withing QADIFF_THRESHOLD
length of each other are used. This eliminates
monologues, which tend to appear in movies
"""
QADIFF_THRESHOLD = 3

#test set parameters
TESTSET_SIZE = 25000

#data bucket
DATABUCKET=10000

# seq2seq model configuration
BUCKETS = [(5, 7), (10, 12) , (16, 18)]
NUM_LAYERS = 2
HIDDEN_SIZE = 128
BATCH_SIZE = 64
LR = 0.5
MAX_GRAD_NORM = 5.0
NUM_SAMPLES = 512

#chatbot parameters
CHECKPOINTSTEP=500
CHECKPOINTSMALL=100
CHECKTESTMULT=10
