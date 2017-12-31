""" 
Neural chatbot configuration
ICU4U Winter 2017 project

Based on:
Sequence to sequence model by Cho et al.(2014)
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
CMDFILENAME = 'test.txt'


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
THRESHOLD = 1

"""
only dialog pairs that are withing QADIFF_THRESHOLD
length of each other are used. This eliminates
monologues, which tend to appear in movies
"""
QADIFF_THRESHOLD = 5

#test set parameters
TESTSET_SIZE = 25000

#data bucket
DATABUCKET=10000

# seq2seq model configuration
BUCKETS = [(5, 10), (10, 16) , (16, 21)]
NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64
LR = 0.1
MAX_GRAD_NORM = 5.0
NUM_SAMPLES = 512

#chatbot parameters
CHECKPOINTSTEP=1000
CHECKPOINTSMALL=100
CHECKTESTMULT=5
