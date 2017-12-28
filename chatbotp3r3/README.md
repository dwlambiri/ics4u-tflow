<<<<<<<
A neural chatbot using sequence to sequence model with
attentional decoder. 

Is is based on the one created by Chip Huyen 
as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

The detailed assignment handout and information on training time can be found at http://web.stanford.edu/class/cs20si/assignments/a3.pdf 

We have changed the data processing file to extract only
select converstations. In particular we removed all conversations
that are not within QA_THRESHOLD in length of each other.

We modified the code to run it on python2 and python3

<h2>Usage</h2>

Step 1: create a data folder in your project directory, download
the Cornell Movie-Dialogs Corpus from 
https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Unzip it

Step 2: python data.py
<br>Configure the config.py for data processing parameters
<br>Set the buckets and thresholds accordying to what your needs
<br>Process the data set
<br>You should have a directory called "processed"
<br>In it there are several files that will be used for training

Step 3:
python chatbot.py --mode train <br>
If mode is train, then you train the chatbot. By default, the model checkpoints
and will restart from the checkpoint if there is any

If you want to start training from scratch, please delete all the checkpoints folder.

If the mode is chat, you'll go into the interaction mode with the bot.

By default, all the conversations you have with the chatbot will be written
into the file output_convo.txt in the processed folder. 

Training the bot can take a while depeniding on several factors.

Step 4:
python chatbot.py --mode chat <br>
Chat and see what it says.

