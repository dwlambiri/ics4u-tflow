# ics4u-tflow

Tensor Flow Project for the ICS4U Course

https://github.com/jtoy/awesome-tensorflow --
https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95 --
https://github.com/pkmital/tensorflow_tutorials -- 
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/ml_introduction.ipynb

The TensorFellow 
A Chatbot

![image](https://user-images.githubusercontent.com/28689806/187086955-56cdba30-4df1-477e-82e7-5b519e70e378.png)


Created By 
Darius W Lambiri - Rassam Yazdi - Justin Lee



ICS4U
Mr.Creelman
01/05/2018
Table of Contents

Initial Proposal ----------------------------------------------------------------------------- 3


Project Management Tools
Gantt Chart -------------------------------------------------------------------------- 7
Github Agile Project Management Tools ---------------------------------------- 8


Daily Progress Reports -------------------------------------------------------------------- 11


User’s Guide ------------------------------------------------------------------------------- 14


Program Design Details ------------------------------------------------------------------ 17


Reflection on the Project ----------------------------------------------------------------- 28


Sources and Honourable Mentions ----------------------------------------------------- 31











Initial Proposal
Artificial Intelligence Program
ICS4U Summative Proposal
Darius Lambiri, Justin Lee, Rassam Yazdi
What do we plan to make?
	We plan to implement Tensorflow machine learning to create a trainable program which does data recognition and pattern matching. Specifically we would like to implement this software to either create an object detection program or an AI chatbot.
Prerequisites / Equipment Needed
There are a few items that we need to complete this assignment: software and hardware:
Hardware:
1)	The only item that our group needs in terms of hardware is a working camera to take pictures and video with. We will use the camera to take video and pictures of an item. This is only necessary for the object detection and not the chatbot.

2)	Computers running Windows or MacOS.

Software: 
1)	Tensorflow is an open source software library for data flow programming which can be applied to a range of tasks. We plan to implement it for item recognition to detect a class of object (ie: a specific toy, or a water bottle) or e implement machine learning to create responses to a conversation for a chatbot. To write the implementation, we need to write our program in a language that supports Tensorflow. Tensorflow machine learning is very versatile and can be used for a multitude or machine learning applications.

2)	We plan to use either Java, Python, or C++. 


Objective / Learning Goals
	We set out to further our understanding of numerous computer science topics. The TensorFlow software is programmed mainly in Python, meaning that before we complete the project we have to learn a new programming language. There are 2 alternatives (Java, C++) however it is unclear on what PC operating systems they are supported. 

Throughout the course of this project we hope to gain an understanding of what is possible and with current artificial intelligence frameworks; specifically we wish to learn about machine learning, data flow programming and implementing data flow principles in our own program. 

TensorFlow is a recent open source software created by Google which is used as the basis for pattern recognition by large software companies today such as Facebook, Amazon, etc. These libraries have a multitude of applications. It is our belief that understanding and implementing will be a valuable experience for our futures in computer science. This will be a tremendous challenge but it is the ultimate learning goal for this assignment as well. We, as a group, are extremely interested in modern software technologies and this is an extremely fascinating recently released software (first released in 2014 and improved frequently).

Challenges Presented By the Project
	This process of creating this project presents a plethora of challenges for our group. First and foremost, we will have to tackle the problem of having to learn a brand new programming language, in our case, Java or Python; these are the programming languages which are able to implement TensorFlow. This will most likely require us to view many tutorials on youtube and try little programs out and work our way to the required depth of knowledge to carry out our project development. 

Secondly, we would need to figure out how to implement Tensorflow, which is a library for data flow programming, which we have not used before. The implementation of the Tensorflow libraries will be the most difficult aspect of our AI Object Detection program. Lastly, time management is going to be a challenge for us since we have a lot of research along with coding to get done in order to be able to complete our project on time, so we will definitely have to stay on task during all our class times.

Project Timetable
We envision the following steps:
1)	Install TensorFlow on our computers
2)	Research and decide which programming language to use: Java, C++ or Python
3)	Install the appropriate IDE and program language translators
4)	If a new programming language is needed, start learning the language
5)	Right small test programs for TensorFlow
6)	Design the TensorFlow program for our application
7)	Debug the program
8)	Collect data for our program. We need to start by selecting a single object which we will take video of and extract 150+ images from the video to be processed later by our program. Each picture must have slight angle adjustments of lighting changes so that the program will be able to recognise the object under any conditions. Should we do the chatbot, the data to be collected will be dialogues of some kind; currently we are looking into movie dialogue files as our testing data.
9)	Annotate the images to put the detection box around each image. We need to define in each picture what are the dimensions in which the object is. We need to do this with all images and then pass the images to the program for training. This step is not necessary for the chatbot.
10)	Create a TensorFlow TFR dataset. This requires us to store datasets into a TFRecord file format. Tensorflow requires its datasets to be formatted as such. There are several ways we could go about doing this. Our choices include  — create_object_tf_record.py and create_object_tf_record.py. We will most likely the create_pet_tf_record.py with minimal edits since labeling already creates annotations in the correct format. And finally, we will also need to create a label.pbtxt file that is used to convert label name to a numeric id. For the chatbot the data we collect will be in the form of a dialogue database. We will not make the database ourselves, as it is not imperative to the process to make one ourselves. For this step we will need to create a parser to break down the dataset and interpret the likelihoods of certain words in responses based on input (similar to object pixel and shape recognition but with words). These two processes have similar implementation but on different kinds of data.
11)	TFR datasets are created, we have concluded that we can either build upon a pre-existing model and modify it to our liking and operations, or we can build it from scratch instead. The benefit of using a pre-existing model is that  most of the features that are learnt by RNNs are often intricately constructed and fine tuning an existing model is usually a challenging process, however building from scratch is much more complex.
12)	Train of our model. This may take a while (2-3 days) as we need to input countless image files to the program for it have enough information to tackle the problem. Following that, we need to test the file. To test the model, we first select a model checkpoint (usually the latest) and export into a frozen inference graph. We test the model on our dataset whether it be a photo for the object recognition or a created questionnaire for the chatbot which we will analyse to create the best responses and fine tune the model accordingly.



	Week 1	Week 2	Week 3
Goal(s):	Accomplish Goals (1) to (5), and (8) since it is fairly simple.	Accomplish Goals (6), (7), and (9).	Accomplish Goals
(10), (11), and (12) along with testing.











Nature of End-Product
	Concept1:
	We would like our final program to be able to analyse a video and in each frame be able to detect the presence of a specific object type which we train it to recognise. To train the program we will need to create a database of hundreds of images of that item and have the program recognise the object and highlight it using a box. We then want the program to remake the video and in the newly created video display the item detection by including the detection indicator in it (the highlight). The program should also be able to recreate specific images to include the object indicators. 
	

![image](https://user-images.githubusercontent.com/28689806/187086684-5284c213-2017-4cbf-a71f-ae0c7181fbb3.png)


Concept 2:
Alternatively, should we deem the visual detection project too simple / uninteractive, we may tackle the more difficult tensorFlow task to make a chat bot. This tensorFlow implementation will be more complex but will still use the basic implementation principles in the object detection assignment (using the dataset + training a model). While it is not visually stunning to receive responses from a terminal, this self learning chatbot technology is extremely complex and can be much more versatile and interesting with regards to training and implementation. A potential visualization of the final product is as follows:
 
![image](https://user-images.githubusercontent.com/28689806/187086694-c0352729-cf22-4b85-81ad-b84d6d99bc04.png)


Project Management
Gantt Chart 
	The following is a Gantt Chart which illustrates our initial project management plan with regards to the completion of our project. Each cell on the Gantt chart has a value associated with it that denotes the percentage to which it has been completed. Currently, every task has either been completed or nearly completed. We would also log in the Gantt chart whether tasks needed to be pushed back because we could not meet the planned “O” completion date or due to absences / loss of class work time.


![image](https://user-images.githubusercontent.com/28689806/187086704-d499ef1f-8638-4b0d-933b-8d87a96da642.png)




Github Agile Project Management Tools
	To boost productivity we also used the agile project management method. To do so we put to use a helpful Github extension called the “Zenhub board” which manages tasks and organises their completion. While the Gantt chart was important for our initial vision, this log helped up be specific and consistent in logging our daily progress or lack thereof. In this chart we would create issues that need to be completed and comment on each issue to log the progress we had made. We were able to place and drag issues between a few stages: backlog, in progress, review, done, and Closed. The following are screenshots of our progress at the end of each work week. To see more: go to our github link: https://github.com/dwlambiri/ics4u-tflow (potential requirement of a Zenhub installation to see the graphic display of the issues)

Week 1 Progress:
 
![image](https://user-images.githubusercontent.com/28689806/187086713-c1a67c3e-4946-4ed8-aeb9-0cf14049459b.png)

Week 2 Progress:
 
![image](https://user-images.githubusercontent.com/28689806/187086726-11d7c146-09d3-4602-9863-2bcd01887def.png)


Week 3 Progress:
 
 ![image](https://user-images.githubusercontent.com/28689806/187086734-e8d5e2c7-dea8-4f80-85cd-ff6ede5e0ae8.png)


Week 4 Progress:
 

![image](https://user-images.githubusercontent.com/28689806/187086743-a30ada2f-05db-43ca-827f-bdde53c34b82.png)


As previously mentioned, within each issue we could comment to log what we had learned / done and what progress still needed to be made. We would also log further topics we needed to explore and did not have enough time to look at in one work session. We have a few examples of issue comments underneath.









Daily Progress Reports
12/4/17	No progress: Presentation Day
12/5/17	No progress: Presentation Day
12/6/17	We installed all the appropriate software on our computers. We each needed to download Python and an appropriate IDE to program in this language. Two of our group members working on Macs installed Eclipse and on the Windows computer we installed PyCharm. In addition, we needed to install the Tensorflow software which took a substantial amount of time due to downloading as well as applying it to the IDE. We now have TensorFlow working on both Mac computers and Python working on all three. We also researched TensorFlow implementation and Python syntax.
12/7/17	We began the day by learning some Python. Only 2/3 group members were present today; Justin was away for a hockey tournament. We began learning python from the CodeAcademy website which provides interactive tutorials and lectures which teach how to use different programming languages (we only studied Python). In addition, we ran through the tensorflow tutorial and started to understand what a tensor represents and how to utilize the computational graphs. We learned basics such as the properties of nodes, and how to use them.
12/8/17	No progress: Darius took the recursion test and Justin + Rassam were absent.
12/11/17	Rassam and Justin were both unable to participate in the summative development as they were both were required to complete the recursion test. Darius, taking matters into his own hands, took the initiative to continue learning the fundamentals of Tensorflow. We learned how to implement Tensorflow ourselves, and discussed possibilities for the future direction of the project with regards to data collection.
12/12/17	Continue the tutorial of Tensorflow. We learned the fundamentals of Tensorflow use. We learned what a Tensor is, how to create Tensor nodes and do computations with the TensorFlow computational graph. We also discussed how to implement Tensorflow to our own program. Finally, we discussed possible challenges that face the development of our program, and discussed possible meet-up dates over the Winter break to be used to further testing and development (since we experienced shortened working days due to team member absences as well as a shortened first week).
12/13/17	Breeze through numerous tutorials of Tensorflow. We researched other people's tensorflow
projects, gathering ideas and techniques that could benefit our own program. We learned how to make a loss function that squares the differences between the model and test data to be able to minimize the difference between them so our computer is going to be
able to draw the most accurate conclusions from the data it was given. Following this, our program is able to evaluate an input and determine if it correlates with the model or not.
12/14/17	We improved our knowledge of Python, and learn how to use for loops, do loops, and reading + writing to a text file. We applied loops in our early stage TensorFlow linear model optimizer program. This code that we are currently writing is mainly preliminary and mainly for educational purposes. It is highly subjective to change but we are slowly creating portions which we will need to implement in the final version of our program.
12/15/17	Justin was feeling ill today, therefore he was unable to work with Rassam and Darius. He did however, contribute towards progress documentation by updating the Gantt chart and daily log. Meanwhile, Rassam and Darius researched different optimization algorithms for the linear model to increase accuracy. This required extensive testing and analysis of the TensorFlow algorithms of advanced linear model creation (much of the code we analysed was from the TensorFlow Advanced Tutorials Page).
12/18/17	Today, as a group we spent our time learning about and testing different types of algorithms that we could potentially use for our program. We also learned about tensorflow functions, such as softmax and matmul. We also read a scholarly article about the implications of AI and Tensorflow, outlining different uses such as bots being used as online representatives of companies
12/19/17	Today we began looking into the Stanford Tensorflow course. We have discovered some video and powerpoint lectures that we have began to study. So far we have looked into the Tensorflow, introduction to tensor and RNN, LSTM, networks + GRU cells. The video
and powerpoint tutorials began simply (review from the tensorflow website) but are rapidly increasing in complexity.
12/20/17	We continued the Stanford Tensorflow course, and completed some of the basic assignments designed to improve our understanding of certain concepts like neural networks and sessions. Most of these concepts were foreign to us, but with our combined experience, we were all able to attain a solid grasp about these concepts.
12/21/17	Justin was on the Ottawa U field trip, meaning that he was unable to aid the group in their continued effort to learn and understand Tensorflow. Rassam and Darius both watched videos outlining the capabilities and potential of Tensorflow and its graphical interface: Tensorboard.
12/22/17	Today was the last day of school before the Christmas break. The group tried to work on a testable model using tensorflow, but was handicapped by the shortened length of the period. The model was only half complete by the end of class, but was still able to be run. The only problem was that the loss function was not working as we had desired, and time for debugging must be put aside.
12/24/17	Today we took the time to work on our summative over the Christmas break, and all met at Darius's house to discuss how to continue the project. We were able to debug the code we worked on in class two days ago, and managed to have it run flawlessly. As a group, we decided to take the Chatbot route, as image recognition would not be interactive or fun enough. At Darius's, we also watched part 3 of the Stanford University lectures on tensorflow, where they outline different optimizers. We decide to use Gradient Descent Optimizer, as opposed to others such as the ADAM optimizer.
12/26/17	Today the plan was to start training our chatbot. However, we ran into a problem as the code for the foundation of the Chatbot was written in both an older version of python and older version of Tensorflow and as a result it was incompatible with some of the computers. The code was not able to work with the newer version of python as strings are not treated as character arrays, and could not be parsed in the same way as planned. Some functions and terminology also had to be updated. We were, however, able to update the code and begin training, but not long enough to test the code or determine the success of the training. We finished with about 1.2 epochs with a test loss of about 4.3
12/28/17	Justin continued training the Chatbot and was able to get it to 30 epochs with a loss of 2.5. We met at Darius's house once again to begin experimentation with the chatbot. Darius was able to use external GPUs on two seperate computers to train multiple bots, all with different characteristics such as number of buckets and number of layers, and observe the differences in performance. We found that the more buckets that were present, the longer it took to train(obviously). However, we could not observe a benefit of using multiple buckets in our dataset, and therefore decided to continue with models using one bucket. We also experimented with the number of layers, making one model with three, one with four, and already having Justin's with one. We also found no significant benefit to having more layers. We are still examining the issue.
1/1/18	We continued with experimentation of different models, specifically optimizers. When viewed through tensorboard(software that visually displays information about your tensorflow program), we saw that using our conventional Gradient Descent Optimizer, words with high
frequency were being clumped together. This is not what we want, as this means that the chatbot will continue to give similar answers for different questions. The ADAM Optimizer, however, was much better on all fronts. It solved the clumping problem, and instead grouped
uncommon words together which are never used. The ADAM Optimizer was also more efficient, and generated a better loss than a similar program using the Gradient Descent Optimizer with the same number of epochs.





User’s Guide
TensorFellow has many user features. Since python is an interpreted language (TensorFellow is written in python), in order to run our program you will need to run both python and tensorflow. Versions of python of 3.5 or 2.7 will suffice (64 bit) and please download version 1.4.1 of TensorFlow (this is the latest version as of December 21 2017).
	
●	To download python, follow the steps in the following link: https://www.python.org/downloads/


●	To download TensorFlow, follow the following download link AFTER you have already installed a version of  python: https://www.tensorflow.org/install/  (make sure to select your operating system)

	Once you have downloaded the above software PROPERLY, you are ready to run the code! In order to do so, follow one of two procedures (terminal procedure OR .bat procedure). The procedure you use is dependant of the feature you wish to use and the operating system that you are working on. 

	




Training (ONLY terminal procedure)
TensorFellow’s personality and quality of answers is dependant upon the parameters with which he is trained with in the config.py file. TensorFellow is released with two pre-trained models; each model has unique configuration parameters, therefore if you just wish to chat, training the model is not necessary. However, there are plenty of operating variations of the TensorFellow, which can be amusing for a user well versed in Tensorflow. 

The dataset that is trained upon is the Cornell Corpus Movie Dialogue Database. This is the only database we tested on and the only database that a user can retrain only on that dataset using our code. Should they choose to, a user can edit TensorFellow and retrain him with new training characteristics and parameters in the config file. 

 In order to retrain you will need to enter the terminal and change directory to the folder possessing your code (ie adam4Layer). The Unix (macOS / Ubuntu) change directory command is: cd. On Windows the change directory terminal command is also cd but directories are found differently (such as the use of backslash character instead of forward slashes). The change directory command is applicable to run ANY program from the terminal / command line and is useful to remember.The following is an example on my mac computer. 
Then to retrain you will need to remove the checkpoints folder and the processed folder (they will be remade properly when you retrain). Then make sure your “cornell movie-dialogs corpus” folder is in the directory with your other code. To retrain type into the command line: 

 (if you desire to use GRU type cells, which is the default)

 (to use with LSTM cells)

Also, if you are on Mac you need to type “python3” instead of “python” as we were not able to install Tensorflow under python 2.7 which comes with the Mac.
Your model will now train with the parameters that you have provided. It is recommended to download the TensorFlow GPU software to train more quickly. Training will increase in speed depending on the computational power of the processor doing the training. Training will save progress training per Epoch which is customizable from the config.py file. An Epoch in machine learning language represents 1000 iterations through the learning process. As you train, you will notice that the chatbot will become gradually better. Each checkpoint is saved into a checkpoint file. 
Chatting (Terminal OR .bat procedure)
Chatting is the most fun feature of TensorFellow! There are two ways to initiate chatting with TensorFellow:

The first method is through the terminal . As previously mentioned in the training section you need to first need to do the cd command and change into the proper directory. Then once changed into the proper directory you input the following line into the command terminal (remember to switch to python3 for mac).  There are other command line prompts depending on the configuration that the user has trained with, but the default one can be seen below (for the pre-trained model).


The second method to run the program is by clicking on the .bat file inside the adam4Layer folder. The .bat file automatically enters the previous lines into the command terminal eliminating possibility for error. This only works on WINDOWS computers.


 **After one of the above methods is completed, the chat mode will run!

You can experiment with by using the same phrases/sentences with different punctuations to see how the bot responds. This trick demonstrates the program's different reaction to the input  because as the input must be read backwards in order to produce a forward oriented output (see Program Design), the end of sentence punctuation plays a pivotal role in determining the location of related word and phrases in the HIDDEN_SIZE internal cell state size. 

As a forewarning, our data is taken from cornell movie scripts, so there is profanity,  inconsistency, misspelled words and strange punctuation. Some of the answers that the bot produces may seem odd or nonsensical, but this is due to the Cornell data set that we worked upon: since the answers are generated are based on the dialogues in films (which for the most part are different and more dramatic than normal conversations), the bot is subject to produce seemingly odd answers; however, it is normal for it to do so. Should we have had a larger and more realistic dataset, then the bot generated responses would be more accurate. This is also why the bot does not have a “consistent” personality. The answers in the scripts are from many characters which the writers have endowed with very different personalities. 

All conversations are automatically time-tagged and saved into a text-file for you (output_convo.txt); this has other benefits too (ie. looking for improvements in answer quality after training ). Once you are done chatting with the Tensorfellow, exit by going to next line then hitting the enter key, then you can input other command lines if you desire. 
Program Design Details
Tensorflow (see www.tersorflow.org for more details)
TensorFlow is an open source software library for numerical computation using data flow graphs. It was originally developed by Google’s Machine Intelligence research organization. 	
				
Nodes in Tensorflow’s data flow graph represent mathematical operations, while the edges represent the multidimensional data arrays (tensors) communicating between them. The advantage of the  architecture is that it allows users to build complex models. TensorFlow programs use a tensor data structure to represent all data – only tensors are passed between operations in the computation graph. We can think of a TensorFlow tensor as an n-dimensional array or list. A tensor has a static type, a rank, and a shape. 									
Variables in Tensorflow are similar to regular variable in that they never change type nor lose their value when it is saved to the disk. Typically, variables are parameters in a neural network. In our example, weights W and bias b are variables.	
				
Placeholders are nodes whose values are fed in at execution time. The rationale behind having placeholders is that we want to be able to build flow graphs without having to load external data, as we only want to pass in them at run time. Placeholders, unlike variables, require initialization. In order to initialize a placeholder, type and shape of data have to be passed in as arguments. Input data and labels are some examples that need to be initialized as place- holders.  
						
Tensorflow uses sequences of connected nodes that perform mathematical operations on inputs to produce desired results. The nodes each contain a specific variable matrix, referred to as a weight, that is combined with the input that make each neural network unique. The weights are set when the model is trained where, using positive reinforcement, the model uses trial and error to find the best weights to produce desired results.

For this project, we used a specific type of neural network called a recurrent neural network (RNN) , which contains hidden states derived from previous inputs that acts as memory for the network. This type of model has been proven to work best for translation problems, and we thought it successfully table to use ourselves. However, this type of neural network poses a problem called the vanishing gradient problem. This is when, after too many operations, the hidden state matrix becomes ‘diluted’, and forgets earlier information. This can happen, if the network is overtrained. To combat this problem, researchers have invented special nodes called LSTM(Long Short Term Memory) cells and GRU cells, which we used as building blocks for the chatbot. However, we observed that the performance of the LSTM cell is very similar to the GRU cell. This is in line with similar observations by other people. Because the GRU has a simpler computational structure, it trains faster. Thus, we used GRU cells for most of our experimemt. 



What we changed from the original code
We have made extensive changes that can be checked by diffing the original files with the files that we provided. In this section we will detail some of the more substantial changes made to the code. 
The original code did not run as provided. That was because Google has changed the location of the seq2seq code in its Tensorflow base. The Stanford code did not use Tensorflow 1.4 and in this version Google has moved the seq2seq to the “contrib” tree.
It took some time to figure out how to change to make it Tensorflow 1.4 compatible. Once we changed all accesses to seq2seq related code to the appropriate paths the code started to run in a Python 2.7 environment. However the code did not run in Python 3.5. This was ok for the Windows machines that we have (Justin’s and some of Darius’ machines). However we also have 2 MacOS machines which needed Python 3.5 to run Tensorflow.
To update the code to work with Python 2.7 and 3.5 we needed to investigate the differences between the 2 language version. We discovered that the base code used a string parsing method not possible in python 3.5 as strings were no longer treated as char arrays. The base code also contained functions that were renamed in the latest version. 
The original model used punctuation marks such as commas and apostrophes; however, we felt that this was both unnecessary and a hindrance to the bot’s performance.  Because of this, we decided to make the Chatbot usable with and without punctuation. The program behaves extremely differently depending on this condition, as the input must be read backwards in order to produce a forward oriented output. This means that the hidden states will start much differently based on the type of punctuation used. We also added the option to not use punctuation(use a vocabulary without punctuation) and to make the response more readable, add punctuation to the sentences produced using the non-punctuated dictionary.


Code Structure - The code uses 4 files which we detail below:

Model.py
This file uses the seq2seq model that was originally that was originally invented for language translation. The chat model "translates" from English to English, but translates a user’s question or statement into a response. The file contains a class that implements a multi-layer recurrent neural network as encoder, and an attention-based decoder. We configure the model with appropriate parameters so it can be used for a chatbot. This class also allows to use GRU cells in addition to LSTM cells, and sampled softmax to handle large output vocabulary size.









The seq2seq model with attention that we are using will create 3 structures:
-	An encoder that is of width MAX_ENC_SENTENCE_WITH and depth NUM_LAYERS
-	A decoder that is of width MAX_DEC_SENTENCE_WITH and depth NUM_LAYERS
-	An attention mechanism (ie a means to send state from encoder to decoder)
During training the encoding statement and decoding statements are inputs to structures 1 and 2 and the output is not fed (ie the dotted lines are *not* used).

Note that the output is a vector of the size DICTIONARY and the most probable word is the one in the vector with the highest numerical values.

The quality of the attention layer matters a lot. In theory a good attention decoder would weigh all words and thus punctuation should not matter that much. This however for our bot is very difficult because there are many answers to the same question posed with different punctuations in the data set. See the following exchanges:

HUMAN ++++ hi!
BOT ++++ jeeezzz !
HUMAN ++++ hi
BOT ++++ you smell like garlic . you ?
HUMAN ++++ hi?
BOT ++++ hi .
HUMAN ++++ hello!
BOT ++++ makes you think im under this ?
HUMAN ++++ hello
BOT ++++ you ready ?
HUMAN ++++ hello?
BOT ++++ 
HUMAN ++++ how are you?
BOT ++++ im hungry .
HUMAN ++++ how are you
BOT ++++ 
HUMAN ++++ where are you?
BOT ++++ in my room .
HUMAN ++++ where are you


During decoding the blue cells are fed the user input and the first red cell is fed a special character (“<s>” in our case). The dotted lines are in use and the system goes through several loops, moving words towards the right in the decoder. The output is then provided to the user.

We heavily edited bot model class. Here are some changes we made to the code, in no particular order:
-	Detailed comments explaining model type, purpose, cells used, and data member descriptions
-	Only one dictionary as opposed to the original model that used 2. The original code translates language A to language B. We ‘translate’ from English to English.
-	Added variables expressing which optimizer to use
-	Changed code to allow for different optimizers to be used
-	Added createCells method that creates GRU cells or LSTM cells
-	Changed names of numerous method as the names did not relate to what the code was doing
-	Adjusted threshold factor to decrease frequency of unknown tokens.
-	Refactored runStep to model class, made some small changes
-	Increased batch size, meaning that we increased the amount of sentences it processes at once during training. It still does the same amount of computations, but does them in different increments. Using the increased batch size and the Adam optimizer, we were able to reduce the loss to 0.44, making it our best model yet (see next section for more information) .
-	Added comments
-	Cleaned up imports



Config.py
A python file that is used to store changeable parameters that influence the chatbot. These include variables like the path to the training file, a boolean variable expressing whether or not the user wants the chatbot to use punctuation, dictionary threshold and question answer difference threshold. It also contains the number of buckets and the size of the buckets to be used for the model. In some cases, if you train a model on these parameters, you must retrain the model to apply any changes made. 

The following parameters will heavily determine the outcome of training:
-	THRESHOLD = 3 : This parameter determines how big the dictionary will be. During processing all words found in the movies_lines file are introduced in a dictionary that keeps count of how many times the word appears in file. This dictionary is then sorted in decreasing order (ie higher count numbers at the top). To reduce the overall dictionary size only words that show THRESHOLD or more times in the input text are then saved in vocab.txt. The data is then processed to replace the words that did not make it with the <unk> token. Therefore this parameter controls dictionary size.
-	QADIFF_THRESHOLD = 30 : Only Q and A tuples that are less than QADIFF_THRESHOLD different in length (as measured in number of words) are used. This makes the input set more “balanced” as sometimes in movies characters go on monologue rampage. 
-	BUCKETS = [(30, 30)] : This list has entries for each “bucket”. Each tuple represents the number of words in the encoder and the number of words in the decoder. First value represents the number of blue cells in a row in the picture above and the second value represents the number of red cells in a row in the picture above. These 2 can be different. We experimented with many geometries.
-	NUM_LAYERS = 1 : this represents the depth of the RNN multi cell (ie how many rows of cells are in figure above). In the figure above there are 2 layers. We experimented with 1,2,3,4 layers.
-	HIDDEN_SIZE = 512 : this is the size of the internal state for each of the cells in the model. 
-	BATCH_SIZE = 256 : this represents the number of Q and A pairs that are grouped together for a learning step. This parameter has big impact on learning and performance, although we could not understand very well how it works. This parameter is used only during learning. During regular use the batch is of course 1 as only 1 sentence is passed to the model.

We have done the following changes to the file:
-	Added option that allows user to toggle the chatbot’s use/disuse of punctuation
-	Added checkpoint variable that allows user to change the frequency of checkpoints based on the processing power of their computer
-	Automatically deletes earlier checkpoints to model learning efficiency, but keep enough to maximize learning efficiency

Data.py

This file creates all the subfiles for seq2seq model training and chat sessions, and processes the Cornell movie corpus. The file reads the list of conversations and splits the data into two files: train.enc and train.dec. These files are used to train the bot, with train.enc being the input and train.dec being the desired response. Data.py also contains a sentence parser that splits the sentence into a list of words, then puts it through a dictionary to turn it into a list of numbers. Data.py then pads the sentence if necessary. Words are then added to the vocabulary, and the sentence is ready for input into the neural network. ALso contains a variable that makes the chatbot produce longer sentences as answers during the chatting mode by using a decaying multiplier on all words other than the endline token.





The initial files are broken into tokens using regular expressions. We have implemented 2 variants: 
-	One with punctuation
-	One without punctuation
USEPUNCTUATION=True
USEAPO = True
if USEPUNCTUATION:
    REGRULE = "([a-z]+|[\?]+|[!]+|[\.]+|[\']+|[,]+|<s>|<unk>|<pad>|</s>)"
    if USEAPO:
        PUNCTCHAR = ",.?!\'"
    else:
        PUNCTCHAR = ",.?!"
else:
    REGRULE = "([a-z]+|<s>|<unk>|<pad>|</s>)"
    USEAPO = False

The REGRULE contains the recipe that will break a line into pieces. Each rule is separated from another rule by the “|” character. Because some characters such as ‘.’ ‘!’, ‘?’ to match them in a rule we need to append them by ‘\’. 
(For more information see here: https://docs.python.org/3/library/re.html)

For easy processing every character is set to lower case before the sentence is passed to the tokenizer. Once broken into words the sentences are transformed into series of numbers.

Example:
Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.

Is transformed into (each integer represents the position of the word above in the vocab.txt file)

43 25 119 31 916 9 3 3 21 4509 6409 40 430 88 4019 17973 1227 514 65 42 10 3 4 189 4

We have done the following changes to this file:
-	Detailed comments for explaining functions
-	Improved readability: create and adjust variable names, improve syntax, 
-	Added catch to make sure input and output are formatted correctly
-	Added condition that analyzes the user’s use of punctuation and adjusts its own response similarly. Punctuation is extremely important, as the input must be read backwards in order to produce correctly oriented results
-	Added a vectorDiff function that returns the difference of the question and answer sentence length
-	Eliminated unnecessary/inefficient code, improving clarity and readability
-	 Added a function that improves the ‘chattiness’ of our chatbot. This uses a decaying variable that is raised to y(value less than 1) then multiplied by every word other than the end of line token.

Chatbot.py

This file contains the main program that is the chatbot. This file is responsible for saving the changes it made after training, and creating a checkpoint file after a specified number of iterations. The file asks the user for their question or statement to be delivered to the chatbot and uses the other files to generate a decent response. The file then cleans up the response(adds punctuation if the dictionary contains none) to make it more readable. Also contains a test function that asks the chatbot a list of questions and prints the answer in a text file. This is useful to reduce time typing when testing performance, and for viewing improvement of the bot over time. 

The file has many changes and additions:
-	Detailed comments for explaining functions
-	Improved readability: create and adjust variable names, etc.
-	Added a test mode that asks the chatbot a list of questions and prints the answer in a text file. This is useful to reduce time typing when testing performance, and for viewing improvement of the bot over time. 

Tests and Experiments

We tested program with different parameters. We also experimented with various seq2seq configurations to see which one provides better results. At first, we trained our model with one bucket and one layer, but once acquiring more computing power, trained other models with more buckets, more layers, etc. We finally settled on a chatbot with 4 layers and 1 bucket. Some parameters we kept unchanged.

The figures below depict collected performance data during training. In NN language an Epoch is 1000 iterations through the model during training. The Loss is generated by the Tensorflow model as a numerical representation of how well the training is going. The lower the value, the better the training.

The first figure uses a linear x and y axes. The loss curves over Epocs are dropping very fast over the first Epoch. The naming of the curves below is as follows: LxLyHzB[string]vqDw => x layers in the model, each using y Hidden states, while the model was trained using z Batched sentences. [string] represents the type of punctuation used (“apo” with apostrophies, “noapo” no apostrophies). “Vq” represents the dictionary cutoff for the experiment: v2 means a THRESHOLD of 2 was used while “v3” means a THRESHOLD of 3 was used. The larger the threshold the smaller the used vocabulary and more <unk> symbols being used in sentences. 
“Dw” represents the maximum length difference (as measured in words) between the Q and A sentences that were used to train the model: D8 represents an 8 word maximum difference, while D30 represents a 30 word maximum difference. The larger the difference, the larger the set of sentences that was used to train. In general the D8 set resulted in about 100k pairs of sentences while the D30 set resulted in about 180k sentences.

![image](https://user-images.githubusercontent.com/28689806/187086806-5990afe0-4590-4230-8e82-b15b90d85b88.png)



The  2nd figure uses a logarithmic (base 10) for the loss (y axis) to better see the data difference.


![image](https://user-images.githubusercontent.com/28689806/187086823-7eb90ce8-470d-4bf4-83e9-4cfb42b92ea5.png)


The next figure displays the x axis in log10 base. This allows for better view of what is happening in the first Epoch.


![image](https://user-images.githubusercontent.com/28689806/187086832-004917c6-e160-4bbc-99a7-2aa98e52b745.png)


We experimented with different optimizers. At first, we used the default Gradient Descent  Optimizer, yet realized that the Adam Optimizer better suited our dataset. While we have trained several bots with GDO the data above is only for Adam Optimizer.

In all we have done a large number of experiments over the span of 2 weeks. We have collected more than 100 GB of data from the models. We were able to do this by using GPUs to speed up the computation.

We had 2 machines with GPUs to run the training:
-	One machine with a 2 GB RAM on the GPU and 16 GB RAM on the CPU
-	One machine with 4GB RAM on the GPU and 32 GB RAM on the CPU.

With 4Gb GPU, 0.4s per step, totaling 100,000 steps(11.1h total). We observed, while running on GPU, fans were not running, meaning that using a GPU is much more energy efficient, thus reducing our carbon footprint. The GPU also was able to train the model much faster.

The images above display only Epochs (ie 1000 step increments). However what is not represented is the amount of time it takes to compute a step. The curves that drop faster need less Epochs but the per step computation time is increased. For example the blue and green curves have an average per step computation time of 0.95s while the brown curve as computation time of about 1.1 seconds. All these times were on the GPU. On a CPU the exact same model runs about 13 times slower: the steps would increase in time form about .95s to about 13 seconds.
Tensorboard
Google provides a very useful companion program to tensorflow named Tensorboard. When run on a model checkpoint Tensorboard can display information about the data and about model itself. To see the model structure (ie. how the blocks are connected) we have instrumented the files to write it to file. Every Epoch we wrote the model to the disk and we were able to visualize the relation between words that formed the vocabulary.

To run Tensorboard, type the following:
>tensorboard --logdir checkpoints --port 5005

You can set the absolute file path for the --logdir parameter. The --logdir parameter tells Tensorboard where the model files are. The --port parameter tells tensorboard what IP port to use. By default it uses port 6006 but we were able to run multiple tensorboard instances with different ports to compare different models and data.

For example we run:
>tensorboard --logdir adam1/checkpoints 
>tensorboard --logdir adam2/checkpoints --port 5005

Tensorboard allows one to visualize the actual model. The next 2 figures are screen shots from one of our many models. We are still learning many of its features.

![image](https://user-images.githubusercontent.com/28689806/187086879-bd84f13f-a27a-4543-bf0a-4d797ea0df86.png)




The most interesting feature relates to data visualization. Most of the models showed a vocabulary structures as below: a bowl of low use words somewhat separated from the more used words. Each dot below represents a word in the dictionary. 


![image](https://user-images.githubusercontent.com/28689806/187086893-af91ea58-e9bb-4f28-8683-0984622e2bf7.png)
![image](https://user-images.githubusercontent.com/28689806/187086899-5915c4ca-ab80-4621-8cb5-1b32e9a32d78.png)

In fact when clicking a dot the following is revealed:

![image](https://user-images.githubusercontent.com/28689806/187086913-fb1af654-312c-4fdf-8a2b-c94b62e432de.png)


The image on the right is a list of words produced on Tensorboard that ranks the likeliness of words to appear based on a selected word in the graph being in the sentence. 
Reflection on the Project
During the course of the project we faced a variety of challenges and difficulties but we fought through and surmounted them to come out with an extremely satisfying end-product. At the start of our project, we had no idea about using Tensorflow, Python, or neural networking so a big chunk of the start of our project was allocated to learning our way around these new concepts to be able to create our chatbot, plus we lacked the required software on our computers to start making progress. 
We began by downloading Eclipse as our IDE, then we downloaded Python (versions 2.7/3.5) along with TensorFlow. However, we had difficulties with Justin’s computer which caused a small setback. 
Following that, we used TutorialsPoint to deepen our knowledge of python so we can not only write code, but also understand the code we researched on the internet. Once we were set on that, we started one of the most challenging tasks of familiarizing ourselves with Tensorflow. We began with going through the basic tutorial on the website, learning about Tensors (nodes on the computational graph), carrying out mathematical operations, building linear models, and training the model to work with a data set with minimal error. 
Building off that base, we moved on to making simple programs that use a dataset and form a line of best fit for extrapolation, as well as a handwriting recognition program, that can tell apart all numbers. This lead to us taking our research and independent learning a step further by watching Stanford University Tensorflow tutorials on youtube, which were pivotal for us to carry out our project since it taught us a lot about neural networking, specifically, Recursive Neural Networks (RNN’s). We also learned about sequence to sequence machine learning to be able to train our model into responding to anything the user has to say.
After our new depth of knowledge in this area increased, we set out on picking a model for our chatbot and making proper adjustments to it so it works best for our data and purpose. We found a number of models on the internet. Interestingly all seemed to be very close in syntax: we believe all were derived from the same code. We picked the Stanford code as the course instructor stated that it worked and posted quite a number of examples of its output.
We quickly discovered that the code did not work with Tensorflow 1.4.1, the version we downloaded or with Python 3.5. We had to overcome these problems before we were able to proceed.
We picked a picked a very large dataset of movie dialogs which we then had to parse through and allow our model to synthesize the information in the data set and develop proper responses. We tried experimenting with our model so we made many different versions, some with more buckets (categorize sentences based on lengths), some with more layers (increase complexity of model to reduce error), and one without punctuation (more words synthesized opposed to punctuation marks). Therefore, once the testing phase of these models were over, we picked the best performing one and made it our main model, and we incorporated selective punctuation where the bot only responds with punctuation if the user does too. 
Upon completion of the main program, we deliberated on whether we would like a fancy graphics package for the chatbot (use something like allegro with animation) or experiment with the various model parameters. We decided that experimentation would provide us with more insight into machine learning and we spent many hours doing that. In fact most of the machine learning is about “How to train your model” to paraphrase a computer animated movie. The performance experienced by the program user, as in Olympic training, depends heavily on the knowledge of the coach: as we explain above there are many knobs to turn the final product which the *trained* model depends heavily on them.

After a lot of experimentation we discovered the following very important rules. The rules are listed in the order of the importance that we believe should be accorded to them.
-	Adam optimizer has significantly better performance. We discarded the use of GDO once we started using Adam. This was the first and most important breakthrough we had. The original code did not have an Adam optimizer and we coded its use by looking at other examples on the internet.
-	The batch size of the data that is passed during training makes a huge difference. The larger the batch the better. We experimented with batches of 64 sentences, 128, 256 and 512.  We hit GPU memory limits for various hidden state sizes and batch sizes. This is clearly visible in the pictures that show loss vs Epoch for various experiments. The curves get better and better as the batch is increased. This is one parameter however that we could not understand very clearly how it works as the Google documentation is very sparse in this area.
-	The hidden state size has an effect on the quality of the bot: higher hidden size is better. We experimented with 128, 256, 512 and 1024 sizes. We could not use sizes larger than that because of memory limits on the GPUs. 
-	We experimented with models of various depths (1,2,3,4). Deeper models seem better but here again there is a memory limit. Single layers with with wide hidden states performed best. This experience is similar to what we saw regarding image recognition: over the years the image recognition stacks have less and less depth but are “wider” (more state). This was in some of the lectures in the Stanford course on Tensorflow.

Overall we discovered that the larger the hidden cell size (512 or 1024 states), the bigger the batch and 1 layer are the best parameters. Overall our computers have generated a lot of heat over the 2 weeks we have experimented. The amount of data we have collected is yuuge. This was the most interesting part of the project: the programs were super tiny compared with the data sets and a lot of time was spent just processing data.

The parts of the project that we are proud of would be a couple of different things. First of all, we are very proud of the way we were able to use Tensorflow in our project to really demonstrate the computations carried out. We are also very satisfied that we could use Tensorboard: for example, we can load a graph of the bot’s vocabulary and how often each word is used in relation to each other so it really gives a neat insight to what sort of words the majority of responses comprise of. Also, there is a view where one can visualize the model and how different parts of the network interact with one another. Secondly, we are proud of the way we handled the very heavy training of our models by using an external Graphics Processing Unit to speed up training since without it, we had to wait for days to carry out 50 Epochs (50000 iterations) of training. Lastly, we are very proud of the way we were able to learn all this information so quickly, even though it is a state-of-the-art field in computer science most of the breakthrough work in neural networks language recognition occuring after 2014. While our bot will certainly not pass any Turing test, the fact that it can give some reasonable answers in at least some cases is very impressive.
	Some of our struggles during the creation of our chatbot would be debugging and the sensitivity of our code to even the most minute logic changes, even if they are not necessarily wrong. For example, we had a very big problem dealing with punctuation marks and whatever inputs that were given to the chatbot, it would result in a long string of periods and question marks. We only discovered this was causing the bogus responses when we made a second version of the model that ignores all punctuation and artificially added them to the end of sentences. However, after 20 epochs of training, it started to work with punctuations, and as well as the sentence structure and punctuation is, the answers were not as interesting as the on without punctuation. Lastly, we also had difficulties learning Tensorflow to start the project since it is so difficult to visualize, and because it uses concepts that were completely foreign to us prior to our project (ie. neural networks, computational graphs, convolutional layers, etc).
	If we were to redo this project, we might look into using an even better computer with a larger memory and RAM so we can feed it more data without running into segmentation faults, where the computer simply runs out of memory to keep the training going (occured on Justin’s computer). The greater amount of data would better prepare the bot to develop long and coherent responses. Moreover, we would definitely use the external GPU a lot sooner so we can train each variation of the model a lot more and select the best one with confidence, then also train that one plenty as well, where we currently lost valuable time that could have been used to train our models to a greater depth.
	In terms of cooperation and classroom community, it was basically non-existent with all of our peers who were not a part of our group because they had completely different tasks to deal with themselves and our project is so advanced that it is highly unlikely any of them understand enough to help us. However, within our group, we cooperated very well with no conflicts between group members and a high degree of efficiency when it came to learning together and explaining our ideas to each other, and dividing the work up between each other and meeting almost all our personal goals/deadlines. We had very good chemistry and used technology to effectively communicate over the break about the project and organize meetings. Furthermore, we also owe a lot of our success to many sources that we used to deepen our knowledge of Tensorflow and deep learning algorithms such as youtube videos by helpful people, TF tutorials, Stanford Slideshows and etc.


Sources and Honourable Mentions
In this section, We acknowledge all the people/sources of informations we have encountered and used during the length of our project. Without their help in our learning and contributions, we would not have achieved the success we have been able to up to this point. Special thanks go out to the following people and sources:

Tutorialspoint - Python:
Familiarization and learning the Python language in great depth, both simple and complex concepts.
https://www.tutorialspoint.com//python_online_training/python_overview_of_oops_terminology.asp

TensorFlow API:
Covering the conceptual familiarization (ie. computational graphs) and explanations of the basics.
https://www.tensorflow.org/versions/master/tutorials/seq2seq


Stanford Tutorials (CS 20SI)
Review previously learned material and add greater depth, plus introduction to RNN’s (GRU’s/LSTM’s).
https://web.stanford.edu/class/cs20si/syllabus.html


Labhesh Patel
Clarified confusions about the Stanford slides by making in depth explanational youtube videos.
https://www.youtube.com/channel/UCMq6IdbXar_KtYixMS_wHcQ


Suriyadeepan Ram
The person whose code we used as a basic outline of our own program, allowing us to improve off of it and build something better and more applicative to our intent, and plenty of teaching about seq2seq.
http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/


WILDML
Introduction to chatbots and how deep learning is used for generative-based models, plus pros and cons.
http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/

Andrej Karpathy
Using and implementing Recursive Neural Networks, plus the different tasks they can perform
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

Deep Learning 4j
Explain Recurrent Neural Networks and Long Short Term Memory Networks
https://deeplearning4j.org/lst

