This file contains the data accompanying the following paper:

Lara Grimminger and Roman Klinger (2020):
Hate Towards the Political Opponent: A Twitter Corpus Study of the
2020 US Elections on the Basis of Offensive Speech and Stance Detection.
11th Workshop on Computational Approaches to Subjectivity, Sentiment &
Social Media Analysis (collocated with EACL 2021).


CONTACT:
--------
In case of questions please contact
lara.grimminger@ims.uni-stuttgart.de and/or
roman.klinger@ims.uni-stuttgart.de



CONTENTS OF THIS FILE
---------------------

* Introduction
* Data
* Columns
* Annotation Guidelines



INTRODUCTION
------------


Given the text of a tweet, we annotated the stance the tweet text holds towards our pre-determined targets (Trump, Biden and West) and the presence or lack of hateful and offensive speech.

The detected stance is from one of the following labels: Favor, Against, Neither, Mixed, Neutral mentions. 
Hateful and offensive speech (HOF) is labeled with Hateful or Non-Hateful.


DATA
----

The annotated data is stored as a TSV and split into a train and test set.

The split percentage is:
train 80%
test 20%



COLUMNS
-------

Each data file contains 5 columns:

	text	=  text of the tweet.
	
	Trump 	=  if tweet text is favorable, against, neither, mixed or neutral towards target Trump.
	
	Biden 	=  if tweet text is favorable, against, neither, mixed or neutral towards target Biden.
	
	West 	=  if tweet text is favorable, against, neither, mixed or neutral towards target West.
	
	HOF 	=  if tweet text is hateful and offensive or neither hateful nor offensive.
	
	
ANNOTATION GUIDELINES
---------------------

We included our Annotation Guidelines.
