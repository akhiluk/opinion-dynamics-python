#Basic imports
#-------------
import os
import subprocess
import ebooklib
import nltk
import sys
import re 
import json
import codecs
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#'From' imports
#--------------
from flask import Flask, flash, Markup, request, render_template
from IPython.display import YouTubeVideo
from os.path import exists
from werkzeug.utils import secure_filename
from ebooklib import epub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from ruamel.yaml import YAML

#NLTK requirements
#-----------------
nltk.download('punkt')
nltk.download('stopwords')

#Basic Flask app set-up
#----------------------
app = Flask(__name__)
app.secret_key = "N7w0$*4*AUF"
upload_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_FOLDER
app.config["SESSION_TYPE"] = "filesystem"

#Routes and methods
#------------------
@app.route('/')
def index():
	return render_template('main.html')

@app.route('/ebook')
def funcEbook():
	return render_template('ebook.html')

@app.route('/video')
def funcVideo():
	return render_template('video.html')

@app.route('/video-analysis', methods=['GET', 'POST'])
def funcVideoAnalysis():
	if request.method == 'POST':
		chapterwise = []
		myDict = {'Anger':0, 'Anticipation':0, 'Disgust':0, 'Fear':0, 'Joy':0, 'Sadness':0, 'Surprise':0, 'Trust':0}
		counts = {}
		tester = []
		tester.append(request.form.get("videoID"))
		tester.append(request.form.get("videoID2"))
		for y in range(0,2):
			
			superCmd = "youtube-dl --extract-audio --audio-format wav --output \"download.%(ext)s\" "+tester[y] #https://www.youtube.com/watch?v=FZUcpVmEHuk
			subprocess.call(superCmd, shell=True)
			flash("Video downloaded!")
			secondCmd = "ffmpeg -loglevel panic -y -i download.wav -acodec pcm_s16le -ac 1 -ar 16000 test.wav"
			subprocess.call(secondCmd, shell=True)
			changeSample = "deepspeech --model deepspeech-0.6.1-models/output_graph.pbmm --lm deepspeech-0.6.1-models/lm.binary --trie deepspeech-0.6.1-models/trie --audio test.wav > file_"+str(y)+".txt"
			subprocess.call(changeSample, shell=True)
			flash("Text conversion done!")
			
			name = "file_{}.txt".format(y)
			f=codecs.open(name, "r", "utf-8")
			data = pd.read_csv("vid_emotions.csv",sep=',',delimiter=None)
			fish = data['DaList'].tolist()
			def emotionCounter(k):
				klen = len(k)
				for i in range(0,klen):
					if (k[i]==1):
						if(i==1):
							myDict['Anger']+=1
						if(i==2):
							myDict['Anticipation']+=1
						if(i==3):
							myDict['Disgust']+=1
						if(i==4):
							myDict['Fear']+=1
						if(i==5):
							myDict['Joy']+=1
						if(i==6):
							myDict['Sadness']+=1
						if(i==7):
  							myDict['Surprise']+=1
						if(i==8):
							myDict['Trust']+=1
			myDict = {'Anger':0, 'Anticipation':0, 'Disgust':0, 'Fear':0, 'Joy':0, 'Sadness':0, 'Surprise':0, 'Trust':0}
			for x in f:
				tokens = word_tokenize(x)
				words = [word for word in tokens if word.isalpha()]
				#myDict = {'Anger':0, 'Anticipation':0, 'Disgust':0, 'Fear':0, 'Joy':0, 'Sadness':0, 'Surprise':0, 'Trust':0}
				for j in range(len(words)):
					words[j] = words[j].lower()
				stop_words = stopwords.words('english')
				words = [w for w in words if not w in stop_words]
				for word in words:
					if word in counts:
						counts[word] += 1
					else:
						counts[word] = 1
					for item in fish:
						if(word == item):
							rows = data.loc[data['DaList'] == word]
							for i, j in rows.iterrows():
								k=j.to_list()
								emotionCounter(k)
			chapterwise.append(myDict)
		flash("Analysis done! Generating plot...")
		df = pd.DataFrame(chapterwise)
		print(df)
		ax = plt.gca()
		df.plot()
		plt.xlabel('Videos -->')
		plt.ylabel('Count')
		vfig = plt.gcf()
		plt.draw()
		full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'video.png')
		vfig.savefig(full_filename3, dpi=100)
		return render_template('video-output.html', item=full_filename3)

@app.route('/ebook-upload', methods=['GET', 'POST'])
def funcEbookAnalysis():
	if request.method == 'POST':
		f = request.files['file']
		f.save(f.filename)
		book = epub.read_epub(f.filename)
		chapters = 1
		for item in book.get_items():
			if item.get_type() == ebooklib.ITEM_DOCUMENT:
				chapters+=1
		chapter = []
		for i in range(0,chapters):
			chapter.append({'joy':0, 'fear':0, 'anger':0, 'surprise':0, 'sadness':0, 'disgust':0})
		data = pd.read_csv("emotions.csv",sep=',',delimiter=None)
		dict1=data.set_index('word')['emotion'].to_dict()
		i = 0
		flag=0
		word3count = {'joy':0, 'fear':0, 'anger':0, 'surprise':0, 'sadness':0, 'disgust':0}
		for item in book.get_items():
			if item.get_type() == ebooklib.ITEM_DOCUMENT:
				a = item.get_body_content()
				b = a.decode("utf-8")
				tokens = word_tokenize(b)
				words = [word for word in tokens if word.isalpha()]
				stop_words = stopwords.words('english')
				words = [w for w in words if not w in stop_words]
				myDict = {'joy':0, 'fear':0, 'anger':0, 'surprise':0, 'sadness':0, 'disgust':0}
				for j in range(len(words)): 
					words[j] = words[j].lower()
				for word in words:
					if flag == 1:
						flag=0
						continue
					elif word == 'not':
						flag=1
					elif word not in dict1.keys():
						continue
					else:
						chapter[i][dict1[word]] += 1
						myDict[dict1[word]] += 1
						word3count[dict1[word]] +=1
				i+=1
		myFig = 'myfig1.png'
		x = word3count.keys()
		y = word3count.values()
		plt.xlabel('Emotions')
		plt.ylabel('Count')
		plt.title('Dynamic Emotion Counts')
		plt.bar(x, y)
		fig1 = plt.gcf()
		plt.draw()
		full_filename = os.path.join(app.config['UPLOAD_FOLDER'], myFig)
		fig1.savefig(full_filename, dpi=100)
		fig1.clear()
		plt.close(fig1)
		
		df = pd.DataFrame(chapter)
		print(df)
		ax = plt.gca()
		df.plot(kind='line',y='joy',ax=ax)
		df.plot(kind='line',y='fear', color='green', ax=ax)
		df.plot(kind='line',y='anger', color='red', ax=ax)
		df.plot(kind='line',y='surprise', color='orange', ax=ax)
		df.plot(kind='line',y='sadness', color='cyan', ax=ax)
		df.plot(kind='line',y='disgust', color='magenta', ax=ax)
		plt.xlabel('Chapter')
		plt.ylabel('Count') 
		plt.title('Chapter wise emotions')
		fig2 = plt.gcf()
		plt.draw()
		full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'myFig2.png')
		fig2.savefig(full_filename2, dpi=100)
		figList = [full_filename, full_filename2]
		return render_template('ebook-output.html', i1=full_filename, i2=full_filename2)
	return 'File uploaded successfully, but analysis failed.'

#Main function of the Flask app
#------------------------------
if __name__ == '__main__':
	app.debug = True
	app.run(threaded=True, port=5000)