import urllib.request as urllib2
import os
import requests
#import wget
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
#import deepdish as dd
import json
import xml.etree.ElementTree
import librosa
import pretty_midi
import numpy as np
import sys
import random
#reload(sys)
#sys.setdefaultencoding('utf-8')

# parse the url from webpage
def parse_url():
	for i in range(0,1000):
		try:
			page = i + 1
			page = urllib2.urlopen('https://musescore.com/sheetmusic?page='+str(page))
			parsed_html = BeautifulSoup(page)
			for a in parsed_html.find_all('article', attrs={'role':'article'}):
				myfile = open("list.txt", "a")
				url = a.find('h2').find('a', href=True)['href']
				myfile.write(url+'\n')
				print ("Found the URL:" + url)
				myfile.close()
		except Exception as e:
			print (e)

# parse metadata
def search_score_oj():
	key = 'apply from musescore'
	secret = 'apply from musescore'
	s_path = 'list.txt' #1
	d_path = 'xml/'

	lines = [line.rstrip('\n') for line in open(s_path)]
	for i,f in enumerate(lines[:]):
		_id = f.split('/')[-1]
		if not (os.path.isfile(d_path+_id+'.xml')):
			try:
				url = 'http://api.musescore.com/services/rest/score/'+_id+'.xml?oauth_consumer_key='+key
				xml_str = urllib2.urlopen(url)
				parsed_xml= BeautifulSoup(xml_str)
				open(d_path+_id+'.xml', "w").write(parsed_xml.prettify())  
			except Exception as e:
				print (e)
		else: print ('exist')


#file download
def download_file():
	#destination path
	mp3_d_path = 'data/mp3/'
	mid_d_path = 'data/mid/'
	#source path
	files = os.listdir('xml/')
	for i,f in enumerate(files[:]):
		_id = f.split('.')[0]
		if (not (os.path.isfile(mp3_d_path+_id+'.mp3'))) or (not (os.path.isfile(mid_d_path+_id+'.mid'))):
			try:
				xmloj = open('xml/'+f, "r").read()
				secret = BeautifulSoup(xmloj).find('secret').text.rstrip().strip() 
				#file extension can be (pdf, mid, mxl, mscz, mp3)
				mp3_s_path = 'http://static.musescore.com/'+_id+'/'+secret+'/score.mp3'
				mid_s_path = 'http://static.musescore.com/'+_id+'/'+secret+'/score.mid'
				mp3ifile = urllib2.urlopen(mp3_s_path)
				open(mp3_d_path+_id+'.mp3','wb').write(mp3ifile.read())
				midifile = urllib2.urlopen(mid_s_path)
				open(mid_d_path+_id+'.mid','wb').write(midifile.read())
				print (str(i)+'/'+_id)
			except Exception as e:
				print (e)
		else: print ('exist')
