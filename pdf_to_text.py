import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
import os
import re
import numpy as np
import math
import pandas as pd

stopwords_eng = ['me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 
		'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
		'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
		'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
		'doing', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
		'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
		'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
		'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
		'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'll','re', 've', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
		'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
		'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
		'wouldn', "wouldn't", 'a']

def folder_reader(path, tag):
	files = os.listdir(path)
	files_pdf = [i for i in files if i.endswith('.'+tag)]
	return files_pdf

def only_text(all_data):
	all_data_txt = []
	for data in all_data:
		data_i = " ".join(re.findall(r"(?i)\b[a-z]+\b", data))
		data_i = " ".join([i for i in data_i.lower().split(" ") if i not in stopwords_eng])
		all_data_txt.append(data_i)
	return all_data_txt

def pdfparser(path, file):
	fp = open(path+'/'+file, 'rb')
	rsrcmgr = PDFResourceManager()
	retstr = io.StringIO()
	codec = 'ascii'
	laparams = LAParams()
	device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
	# Create a PDF interpreter object.
	interpreter = PDFPageInterpreter(rsrcmgr, device)
	# Process each page contained in the document.
	
	data_pg = []
	for page in PDFPage.get_pages(fp):
		interpreter.process_page(page)
		data =  retstr.getvalue()
		k = sum([len(i) for i in data_pg])
		data_pg.append(data[k:])
	return data_pg

def pdf_to_txt(path):
	files_pdf = folder_reader(path, 'pdf')
	for file in files_pdf:
		data = pdfparser(path, file)
		data = only_text(data)
		txt_path = path+'/texts/'
		for pg_no, pg in enumerate(data):
			write_pg = open(txt_path+file[:-4]+'_pg_'+str(pg_no+1)+'.txt', 'w')
			write_pg.write(pg)

def txt_loader(path):
	files_txt = folder_reader(path+'/texts/', 'txt')
	data_txt = []
	all_txt = ""
	for file in files_txt:
		read_pg = open(path+'/texts/'+file, 'r')
		data = read_pg.readlines()
		data = " ".join(data)
		data_txt.append(data)
		all_txt += data + ' '

	uniqueWords = list({i for i in all_txt.split(' ')})
	uniqueWords.sort()

	return data_txt, uniqueWords

def tf(data_txt, uniqueWords):
	tf_table = []
	for pg in data_txt:
		pg_data = []
		for uw in uniqueWords:
			pg_data.append(pg.count(uw))
		pg_data = np.array(pg_data)
		pg_data = pg_data/sum(pg_data)
		tf_table.append(pg_data)

	tf_table = np.array(tf_table)
	tf_table = pd.DataFrame(tf_table)
	tf_table.columns = uniqueWords
	tf_table.to_csv('tf_table.csv', index=False)

def idf(data_txt, uniqueWords):
	idf_table = []
	for uw in uniqueWords:
		df = 0
		for pg in data_txt:
			if uw in pg:
				df += 1
		idf = math.log(len(data_txt)/(df+1))
		if idf<0:
			idf = 0
		idf_table.append(idf)
	idf_table = np.array([idf_table])
	idf_table = pd.DataFrame(idf_table)
	idf_table.columns = uniqueWords
	idf_table.to_csv('idf_table.csv', index=False)

def tf_idf():
	idf_table = pd.read_csv('idf_table.csv')
	uniqueWords = idf_table.columns
	idf_table = idf_table.values
	tf_table = pd.read_csv('tf_table.csv')
	tf_table = tf_table.values

	tf_idf_table = []
	for tf in tf_table:
		tf_idf_table.append(tf * idf_table[0])
	tf_idf_table = np.array(tf_idf_table)
	
	return tf_idf_table, uniqueWords

def generator():
	path = './data'
	pdf_to_txt(path)
	data_txt, uniqueWords = txt_loader(path)
	tf(data_txt, uniqueWords)
	idf(data_txt, uniqueWords)

def apply_query():
	path = './data'
	tf_idf_table, uniqueWords = tf_idf()
	query = input("Enter your query:\t")
	query = " ".join(re.findall(r"(?i)\b[a-z]+\b", query))
	query = " ".join([i for i in query.lower().split(" ") if i not in stopwords_eng])
	txt = folder_reader(path+'/texts/', 'txt')
	N = len(txt)
	tf = []
	for uw in uniqueWords:
		tf.append(query.count(uw))
	tf = np.array(tf)/sum(tf)
	
	idf_table = pd.read_csv('idf_table.csv')
	uniqueWords = idf_table.columns
	idf_table = idf_table.values
	
	tf_idf_c = tf * idf_table[0]

	tf_idf_table, uniqueWords = tf_idf()

	ae = np.mean(np.absolute(tf_idf_table - tf_idf_c), axis=1)

	sorted_pgs = np.argsort(ae) + 1

	best_5 = sorted_pgs[:5]

	print(best_5)






if __name__ == '__main__':
	
	gen = 0#int(input("Do you want to generate the tf-idf vectors Enter 1 to generate 0 to not:\t"))
	if gen == 1:
		generator()
	apply_query()


# Find the area of the region enclosed by the parabola 𝑦 = 𝑥2 and the line 𝑦 = 𝑥 + 2