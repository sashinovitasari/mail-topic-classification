import pandas as pd
import file_handler as fh
import csv
from random import shuffle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake


wordnet_lemmatizer = WordNetLemmatizer()
symbol='\t!"#$%&\'()=-~^|\\`@{[+;*:}]<,>.?/_'
stop_w = fh.loadlist('stopword.txt',None)
wordlist=fh.loadlist('wordlist.txt',None)
r = Rake()

#---Header Handler----------
def extract_header_field(data,text_col=1):
	field = []

	for d in data:
		txt = d[text_col].split('\n')
		idx = 0
		while idx<len(txt) and ':' in txt[idx]:
			fi = txt[idx].split(':')[0].lower()
			if fi[0]!=' ' and fi[0]!='\t' and fi not in field:
				field.append(fi)
			idx+=1
	return field

def load_header_field(path):
	return fh.loadlist(path)

def verify_header_field(data,field,text_col=1):
	field_count = []
	for i in range(0,len(field)):
		field_count.append([field[i],0])
	
	for d in data:
		txt = d[text_col].split('\n')
		idx = 0
		while idx<len(txt) and ':' in txt[idx]:
			fi = txt[idx].split(':')[0].lower()

			if fi[0]!=' ' and fi[0]!='\t' and fi in field:
				field_count[ field.index(fi) ][1]+=1
			idx+=1
	for i in range(0,len(field_count)):
		field_count[i]+=[field_count[i][1]==len(data)]
	return field_count

#---Instance Handler----------
def separate_data_by_field(data,header,text_col=1,label_col=2,is_id=True):
	#last index - label
	data_final = []
	for d in data:
		header_data=['']*len(header)

		body 	= ''
		txt = d[text_col].split('\n')
		idx = 0
		while idx<len(txt) and ':' in txt[idx]:
			fi = txt[idx].split(':')[0].lower()
			if fi in header:
				header_data[header.index(fi)]=txt[idx].lower().replace(fi+': ','')
			idx+=1

		while idx<len(txt) and len(txt[idx])>0:
			idx+=1
		for i in range(idx,len(txt)):
			body+=txt[i]+'\n'
		body = remove_line_in_article(body)

		if is_id:
			data_final.append([d[0]]+[body]+header_data+[d[label_col]])
		else:
			data_final.append([body]+header_data+[d[label_col]])
	return (data_final)

def remove_line_in_article(text): 
	_text = [t for t in text.split('\n') if not('article' in t.lower() and '<' in t.lower() and '>' in t.lower())]
	text = ''
	for t in _text:
		text+= t+'\n'
	return text


import math
def generate_significant_word(data,target_col,label_col):
# The inverse document frequency is a measure of how much information the word provides, i.e., if it's common or rare across all documents
	labels = list(set([x[label_col] for x in data]))

	for l in labels:
		keys 	= []
	
		txt = [x[target_col].split(' ') for x in data if x[label_col]==l]
		flat_list=txt[0]
		for i in range(1,len(txt)):
			flat_list += txt[i]
		key = list(set(flat_list))

		while '' in key:
			key.remove('')
		while '' in flat_list:
			flat_list.remove('')

		for k in key:
			tf = flat_list.count(k)/len(flat_list)
			if tf>0.001:
				d = 1+len([x for x in txt if k in x])
				idf= math.log(len(txt))/d 
				keys.append([k,tf*idf])

		fh.savelist(keys,'tfidf_'+l+'.txt','\t')

def extfeat_majorword(data,target_col,wordlist):
	feature = []
	for d in data:
		_feat = []
		txt = d[target_col].split(' ')

		for w in wordlist:

			_feat.append(txt.count(w))
			
		feature.append(_feat)
	return feature

def combine_lists(list1,list2):
	return	([list1[i]+list2[i] for i in range(0,len(list1))])

#---Text Cleaner------------

#1. clean symbol
def clean_symbol(text):
	for s in symbol:
		text = text.replace(s,' ')
	while '  ' in text:
		text = text.replace('  ',' ')
	return text

def rem_symbol(data,target_col=1):
	for i in range(0,len(data)):
		data[i][target_col] = data[i][target_col].lower().replace('\n',' ')
		data[i][target_col] = clean_symbol(data[i][target_col])
	return data

#2. lemmatize
def lemmatize(data,target_col=1):
	for i in range(0,len(data)):
		token = data[i][target_col].lower().split(' ')
		token = ' '.join(wordnet_lemmatizer.lemmatize(t) for t in token)
		data[i][target_col] = token
	return data

#3. stop words removal
def rem_stopwords(data,target_col=1):
	for i in range(0,len(data)):
		token = data[i][target_col].lower().split(' ')

		for w in stop_w:
			while w in token:
				token.remove(w)

		data[i][target_col]=' '.join(token)
	return data

#---Data Handler----------------
def read_data(filepath):
	return pd.read_csv(filepath).values.tolist()

def filter_data_size(data,max_char,max_word):
	final_data = []
	for d in data:
		_str = d[1].replace('\n',' ')
		_str = clean_symbol(_str).lower()

		if len(_str)<max_char or len(_str.split(' '))<max_word:
			final_data.append([d[0],_str])
	return final_data

def split_data(data,output,partition=[0.75,0.25,0],header=['label','text'],index=False):
	idx_rand = [i for i in range(len(data))]
	shuffle(idx_rand)
	if index:
		header=['id']+header
	if partition[0]>0:
		_list = [ data[idx] for idx in idx_rand[:int(len(idx_rand)*partition[0]) ] ]

		df = []
		for i in _list:
			df.append({header[0]:i[0],header[1]:i[1]})
		df = pd.DataFrame(df)
		df.to_csv(output+'_train.csv',index=index)
		
	if partition[1]>0:
		_list = [ data[idx] for idx in idx_rand[ int(len(idx_rand)*partition[0]):int(len(idx_rand)*partition[0]+len(idx_rand)*partition[1])]]

		df = []
		for i in _list:
			df.append({header[0]:i[0],header[1]:i[1]})
		df = pd.DataFrame(df)
		df.to_csv(output+'_dev.csv',index=index)

	if partition[2]>0:
		_list = [ data[idx] for idx in idx_rand[int(len(idx_rand)*partition[0]+len(idx_rand)*partition[1]):]]
		df = []
		for i in _list:
			df.append({header[0]:i[0],header[1]:i[1]})
		df = pd.DataFrame(df)
		df.to_csv(output+'_test.csv',index=index)

l=read_data('data_naist_train_0.csv')

fi=load_header_field('header_field.txt')
l=separate_data_by_field(l,fi,text_col=1,label_col=2,is_id=True)

l= rem_symbol(l,target_col=1)#body
l= rem_symbol(l,target_col=2)#subject
l= rem_symbol(l,target_col=4)#keywords

l=rem_stopwords(l,target_col=1)#body
l=rem_stopwords(l,target_col=2)#subject
l=rem_stopwords(l,target_col=4)#keywords

'''
l=lemmatize(l,target_col=1)
l=lemmatize(l,target_col=2)
l=lemmatize(l,target_col=4)
'''


#l= generate_significant_word(l,1,-1)


l1=extfeat_majorword(l,1,wordlist)

label=[]
for i in range(0,len(l)):
	label.append([l[i][-1]])
l1=combine_lists(l1,label)
fh.savelist(l1,'feature.csv',',')
#l2=extfeat_majorword(l[-10:],1,wordlist)


#print(combine_lists(l1,l2))


#split_data(filter_data_size(read_data('data_naist_train.csv'),999,499),'data_naist_train',index=True)