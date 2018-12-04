'''
Author	: Sashi Novitasari
Date	: January 2018
Desc	: Handler for I/O file (save, load)
'''

#-----LOAD------

#load file to string
#Input	: file 		- filename
#Output	: string
def loadtxt(file):
	with open(file,"r",encoding='utf8') as myfile:
		return myfile.read()

#Load file to list (default separator: None)
#Input	: file 		- filename
#		  separator - separator of variables
#Output	: list
def loadlist(file,separator=None):
	list_ = []
	with open(file,"r",encoding='utf8') as myfile:
		rows = myfile.read().split("\n")
		for r in rows:
			if len(r)>0:
				if separator==None:
					list_.append(r)
				else:
					list_.append(r.split(separator))
	return list_

def loaddict(file,separator='\t'):
	dict_ = {}
	with open(file,"r",encoding='utf8') as myfile:
		rows = myfile.read().split("\n")
		for r in rows:
			if separator in r:
				item = r.split(separator)
				dict_[item[0]] = item[1]
	return dict_

#-----SAVE------

#Save string to file
#Input	: string 	- string to be saved
#		  file 		- filename
#Output	: file
def savetxt(string,file):
	with open(file,"w",encoding='utf8') as myfile:
		myfile.write(str(string))

#Save list to file (default separator:\t)
#Input	: list_ 	- list to be saved
#		  separator	- separator of variables
#Output	: file 

def savelist(list_,file,separator="\t"):
	myfile = open(file,'w',encoding='utf8')
	for row in list_:
		if separator!="":
			for col in range(0,len(row)-1):
				myfile.write(str(row[col])+separator)
			myfile.write(str(row[-1])+"\n")
		else:
			myfile.write(row+"\n")
	myfile.close()