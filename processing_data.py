import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json

#GLOBAL VARIABLES
stop_words = stopwords.words('english')
cip_word_dictionary = {}
stem_designated_cip_codes = []

#READ IN STEM DESIGNATED DEGREES
stem_degrees = open("data/stem_designated.txt")
for degree in stem_degrees:
	string_parts = degree.split(" ")
	stem_designated_cip_codes.append(string_parts[1])

#LOAD DATASET
cip_codes = pd.read_csv('data/cip_codes.csv',
	usecols = ['CIPFamily','CIPCode','CIPTitle','CIPDefinition'])

#PROCESS TEXT
for x in range(len(cip_codes)):
	cip_codes.loc[x, 'CIPFamily'] = cip_codes.loc[x, 'CIPFamily'].replace('=','').replace('"','')
	cip_codes.loc[x, 'CIPCode'] = cip_codes.loc[x, 'CIPCode'].replace('=','').replace('"','')
	cip_codes.loc[x, 'CIPDefinition'] = cip_codes.loc[x, 'CIPDefinition'].lower()

#ADD STEM DESIGNATION
cip_codes['STEM_DESIGNATION'] = [0 for x in range(len(cip_codes))]
for x in range(len(cip_codes)):
	if cip_codes.loc[x,'CIPCode'] in stem_designated_cip_codes:
		cip_codes.loc[x,'STEM_DESIGNATION'] = 1
	if cip_codes.loc[x,'CIPFamily'] in ['14','26','27','40']:
		cip_codes.loc[x,'STEM_DESIGNATION'] = 1

#FILTER OUT CIP CODES NOT OF LENGTH 7
cip_codes = cip_codes[cip_codes['CIPCode'].str.len() >= 7]
cip_codes = cip_codes[~cip_codes['CIPDefinition'].str.contains(
	'This CIP code is not valid for IPEDS reporting'
)]

#TOKENIZE DESCRIPTIONS AND MAKE BAG OF WORDS
for index, row in cip_codes.iterrows():
	cip_description = row['CIPDefinition']
	word_tokens = word_tokenize(cip_description)
	for word in word_tokens:
		if word not in stop_words:
			try:
				cip_code_value = float(word)
			except:
				cip_word_dictionary[word] = 0

#MAKE ONE HOT ENCODING FOR EACH DESCRIPTION
headers = sorted(cip_word_dictionary)
rows = []
number_of_words = len(headers)

#CREATE ROW FOR EACH DESCRIPTION
for index, row in cip_codes.iterrows():
	cip_description = row['CIPDefinition']
	data_row = [0 for x in  range(number_of_words + 1)]
	for x in range(len(headers)):
		if headers[x] in cip_description:
			data_row[x] = 1

	data_row[number_of_words] = row['STEM_DESIGNATION']
	rows.append(data_row)

#CREATE PANDAS DATA FRAME
headers = headers + ['STEM_DESIGNATION']
df = pd.DataFrame(data = rows, columns = headers)

#SAVE DATA TO 
df.to_csv('data/examples.csv', index_label = "DOC_INDEX")
with open('data/data_dictionary.json','w') as file_to_write:
	json.dump(cip_word_dictionary, file_to_write, indent = 4)	

