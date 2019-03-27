import codecs

# This is where the MegaDictionary is stored
path_to_dict = 'C:\\Users\\bscuser\\Downloads\\MegaDicionario.txt'

with codecs.open(path_to_dict, 'r', utf-8') as f:
	items = f.readlines()
	
# Remove \n in the end of concepts
items_cleaned = []
for item in items:
	items_cleaned.append(item.replace('\n',''))
	
# Remove original dictionary	
del(items)

sentences = ['bla bla bla 1 Samamicina bla', 'El paciente fue tratado con zidovudina', 'presenta glucosa alta']

# Now for each sentence, find all the concepts present in the dictionary
# I am organizing things as a list of dictionaries with two positions: Sentence and tags
# This can be modified so that it suits you better

Mapped_Sentences = []
for sentence in sentences:
	found = []
	for item in items_cleaned:
		# Check if concept is inside the sentence
		if item in sentence:
			found.append(item)

	Mapped_Sentences.append(dict(sentence=sentence,mapped=found))

			