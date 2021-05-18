import os
from tensorflow import keras
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

_current_dir = os.getcwd()

class MyTokenizer(object):

	def __init__(self, num_words=5000, separator=' '):
		self.num_words = num_words
		self.word_index = dict()
		self.separator = separator

	def tokenize_vocab(self, vocab):
		word_number = 1
		for word in vocab:
			self.word_index[word] = word_number
			word_number += 1 


	def fit_on_text_file(self, path_to_file, enc):
		word_number = 1
		with open(path_to_file, 'r', encoding=enc) as file:
			for line in file:
				for word in line.split():
					if not (word in self.word_index.keys()):
						self.word_index[word] = word_number
						word_number += 1


	def text_to_seq(self, text):
		seq = []
		for word in text.split(self.separator):
			if word in self.word_index.keys():
				seq.append(self.word_index[word])
			else:
				seq.append(0)

		return seq



class MonitorCallback(CallbackAny2Vec):
		def __init__(self):
			self.epoch = 0
		def on_epoch_begin(self, model):
			print("Epoch #{} start".format(self.epoch))

		def on_epoch_end(self, model):
			print("Epoch #{} end".format(self.epoch))
			self.epoch += 1

def load_word2vec_model(path=_current_dir):
	if path == _current_dir:
		model = Word2Vec.load(os.path.join(_current_dir, 'word2vec_model', 'word2vec.model'))
	else:
		model = Word2Vec.load(path)

	return model

def load_hyponym_model(path=_current_dir):
	if path == _current_dir:
		model = keras.models.load_model(os.path.join(_current_dir, 'hyponym_model'))
	else:
		model = keras.models.load_model(path)

	return model

def load_hypernym_model(path=_current_dir):
	if path == _current_dir:
		model = keras.models.load_model(os.path.join(_current_dir, 'hypernym_model'))
	else:
		model = keras.models.load_model(path)

	return model




def main():
	pass

if __name__ == '__main__':
	main()
