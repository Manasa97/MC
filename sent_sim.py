import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import math
import os
#from nltk.tag.stanford import StanfordPOSTagger

alpha = 0.2
phi = 0.2
beta = 0.45
delta = 0.85
eta = 0.4
vbi = 0.5

#other params = verb_match

class SentenceSimilarity:
	def __init__(self):
		'''
		self.stpos_path_to_model = os.environ.get('STANFORD_POS_CLASSPATH')
		self.stpos_jar = os.environ.get('STANFORD_POS_JAR')
		self.stpos = StanfordPOSTagger(self.stpos_path_to_model, self.stpos_jar)
		'''
		
	def similarity(self, s1, s2):
	    t1 = nltk.word_tokenize(s1)
	    t2 = nltk.word_tokenize(s2)
	    return (delta * self._semanticSimilarity(t1,t2)) + ((1.0 - delta) * self._wordOrderSimilarity(t1,t2)) #+ (vbi * 
	    	#self._addVerbImportance(s1, s2))

	def _wordOrderSimilarity(self, t1, t2):
	    jointWords = list(set(t1).union(set(t2)))
	    r1 = self._wordOrderVector(t1, jointWords)
	    r2 = self._wordOrderVector(t2, jointWords)
	    #r1 and r2 are vectors so subtract first and then computes the norm ~ sum of single values
	    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

	def _wordOrderVector(self, t1, jointWords):
	    #size of wov = size of jointWords
	    wov = np.zeros(len(jointWords))
	    t1index = {i[1]:i[0]+1 for i in enumerate(t1)}
	    t1words = set(t1)
	    for x in range(len(jointWords)):
	        if jointWords[x] in t1words:
	            wov[x] = t1index[jointWords[x]]
	        else:
	            #find the most similar word
	            simWord, maxSim = self._mostSimilarWord(jointWords[x], t1words)
	            if maxSim > eta:
	                wov[x] = t1index[simWord]
	    return wov

	def _mostSimilarWord(self, a, words):
	    maxSim = -1.0
	    simWord = ""
	    for i in words:
	      sim = self._wordSimilarity(a, i)
	      if sim > maxSim:
	          maxSim = sim
	          simWord = i
	    return simWord, maxSim

	def _wordSimilarity(self, w1,w2):
	    maxSynSim = -1.0
	    ss1 = wn.synsets(w1)
	    ss2 = wn.synsets(w2)
	    bestPair = None, None
	    if not (len(ss1) == 0 or len(ss2) == 0):
	        for x in ss1:
	            for y in ss2:
	                synSim = wn.path_similarity(x, y)
	                if synSim is not None and synSim > maxSynSim:
	                    maxSynSim = synSim
	                    bestPair = x, y
	                    #print('bestPair: ', bestPair)
	    return (self._lengthDistance(bestPair) * self._heightDistance(bestPair))
	def _lengthDistance(self, pair):
	    l = float('inf')
	    p1 = pair[0]
	    p2 = pair[1]
	    if p1 is None or p2 is None:
	        return 0.0
	    if p1 == p2:
	        l = 0.0
	    else:
	        x1 = set([str(a.name()) for a in p1.lemmas()])
	        x2 = set([str(a.name()) for a in p2.lemmas()])
	        if len(x1.intersection(x2)) > 0:
	            l = 1.0
	        else:
	            l = p1.shortest_path_distance(p2)
	            if l is None:
	                l = 0.0
	    #print(l)
	    return math.exp(-alpha * l)

	def _heightDistance(self, pair):
	    h = float('inf')
	    p1 = pair[0]
	    p2 = pair[1]
	    if p1 is None or p2 is None:
	        return h
	    if p1 == p2:
	        #both their depths are the same so return either
	        h = max([x[1] for x in p1.hypernym_distances()])
	        #op for p1.hy_dist() - {(Synset('entity.n.01'), 5), (Synset('physical_entity.n.01'), 4)
	        #max of each distance gives the depth of the synset
	    else:
	        #find the max depth of the common subsuming synset
	        h1 = {x[0]:x[1] for x in p1.hypernym_distances()}
	        h2 = {x[0]:x[1] for x in p2.hypernym_distances()}
	        commonSubsumers = set(h1.keys()).intersection(set(h2.keys()))
	        #print(commonSubsumers)
	        if len(commonSubsumers) > 0:
	            maxH = 0
	            for c in commonSubsumers:
	                d1 = 0
	                #if c in h1.keys():
	                if c in h1:
	                    d1 = h1[c]
	                d2 = 0
	                #if c in h2.keys():
	                if c in h2:
	                    d2 = h2[c]
	                d = max(d1,d2)
	                if d > maxH:
	                    maxH = d
	            h = maxH
	        else:
	            h = 0.0
	    #print(h)
	    return ((math.exp(beta*h) - math.exp(-beta*h))/(math.exp(beta*h)) + math.exp(-beta*h))

	def _semanticSimilarity(self,t1,t2):
	    jointWords = list(set(t1).union(set(t2)))#try removing list
	    #print(jointWords)
	    sv1 = self._getSemanticVector(t1,jointWords)
	    sv2 = self._getSemanticVector(t2,jointWords)
	    #print(sv1)
	    #print(sv2)
	    return np.dot(sv1, sv2.T) / (np.linalg.norm(sv1)*np.linalg.norm(sv2))#chekc accuracy without .T

	def _getSemanticVector(self, t1, jointWords):
	    t1words = set(t1)
	    sv = np.zeros(len(jointWords))
	    for x in range(len(jointWords)):
	        if jointWords[x] in t1words:
	            sv[x] = 1.0
	            sv[x] = sv[x] * math.pow(self._informationContent(jointWords[x]), 2)
	        else:
	            simWord, maxSim = self._mostSimilarWord(jointWords[x],t1words)
	            if maxSim > phi:
	                sv[x] = phi
	            else:
	                sv[x] = 0.0
	            sv[x] = sv[x] * self._informationContent(jointWords[x]) * self._informationContent(simWord)
	    return sv

	def _informationContent(self, word):
	    return 1

	'''

	def _addVerbImportance(self, s1, s2):
		stop_words = nltk.corpus.stopwords.words('english')
		stemmer = nltk.stem.PorterStemmer()
		tokenize = nltk.word_tokenize

		stemmeds1 = [stemmer.stem(word) for word in tokenize(s1) if not word in stop_words]
		stemmeds2 = [stemmer.stem(word) for word in tokenize(s2) if not word in stop_words]
		#taggeds1 = nltk.pos_tag(stemmeds1)
		#taggeds2 = nltk.pos_tag(stemmeds2)
		verb_match = 0
		taggeds1 = self.stpos.tag(stemmeds1)
		taggeds2 = self.stpos.tag(stemmeds2)

		verb_types = ['VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG']
		verbs1 = [x[0] for x in taggeds1 if x[1] in verb_types]
		verbs2 = [x[0] for x in taggeds2 if x[1] in verb_types]

		for v1 in verbs1:

			if v1 in verbs2:
				verb_match += 1
			else:
				for v2 in verbs2:
					if v1 in wn.synsets(v2):
						verb_match += 0.8


		return verb_match

		'''
