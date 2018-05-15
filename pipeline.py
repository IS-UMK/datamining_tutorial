#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import nltk
import string
import scipy.sparse as sparse
import math
import numpy as np

data_dir = "/home/mich/zajecia/wdam/datamining_tutorial/data/pravda_extracted/"


tokenizer = nltk.word_tokenize
stemmer = nltk.PorterStemmer()
stoplist = set(nltk.corpus.stopwords.words())

def read_file(fn):
	f = open(fn)
	lines = f.readlines()
	title = lines[0]
	body = '\n'.join(lines[2:])
	f.close()
	return {'title': title, 'body': body}

def tokenize_document(doc):
    text = doc['title'] + ' ' + doc['body']
    text = text.decode("utf8")
    tokens = tokenizer(text.replace('"', '').lower())
    tokens = [w for w in tokens if not w in stoplist and not w in string.punctuation and not w.isdigit()]
    return map(stemmer.stem, tokens)

def read_pravda(start_date, end_date):
	f_dates = open(data_dir + "dates.txt")
	date2idx = f_dates.readlines()
	f_dates.close()
	date2idx = [datetime.datetime.strptime(d.strip(), "%Y-%m-%d") for d in date2idx]

	sd = datetime.datetime.strptime(start_date, "%Y-%m-%d")
	ed = datetime.datetime.strptime(end_date, "%Y-%m-%d")
	date_span = []
	while sd != ed:
		date_span.append(sd)
		sd = sd + datetime.timedelta(days=1)
	date_span = set(date_span)

	doc_idxs = [i for (i,d) in enumerate(date2idx) if d in date_span]

	return [read_file(data_dir+str(idx) +'.txt')  for idx in doc_idxs]

def construct_dictionary(tokenized_docs):
	f_dist = nltk.FreqDist()
	for doc in tokenized_docs:
		for token in doc:
			f_dist[token] += 1
	return f_dist

def construct_matrix(tokenized_docs, tokens):
    id2token = dict(enumerate(tokens))
    token2id = {v:k for (k,v) in id2token.iteritems()}
    row, col, data = [], [], []
    for d_id, doc in enumerate(tokenized_docs):
        print d_id, '/', len(tokenized_docs)
        tf, wid = {}, None
        for token in doc:
            if token in token2id:
                wid = token2id[token]
                tf[wid] = tf.get(wid, 0) + 1
        for wid, tf in tf.iteritems():
            row.append(d_id)
            col.append(wid)
            data.append(tf)
    M = sparse.coo_matrix( (data,(row,col)), shape=(len(tokenized_docs), len(token2id)))
    return M, {v:k for (k,v) in token2id.iteritems()}

def compute_tfidf(M):
	M = M.tocsr()
	idfs = {}
	for docNo, vec in enumerate(M):
		for termId, termCount in zip(vec.indices, vec.data):
			idfs[termId] = idfs.get(termId, 0) + 1

	num_docs = float(docNo + 1)

	#idf weight formula (simple)
	idfs = dict((termId, math.log(num_docs / docFreq, 2))
							for termId, docFreq in idfs.iteritems())

	M = M.tocoo()
	idf = tuple( map( lambda i:idfs.get(i, 0.0) ,  xrange(M.shape[1]) ))

	data = tuple( float(v) * idf[t]  for (t, v) in zip(M.col, M.data))
	return sparse.coo_matrix((data, (M.row, M.col)), dtype=float)

def pipeline(sd = '2013-11-01', ed ='2014-04-02'):
	min_tf = 10
	docs = read_pravda(sd, ed)
	tokenized_docs = map(tokenize_document, docs)
	f_dist = construct_dictionary(tokenized_docs)
	tokens = [token for (token, freq) in f_dist.items()[2:] if freq > min_tf]
	print len(tokens)
	M, id2word = construct_matrix(tokenized_docs, tokens)
	return M, id2word

def print_topics(u, id2word, num_topics=10, num_tokens=10):
	topic_vectors = u.T[0:num_topics]
	for vec in topic_vectors:
		most_contributing = np.abs(vec).argsort()[::-1]
		for token_idx in most_contributing[:num_tokens]:
			print str(round(vec[token_idx], 2)) + "*" +id2word[token_idx], '+',
		print "..."

M, id2word = pipeline()
