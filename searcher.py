import xapian
from xapian import SimpleStopper
import loadText as ld
import nltk
from nltk import *
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from nltk.stem import SnowballStemmer
import pickle
import pandas as pd
import math
import re
import itertools
import string
from random import randint
#nltk.download() to download all the tokenizers and other corpora
import networkx as nx
from itertools import chain


stopwords=open('stopwords.txt').readlines()
stopwords=[sword.strip() for sword in stopwords]
np_to_stem=pickle.load(open('np_to_stem.pkl', 'rb'))
stem_to_np=dict((v, k) for k, v in np_to_stem.iteritems())

############
## Search class
############
class Search:
    """ search class to response the query request."""  
    def __init__(self, queryString='', db_name='Psychology_abstracts'):
        global stopwords
        
        try:
            database= ld.databases[db_name]
        except:
            ld.getDatabases()
            database= ld.databases[db_name]
        
        
        ##this should go under loadtext in the future as it is db specific
        self.tf=pickle.load(open('tf.pkl', 'rb'))
        self.df=pickle.load(open('df.pkl', 'rb'))
        self.np_to_stem=pickle.load(open('np_to_stem.pkl', 'rb'))
        
        self.stem_to_np=defaultdict(list)
        for np, stem in self.np_to_stem.iteritems():
            self.stem_to_np[stem].append(np)
        
        
        
        noun_phrases=pickle.load(open('noun_phrases.pkl', 'rb'))
        self.noun_phrase_counts={}
        for k, v in noun_phrases.iteritems():
            self.noun_phrase_counts[str(k)]=Counter(v)
        
        
        noun_stems=pickle.load(open('noun_stems.pkl', 'rb'))
        self.noun_stem_counts={}
        for k, v in noun_stems.iteritems():
            self.noun_stem_counts[str(k)]=Counter(v)       
        
        
        
        
        
        self.database=database
        self.queryString=queryString
        
        
        db_doc=xapian.Database(self.database.xapian)    
        total_documents = db_doc.get_doccount()
        self.N=total_documents
        
        enquire = xapian.Enquire(db_doc)
        enquire.set_query(xapian.Query.MatchAll)
        matches = enquire.get_mset(0, total_documents)
        
        self.allDocs=[]
        for match in matches:
            # id=match.docid
            self.allDocs.append(self.database.get_doc(match.document.get_data()))
        
        stopper=SimpleStopper()

        for sword in stopwords:
            stopper.add(sword)
        self.stopper=stopper
        
        
    def xapian_search(self, k=100, showscore=True ):

        print self.database.xapian
        dbpath_doc = self.database.xapian
        db_doc=xapian.Database(dbpath_doc)
        doc_qp= xapian.QueryParser()
        doc_qp.set_stemmer(xapian.Stem("en"))

        doc_qp.set_stopper(self.stopper)
        doc_qp.set_database(db_doc)
        #doc_qp.set_default_op( xapian.Query.OP_ELITE_SET)
        doc_qp.set_default_op( xapian.Query.OP_AND)
        doc_qp.set_stemming_strategy(xapian.QueryParser.STEM_SOME)
        doc_query = doc_qp.parse_query(self.queryString)

        offset, limit = 0, min(k,db_doc.get_doccount() )
        print limit
        enquire = xapian.Enquire(db_doc)
        enquire.set_query(doc_query )
        doc_matches = enquire.get_mset(offset, limit)
        rset = xapian.RSet()
        
                    


        ids=[]
        scores=[]
        match_terms={}
        for match in doc_matches:
            rset.add_document(match.docid)
            
            document=match.document
            scores.append(match.weight)

            ids.append(document.get_data() )
            m_terms=enquire.matching_terms(match)
            match_terms[match.docid]=[term for term in m_terms] 
        
        self.ranked_ids=ids
        self.match_terms=match_terms
        alternatives=enquire.get_eset(100, rset, 0)
        # print alternatives
        self.alternatives={}
        for a in alternatives.items:
            self.alternatives[a[1]]=a[0]
        
            
        if showscore:
            return ids, scores

    ##later we can paginate the search results here
    def search_results(self ):
        docs=self.ranked_ids
        
        results=[]
        
        for doc in docs:
            tmp = self.database.get_doc(doc)
            results.append(tmp)
            
        return results


      





###########

##############
### commonly used
##############
def search(query=''):
    sr=Search(queryString=query)
    sr.xapian_search()
    return sr.search_results(), sr.alternatives, sr.match_terms, sr.allDocs, sr.ranked_ids, sr.tf, sr.df, sr.np_to_stem  


###############


###############
##1. Extracting noun pharases here 
###############
def extract_noun_phrases():
    st=SnowballStemmer('english')
    
    sr=Search(queryString='')
    allDocs=sr.allDocs
    
    ########
    ###Extracting noun phrase
    grammar = "NP: {<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar) 
    
    doc_nps={}
    doc_stems={}    
    for doc in allDocs:
        print doc['id'] 
        sentences=sent_tokenize(doc['text'])
        noun_phrases_trees=[]
        noun_phrases=[]
        noun_stems=[]
        
        for sentence in sentences:
            sentence = sentence.lower()
            tokens = nltk.pos_tag(word_tokenize(sentence))
            # tokens=[token.strip().lower() for token in tokens]
            result = cp.parse(tokens)
            list_of_noun_phrases2 = ExtractPhrases(result, 'NP')
            noun_phrases_trees.extend(list_of_noun_phrases2)
        
        for phrase in noun_phrases_trees:
            noun_phrases.append(' '.join([txt for txt, tag in phrase.leaves()]))
            noun_stems.append(' '.join([st.stem(txt) for txt, tag in phrase.leaves()]))
        
        doc_nps[doc['id']]=noun_phrases
        doc_stems[doc['id']]=noun_stems
        print noun_phrases
    
    pickle.dump(doc_nps, open('noun_phrases.pkl', 'w+'))  
    pickle.dump(doc_stems, open('noun_stems.pkl', 'w+'))   
    return doc_nps
    
  
def ExtractPhrases( myTree, phrase):
    myPhrases = []
    if (myTree.node == phrase):
        myPhrases.append( myTree.copy(True) )
    for child in myTree:
        if (type(child) is Tree):
            list_of_phrases = ExtractPhrases(child, phrase)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases

    




###############


###############
##2. Counting noun phrases
##(has an in-built tfidf calculation)
###############
def count_phrases():
    st=SnowballStemmer('english')
    #doc_nps=extract_noun_phrases()
    doc_nps=pickle.load(open('noun_phrases.pkl', 'rb'))
    doc_nps_stemmed={}
    np_to_stem={}
    for k, l_np in doc_nps.iteritems():
        l_np_stemmed=[]
        for np in l_np:
            tokens= word_tokenize(np)
            st_tokens=map(st.stem, tokens)
            st_np= " ".join(st_tokens)
            l_np_stemmed.append(st_np)
            np_to_stem[np]=st_np
        # noun_phrases_stemmed.append(st_np)
        doc_nps_stemmed[k]=l_np_stemmed
        
    doc_np_counts={}

    tf=Counter()
    idf=Counter()
    cntrs=map(lambda x: Counter(x[1]), doc_nps_stemmed.iteritems())
    i=0
    for cntr in cntrs:
        i += 1
        print i
        tf=tf + cntr
        for k in cntr.keys():
            idf[k] += 1
    pickle.dump(tf,open('tf.pkl','w+'))
    pickle.dump(idf, open('df.pkl','w+'))
    pickle.dump(np_to_stem, open('np_to_stem.pkl', 'w+'))
    
    file=open('np.csv', 'w+')
    file.write('NP,tf,df,tf/df^2\n')
    for k,v in tf.iteritems():
        tf=int(v)
        df=int(idf.get(k, '0'))
        tf_df_2=float(tf)/(df*df)
        file.write('%s,%s,%s,%.2f\n'%(k,tf,df,tf_df_2))
    file.close()
    
    file=open('np_to_stem.csv', 'w+')
    for  k,v in np_to_stem.iteritems():
        file.write('%s,%s\n'%(k,v))
    file.close()




###############


###############
##3. Find relevant (alternative) noun phrases for a given query
###############
def find_noun_phrases(query=''):
    global stopwords
    st=SnowballStemmer('english')
    sr=Search(queryString=query)
    sr.xapian_search()
    
    
    
    alt_word=[st.stem(word) for word in sr.alternatives.values() if word[0] is not 'Z' and word not in stopwords]
    alt_stem=[word[1:] for word in sr.alternatives.values() if word[0] is  'Z']
    alt_stem_tokens=alt_word+alt_stem
    alt_stem_tokens=list(set(map(unicode, alt_stem_tokens)))
    alt_stem_tokens
    
    
    tf=Counter()
    for id in sr.ranked_ids:
        doc_stems=sr.noun_stem_counts[id]        
        for phrase, count in doc_stems.iteritems():
         #   print phrase
            if any(token in phrase for token in alt_stem_tokens):
                tf[phrase] += count
          #      print phrase, tf[phrase]
               
    df=map(lambda x: sr.df[x], tf.keys())
    
    N=itertools.repeat(sr.N, len(tf))
    
    tf_idf= map(lambda x,y,z: calculate_tfidf(x, y,z), N, tf.values(), df)
    myframe=pd .DataFrame(zip(tf.values(), df, tf_idf) ,index=tf.keys())
    print(myframe)
    
    filename=makeSafeFilename('results_%s.csv'%(query))
    try:
        myframe.sort(2, ascending=False).head(15).to_csv('results/'+filename)
    except:
        pass


    
    
    return myframe.sort(2, ascending=False).head(15)
    
    # file=open('results_np.csv', 'w+')
    # file.write('%s,%s,%s,%s,%s\n'%('Noun Phrase','stem' ,'tf', 'df','tfidf' ))
    # for k, val in tf.iteritems():
    #      pr='~'.join(sr.stem_to_np[k])
    #      stem=k
    #      tf=val
    #      df=sr.df[k]
    #      tfidf= calculate_tfidf(sr.N, tf, df)
    #      file.write('%s,%s,%s,%s,%s\n'%(pr, stem,tf, df,tfidf ))
    # file.close()



def calculate_tfidf(N, tf, df):
    _tf=0
    if tf > 0:
        _tf=1+math.log(float(tf))
    
    _idf= math.log(float(N)/(1+df))
    
    return _tf*_idf
    



###############


##############
##4. Find pattern
#############
def find_pattern(concept1='', concept2=''):
    global np_to_stem
    
    stemmed_concept1=np_to_stem.get(concept1, concept1)
    results1=find_noun_phrases(concept1)
    G=nx.DiGraph()
    edges=zip(itertools.repeat(stemmed_concept1, len(results1.index)), results1.index)
    #print edges
    G.add_edges_from(edges)
    
    
    stemmed_concept2=np_to_stem.get(concept2, concept2)
    results2=find_noun_phrases(concept2)
    edges2=zip(results2.index, itertools.repeat( stemmed_concept2, len(results2.index)) )
    G.add_edges_from(edges2)
    
    processed=[]
    processed.append(stemmed_concept1)
    processed.append(stemmed_concept2)
    
    for key in results1.index:
        if key not in processed:
            print 'FROM 1', key
            processed.append(key)
            query=stem_to_np.get(key, key)
            print key, 'TO', query
            results_key=find_noun_phrases(query)
            edges=zip(itertools.repeat( key, len(results_key.index)), results_key.index)
            G.add_edges_from(edges)
    
    for key in results2.index:
        if key not in processed:
            print 'FROM 2', key
            processed.append(key)
            query=stem_to_np.get(key, key)
            print key, 'TO', query
            results_key=find_noun_phrases(query)
            edges=zip(results_key.index,itertools.repeat( key, len(results_key.index)))
            G.add_edges_from(edges)
    
      
    leaves=[node for node  in G.nodes() if G.degree(node) < 2 ] 
    G1=G.copy()
    G1.remove_nodes_from(leaves)
    G2=G1.to_undirected(reciprocal=False)
    paths=[]
    stemmed_paths=[]
    try:
        p1=nx.all_shortest_paths(G1,stemmed_concept1,stemmed_concept2)
        for path in p1:
            stemmed_paths.append(path)
            p=map(lambda x: stem_to_np.get(x,x), path)
            paths.append(p)
    except: 
        pass
    return G, G1, paths, stemmed_paths, stemmed_concept1, stemmed_concept2




##############

#############
##5. Draw pattern
#############
def draw_pattern(concept1='', concept2=''):
    #G: directed graph
    #G1: undirected graph
    ##test
    #G1=nx.dodecahedral_graph()
    #concept1=1
    #concept2=19
    #paths=nx.all_shortest_paths(G1, concept1,concept2)
    
    G, G1, paths,  stemmed_concept1, stemmed_concept2=find_pattern(concept1, concept2)
    
    
    eg=pd.DataFrame(G1.edges(), columns=['Source','Target'])
    nodes=G1.nodes()
    labels=map(lambda x: stem_to_np.get(x,x), nodes)
    
    shortest_path_nodes=list(chain(*stemmed_paths))
    group=map(lambda (x): int(x in shortest_path_nodes), nodes)
    nodes_g=zip(nodes,labels,group)
    nd=pd.DataFrame(nodes_g, columns=['Node','Label','Group'])
    
    filename=makeSafeFilename('path_%s_%s_nodes.csv'%(concept1,concept2) )
    nd.to_csv(filename)
    filename=makeSafeFilename('path_%s_%s.csv'%(concept1,concept2) )
    eg.to_csv(filename)
#############


############
##utilities
############
## Make a file name that only contains safe charaters
# @param inputFilename A filename containing illegal characters
# @return A filename containing only safe characters
def makeSafeFilename(inputFilename): 
    try:
        safechars = string.letters + string.digits + " -_."
        return filter(lambda c: c in safechars, inputFilename)
    except:
        return 'noname'+str(randint(0,1000))
    pass


