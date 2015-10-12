# coding: utf-8
from sqlalchemy import Table, Column,Integer,String,Float, Boolean, create_engine,MetaData,or_, and_ ,ForeignKey, func
from sqlalchemy.orm import mapper, sessionmaker, class_mapper
import os
import json
from gensim import corpora,models,similarities
from scipy  import stats, io, sparse
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

db_folder = 'databases'

global databases

class Document(object):
    pass

class Database:
    def __init__(self, db_folder, db_name):
        self.dictionary = None
        self.doc_files = None
        self.doc_refs = None
        self.ref_docs = None
        self.tfidf_docs = None
        self.db_folder = db_folder
        self.Session = None
        self.db_name = db_name
        self.folder = os.path.join(db_folder, db_name)
        self.xapian=os.path.join(self.folder, 'xapian')
        self.docs_table = None
        self.lang = None
        # self.log=lb.Log(self.folder)
        self.init_connection()
        
        
    def init_connection(self):
        engine = create_engine('sqlite:///' + os.path.join(self.folder, 'Documents.db'), echo=False)
        metadata = MetaData(bind=engine)
        self.Session = sessionmaker(bind=engine)
        self.load_matrices()
        
        self.docs_table = Table('documents', metadata, autoload=True)
        # try:
        #     self.log.init_connection()
        # except:
        #     print 'can\'t load log dbs '
            
        print 'loading database: ', self.db_name
       
        try:
            docmapper = class_mapper(Document, configure=True)
        except:
            docmapper = mapper(Document, self.docs_table)
        
        try:
            file = open(os.path.join(self.db_folder, self.db_name, "lang.txt"), "r")
            self.lang = file.read().replace("\n", "")
        except:
            self.lang = "no language set"
            
            
            

    def get_doc(self, doc_id):
        session=self.Session()
        # try:
        test_id=int(doc_id)
        # except:
        #             print 'test_id is not converted to integer'
        #         
        hasSite=True
        try:
             
            rs=session.query(Document.id, Document.title, Document.text, Document.keywords,Document.url,Document.published, Document.cited).filter(and_(Document.id == doc_id)) 
            tmp = rs.first()
        except:
             
            hasSite=False     
            rs=session.query(Document.id, Document.title, Document.text, Document.keywords,Document.url).filter(and_(Document.id == doc_id))
            tmp = rs.first() 

        dict={}
        try:

            dict["id"] = tmp.id
            dict["title"] = tmp.title
            dict["text"] = tmp.text
            # dict["text"] = 'tmp.text'
            dict["keywords"] = tmp.keywords
            dict["url"] = tmp.url
            dict["barvalue"] = []
            if hasSite:
                dict["published"]=tmp.published
                dict["cited"]=tmp.cited
            # 
        except:
              print str(doc_id) + " not found"
        return dict

    def load_matrices(self):
        self.dictionary= corpora.Dictionary.load(os.path.join(self.folder, 'documents.dict'))
        _coo=io.mmread(os.path.join(self.folder, 'documents.mm'))
        self.doc_files=sparse.csc_matrix(_coo)
        self.doc_refs=pickle.load(open(os.path.join(self.folder, 'documents.pkl'), 'r'))
        self.ref_docs={v:k for k,v in self.doc_refs.iteritems()}
        transformer = TfidfTransformer()
        self.tfidf_docs=transformer.fit_transform(self.doc_files)                
                
def getDatabases():
    global databases
    databases={}
    folders = []

    print next(os.walk(db_folder))
    dirname, dirnames, filenames = next(os.walk(db_folder))
    for subdirname in dirnames:
        print subdirname
        databases[subdirname] = Database(db_folder, subdirname)
