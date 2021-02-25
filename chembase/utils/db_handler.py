import datetime
import json
import os
import pymongo
from pymongo import MongoClient
from tqdm import tqdm


class MongoDBHandler():
    """
    This class is meant to facilitate file upload and download to and from the master corpus.
    Its uploading functions are to be called during corpus generation and after cleaning,
    extract, and model fuctions to update the corpus with the results of different processes.
    Because the processes act on some form of the documents in the corpus, the downloading
    functions are also neccessary in these processes.

    In order to use this class, you must be a registered user of the corpus's MongoDB cluster,
    with your IP address whitelisted
    """


    def __init__(self, user, password):
        """
        This function initializes the connection to the MongoDB server. It requires valid user
        credentials, which are added into the client string.

        ** Currently, this function connects to a practice cluster, database, and collection **
        ** The final version of this class will need to have these updated. **

        Parameters:
            user (str): A whitelisted username for the MongoDB cluster

            password (str): The corresponding password for the username
        """

        client = MongoClient('mongodb+srv://' + user + ':' + password + '@beto-corpora.cljbt.mongodb.net/Corpus?retryWrites=true&w=majority')

        self.cluster = client.beto_corpora
        self.corpus = self.cluster.Corpus

        return

    def return_client_corpus_object(self, ):
        """
        This function returns self.corpus for users to perform their own queries and updates
        that are beyond the functionality of this class.

        Parameters:

        Returns:
            self.corpus (pymongo.database.collection object): The pymongo object that corresponds
                to the corpus, which is the 'collection' in MongoDB verbiage.
        """

        return self.corpus


    def return_client_db_object(self, ):
        """
        This function returns self.db for users to perform their own queries and updates
        that are beyond the functionality of this class.

        Parameters:

        Returns:
            self.db (pymongo.databaseobject): The pymongo object that corresponds
                to the database object.
        """

        return self.db


    def upload_from_directory(self, directory):
        """
        In the case that a user has been saving documents locally, rather than directly
        uploading them to the MongoDB, this function allows a local directory to be listed
        and the contents uploaded to the MongoDB.

        ** Assumes all article files are named with their DOI and formatted as a JSON **

        Article structure should have at least a doi key:value pair, which is used to create
        the document's unique '_id'.

        ** If a document with the DOI _id already exists in the database, it is updated by
        this function. **

        Parameters:
            directory (str): Directory address for the folder containing article JSON files.
        """

        article_directory = directory

        art_count = len(os.listdir(article_directory))

        pbar = tqdm(total = art_count, position = 0)

        for article in os.listdir(article_directory):
            #article files are named by their DOI, which start with '10-'
            if '10-' in article:
                file = article_directory + article

                with open(file, 'r') as f:
                    art_json = json.load(f)
                    f.close()

                t = datetime.datetime.utcnow()
                timestamp = t.strftime('%Y %b %d %H:%M.%S')
                art_json['upload_timestamp'] = timestamp

                art_json['_id'] = art_json['doi']

                results_count = self.corpus.count_documents({'_id':art_json['_id']})

                #create new object with unique ID
                if results_count == 0:
                    self.corpus.insert_one(art_json)

                #update existing object
                else:
                    for k, v in art_json.items():
                        self.corpus.update_one({'_id': art_json['_id']},
                                          {'$set': {str(k):v}},
                                          upsert = True)

            elif 'search_progress' in article:
                file = article_directory + article

                with open(file, 'r') as f:
                    progress_json = json.load(f)
                    f.close()

                t = datetime.datetime.utcnow()
                timestamp = t.strftime('%Y %b %d %H:%M.%S')
                progress_json['upload_timestamp'] = timestamp

                self.corpus.insert_one(progress_json)

            else:
                pass

            pbar.update()

        return


    def upload_article_document(self, article_json):
        """
        A function that saves a scraped article and its metadata to the MongoDB.

        Parameters:
            article_json (dict): A dict that contains an article and its metadata
        """

        t = datetime.datetime.utcnow()
        timestamp = t.strftime('%Y %b %d %H:%M.%S')
        article_json['upload_timestamp'] = timestamp

        article_json['_id'] = article_json['doi']

        results_count = self.corpus.count_documents({'_id':article_json['_id']})

        #create new object with unique ID
        if results_count == 0:
            self.corpus.insert_one(article_json)

        #update existing object
        else:
            for k, v in article_json.items():
                self.corpus.update_one({'_id': article_json['_id']},
                                  {'$set': {str(k):v}},
                                  upsert = True)

        return


    def upload_progress_document(self, prog_json):
        """
        A function that uploads a document containing scraped htmls and their success status
        to the MongoDB.

        Parameters:
            prog_json (dict): A dict that contains lists of article https links
                and their success status
        """

        t = datetime.datetime.utcnow()
        timestamp = t.strftime('%Y %b %d %H:%M.%S')
        prog_json['upload_timestamp'] = timestamp

        results_count = self.corpus.count_documents({'title':prog_json['title']})

        #create new object with unique ID
        if results_count == 0:
            self.corpus.insert_one(prog_json)

        #update existing object
        else:
            for k, v in prog_json.items():
                self.corpus.update_one({'title': prog_json['title']},
                                       {'$set': {str(k):v}},
                                        upsert = True)

        return


    def retrieve_doc_by_doi(self, doi):
        """
        Queries the database for an existing document by its DOI. DOI must be formatted
        with dashes ('-') instead of dots('.') or slashes('/') (e.g. 10-1039-C5PY00733J)

        Parameters:
            doi (str): A DOI number associated with an article in question. Must be formatted
                as described above

        Returns:
            doc (dict): A JSON-structured object containing the article and its meta-
                data.
        """

        doc = list(self.corpus.find({'_id':doi}))

        if len(doc) == 0:
            return None

        if len(doc) == 1:
            return doc[0]

        if len(doc) > 1:
            print('MULTIPLE MATCHES FOUND')
            return doc[0]


    def retrieve_doc_by_title(self, title):
        """
        Queries the database for an existing document by its title.

        Parameters:
            title (str): A title associated with an article in question.

        Returns:
            doc (dict): A JSON-structured object containing the article and its meta-
                data.
        """

        doc = list(self.corpus.find({'title':title}))

        if len(doc) == 0:
            return None

        if len(doc) == 1:
            return doc[0]

        if len(doc) > 1:
            print('MULTIPLE MATCHES FOUND')
            return doc[0]


    def retrieve_non_article_documents(self, ):
        """
        Queries the database for an existing document that does *not* contain an article and,
        therefore, does not have a doi as its '_id'. This is useful for finding search_progress
        documents and other summary reports saved in the database.

        Parameters:


        Returns:
            docs (list): List of documents in the database without a DOI '_id'
        """

        #get a list of all DOI _id's
        doi_docs = list(self.corpus.find({'_id':
                                            {'$regex': '10-*'}},
                                         {'_id':1}))

        doc_ids = []

        #get all docs not in that list of DOI _id's
        for doc in doi_docs:
            doc_ids.append(doc['_id'])

        docs = list(self.corpus.find({'_id':
                                        {'$nin':doc_ids}
                                     }))

        return docs


    def retrieve_all_article_doi(self, ):
        """
        This function retrieves all ObjectID that correspond to journal articles. It uses a RegEx
        pattern match of '10-', which should be the first 3 characters of any article DOI. These
        _id's are returned as an iterable list.

        Parameters:

        Returns:
            doi_ids (list): List of strings containing the DOI of articles in the MongoDB. These DOI
                correspond to the unique ObjectID.
        """

        #get a list of all DOI _id's
        doi_fields = list(self.corpus.find({'_id':
                                                {'$regex': '10-*'}}, #regex match to DOI
                                           {'_id':1}))   #return only the _id field

        #iterate through field dicts and extract the values
        doi_ids = []
        for doi in doi_fields:
            doi_ids.append(doi['_id'])

        return doi_ids
