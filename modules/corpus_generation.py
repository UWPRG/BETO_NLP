import pybliometrics
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus.exception import Scopus429Error
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import multiprocessing
from os import system, name
import json
import time
from tqdm import trange
from IPython.display import clear_output
from pybliometrics.scopus import config
from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc, AbsDoc

class CorpusGenerator():

    def __init__(self, API_key, cache_path):
        """
        Parameters:
            API_key(str, required): API key generated from Scopus
            cache_path(str, required): path for dumping the cache data
        """
        self.API_list = list(API_key)
        self.cache_path = cache_path

    def make_jlist(self, jlist_url = 'https://www.elsevier.com/__data/promis_misc/sd-content/journals/jnlactivesubject.xls', \
                    journal_strings = ['chemistry','energy','molecular','atomic','chemical','biochem', \
                                       'organic','polymer','chemical engineering','biotech','colloid']):
        """
        This method creates a dataframe of relevant journals to query. The dataframe contains two columns:
        (1) The names of the Journals
        (2) The issns of the Journals

        As inputs, the URL for a journal list and a list of keyword strings to subselect the journals by is required.
        These values currently default to Elsevier's journals and some chemical keywords.
        """
        active_journals = pd.read_excel(jlist_url)
        # This makes the dataframe column names a smidge more intuitive.
        active_journals.rename(columns = {'Display Category Full Name':'Full_Category','Full Title':'Journal_Title'}, inplace = True)

        active_journals.Full_Category = active_journals.Full_Category.str.lower() # lowercase topics for searching
        active_journals = active_journals.drop_duplicates(subset = 'Journal_Title') # drop any duplicate journals
        active_journals = shuffle(active_journals, random_state = 42)

        # new dataframe full of only journals who's topic description contained the desired keywords
        active_journals = active_journals[active_journals['Full_Category'].str.contains('|'.join(journal_strings))]

        #Select down to only the title and the individual identification number called ISSN
        journal_frame = active_journals[['Journal_Title','ISSN']]
        #Remove things that have were present in multiple name searches.
        journal_frame = journal_frame.drop_duplicates(subset = 'Journal_Title')

        return journal_frame

    def build_search_terms(self, kwds):
        """
        This builds the keyword search portion of the query string.
        """
        combined_keywords = ""
        for i in range(len(kwds)):
            if i != len(kwds)-1:
                combined_keywords += kwds[i] + ' OR '
            else:
                combined_keywords += kwds[i] + ' '

        return combined_keywords

    def build_query_dict(self, term_list, issn_list, year_list):
        """
        This method takes the list of journals and creates a nested dictionary
        containing all accessible queries, in each year, for each journal,
        for a given keyword search on sciencedirect.

        Parameters
        ----------
        term_list(list, required): the list of search terms looked for in papers by the api.

        issn_list(list, required): the list of journal issn's to be queried. Can be created by getting the '.values'
        of a 'journal_list' dataframe that has been created from the 'make_jlist' method.

        year_list(list, required): the list of years which will be searched through

        """
        journal_frame = self.make_jlist(jlist_url = 'https://www.elsevier.com/__data/promis_misc/sd-content/journals/jnlactivesubject.xls', \
                        journal_strings = ['chemistry','energy','molecular','atomic','chemical','biochem', \
                                           'organic','polymer','chemical engineering','biotech','colloid'])

        search_terms = self.build_search_terms(term_list)
        dict1 = {}
        #This loop goes through and sets up a dictionary key with an ISSN number

        for issn in issn_list:

            issn_terms = ' AND ISSN(' + issn + ')'
            dict2 = {}
            #This loop goes and attaches all the years to the outer loop's key.
            for year in year_list:

                year_terms = "AND PUBYEAR IS " + str(year)
                querystring = search_terms + year_terms + issn_terms

                dict2[year] = querystring

            dict1[issn] = dict2

        return dict1

    def get_piis(self, term_list, year_list, pii_path, config_path='/Users/nisarg/.scopus/config.ini', keymaster=False):
        """
        This should be a standalone method that recieves a list of journals (issns), a keyword search,
        an output path and a path to clear the cache. It should be mappable to multiple parallel processes.
        """

        fresh_keys = self.API_list

        journal_frame = self.make_jlist(jlist_url = 'https://www.elsevier.com/__data/promis_misc/sd-content/journals/jnlactivesubject.xls', \
                        journal_strings = ['chemistry','energy','molecular','atomic','chemical','biochem', \
                                           'organic','polymer','chemical engineering','biotech','colloid'])


        if pii_path[-1] is not '/':
            raise Exception('Output file path must end with /')

        if '.scopus/scopus_search' not in self.cache_path:
            raise Exception('Cache path is not a sub-directory of the scopus_search. Make sure cache path is correct.')

        # Two lists who's values correspond to each other
        issn_list = journal_frame['ISSN'].values
        journal_list = journal_frame['Journal_Title'].values
        # Find and replaces slashes and spaces in names for file storage purposes
        for j in range(len(journal_list)):
            if ':' in journal_list[j]:
                journal_list[j] = journal_list[j].replace(':','')
            elif '/' in journal_list[j]:
                journal_list[j] = journal_list[j].replace('/','_')
            elif ' ' in journal_list[j]:
                journal_list[j] = journal_list[j].replace(' ','_')

        # Build the dictionary that can be used to sequentially query elsevier for different journals and years
        query_dict = self.build_query_dict(term_list,issn_list,year_list)

        # Must write to memory, clear cache, and clear a dictionary upon starting every new journal
        for i in range(len(issn_list)):
            # At the start of every year, clear the standard output screen
            os.system('cls' if os.name == 'nt' else 'clear')
            paper_counter = 0

            issn_dict = {}
            for j in range(len(year_list)):
                # for every year in every journal, query the keywords
                print(f'{journal_list[i]} in {year_list[j]}.')

                # Want the sole 'keymaster' process to handle 429 responses by swapping the key.
                if keymaster:
                    try:
                        query_results = ScopusSearch(verbose = True,query = query_dict[issn_list[i]][year_list[j]])
                    except Scopus429Error:
                        print('entered scopus 429 error loop... replacing key')
                        newkey = fresh_keys.pop(0)
                        config["Authentication"]["APIKey"] = newkey
                        time.sleep(5)
                        query_results = ScopusSearch(verbose = True,query = query_dict[issn_list[i]][year_list[j]])
                        print('key swap worked!!')
                # If this process isn't the keymaster, try a query.
                # If it excepts, wait a few seconds for keymaster to replace key and try again.
                else:
                    try:
                        query_results = ScopusSearch(verbose = True,query = query_dict[issn_list[i]][year_list[j]])
                    except Scopus429Error:
                        print('Non key master is sleeping for 15... ')
                        time.sleep(15)
                        query_results = ScopusSearch(verbose = True,query = query_dict[issn_list[i]][year_list[j]]) # at this point, the scopus 429 error should be fixed...
                        print('Non key master slept, query has now worked.')

                # store relevant information from the results into a dictionary pertaining to that query
                year_dict = {}
                if query_results.results is not None:
                    # some of the query results might be of type None


                    for k in range(len(query_results.results)):
                        paper_counter += 1

                        result_dict = {}
                        result = query_results.results[k]

                        result_dict['pii'] = result.pii
                        result_dict['doi'] = result.doi
                        result_dict['title'] = result.title
                        result_dict['num_authors'] = result.author_count
                        result_dict['authors'] = result.author_names
                        result_dict['description'] = result.description
                        result_dict['citation_count'] = result.citedby_count
                        result_dict['keywords'] = result.authkeywords

                        year_dict[k] = result_dict

                    # Store all of the results for this year in the dictionary containing to a certain journal
                    issn_dict[year_list[j]] = year_dict
                else:
                    # if it was a None type, we will just store the empty dictionary as json
                    issn_dict[year_list[j]] = year_dict


            # Store all of the results for this journal in a folder as json file
            os.mkdir(f'{pii_path}{journal_list[i]}')
            with open(f'{pii_path}{journal_list[i]}/{journal_list[i]}.json','w') as file:
                json.dump(issn_dict, file)

            with open(f'{pii_path}{journal_list[i]}/{journal_list[i]}.txt','w') as file2:
                file2.write(f'This file contains {paper_counter} publications.')

    def load_journal_json(self, absolute_path):
        """
        This method loads data collected on a single journal by the pybliometrics metadata collection module into a dictionary.

        Parameters
        ----------
        absolute_path(str, required) - The path to the .json file containing metadata procured by the pybliometrics module.
        """
        with open(absolute_path) as json_file:
            data = json.load(json_file)

        return data

    def make_dataframe(self, dataframe_path, pii_path):
        """
        This function takes the output path where the piis are stored and creates a dataframe for all the pii, doi, abstracts and other stuff
        Parameters:
            dataframe_path(str, required): the path to store the dataframe.
        """
        directory_list = os.listdir(pii_path)
        pub_year = []
        pii = []
        doi = []
        title = []
        authors = []
        num_authors = []
        abstract = []
        journal_name = []

        for i in trange(len(directory_list)):
            directory = directory_list[i]
            json_dict = self.load_journal_json(f'{pii_path}/{directory}/{directory}.json')

            for year in json_dict:
                for pub in json_dict[year]:
                    pub_year.append(year)
                    pii.append(json_dict[year][pub]['pii'])
                    doi.append(json_dict[year][pub]['doi'])
                    title.append(json_dict[year][pub]['title'])
                    authors.append(json_dict[year][pub]['authors'])
                    num_authors.append(json_dict[year][pub]['num_authors'])
                    abstract.append(json_dict[year][pub]['description'])
                    journal_name.append(directory)

        columns = ['pub_year', 'pii', 'doi', 'title', 'authors', 'num_authors', 'abstract', 'journal_name']
        df = pd.DataFrame(np.array([pub_year, pii, doi, title, authors, num_authors, abstract, journal_name], dtype=object).transpose(), columns=columns)
        df.to_pickle(dataframe_path + '/dataframe_from_CorpusGenerator' +'.pkl')

    def get_doc(self, dtype, identity):
        """
        This method retrieves a 'Doc' object from the Elsevier API. The doc object contains metadata and full-text information
        about a publication associated with a given PII.

        Parameters:
        -----------
        dtype(str,required): The type of identification string being used to access the document. (Almost always PII in our case.)

        identity: The actual identification string/ PII that will be used to query.
        """
        if dtype == 'pii':
            doc = FullDoc(sd_pii = identity)
        elif dtype == 'doi':
            doc= FullDoc(doi = identity)

        if doc.read(ElsClient(self.API_list[0])):
                #print ("doc.title: ", doc.title)
                doc.write()
        else:
            print ("Read document failed.")

        return doc

    def authorize(self, doc):
        #this method takes a doc object and returns a list of authors for the doc
        auths = []
        for auth in doc.data['coredata']['dc:creator']:
            auths.append(auth['$'])

        return auths

    def get_docdata(self, doc):
        """
        This method attempts to get certain pieces of metadata from an elsapy doc object.

        Parameters:
        -----------

        doc(elsapy object, required): elsapy doc object being searched for

        Returns:
        --------
        text(str): The full text from the original publciation.

        auths(list): The list of authors from the publication.
        """
        try:
            text = doc.data['originalText']                          # grab original full text
        except:
            text = 'no text in doc'

        try:
            auths = self.authorize(doc) # a list of authors
        except:
            auths = []

        return text, auths

    def get_fulltexts(self, pii_path, fulltext_output_path):
        """
        This method takes a list of directories containing 'meta' corpus information from the pybliometrics module and adds full-text information to those files.

        Parameters:
        ___________
        directory_list(list, required): A list of directories which this method will enter and add full-text information to.

        output_path(str, required): The folder in which the new full text corpus will be placed.

        api_keys(list, required): A list of valid API keys from Elsevier developer. One key needed per process being started.
        """
        #client = client
        directory_list = os.listdir(pii_path)


        for directory in directory_list:
            os.mkdir(f'{fulltext_output_path}/{directory}')
            marker = open(f'{fulltext_output_path}/{directory}/marker.txt','w') # put a file in the directory that lets us know we've been in that directory
            marker.close()

            info = open(f'{fulltext_output_path}/{directory}/info.csv','w') # a file to keep track of errors
            info.write('type,file,year,pub') # header

            #print(f'made marker and errors in {directory}')


            json_file = f'{pii_path}/{directory}/{directory}.json'
            j_dict = self.load_journal_json(json_file) # now we have a dictionary of information in our hands. Access it via journal_dict['year']['pub_number']
            rem_list = ['num_authors', 'description', 'citation_count', 'keywords']
            for year in j_dict:

                if j_dict[year] is not {}:
                    for pub in j_dict[year]:

                        pii = j_dict[year][pub]['pii'] # the pii identification number used to get the full text

                        try:

                            doc = self.get_doc('pii',pii) # don't know if doc retrieval will fail
                            print(f'Process {self.API_list} got doc for {directory}, {year}')
                        except Exception as e:
                            print(f'EXCEPTION: DOC RETRIEVAL. Process {self.API_list}')
                            print(f'Exception was {e}')
                            doc = None
                            info.write(f'doc retrieval,{json_file},{year},{pub}')

                        text, auths = self.get_docdata(doc) # doesn't crash even if doc = None


                        if text is 'no text in doc':
                            info.write(f'no text in doc,{json_file},{year},{pub}')
                        elif auths is []:
                            info.write(f'no auths in doc,{json_file},{year},{pub}')

                        j_dict[year][pub]['authors'] = auths
                        j_dict[year][pub]['fulltext'] = text # the real magic

                        for key in rem_list:
                            j_dict[year][pub].pop(key)

                else:
                    # the year was empty
                    info.write(f'year empty,{json_file},{year},{np.nan}')

            info.close()
            j_file = f'{fulltext_output_path}/{directory}/{directory}.json'

            with open(j_file,'w') as file:
                json.dump(j_dict,file)
