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
import natsort

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

    def get_fulltexts(self, pii_path, fulltext_output_path, pre_partition = True):
        """
        This method takes a list of directories containing 'meta' corpus information from the 
        pybliometrics module and adds full-text information to those files.

        Parameters:
        ___________
        directory_list(list, required): A list of directories which this method will enter and add 
            full-text information to.

        output_path(str, required): The folder in which the new full text corpus will be placed.

        api_keys(list, required): A list of valid API keys from Elsevier developer. One key needed 
            per process being started.
        
        pre_partition (bool): A variable denoting whether or not to separate full-texts into
            sections before writing to the save directory or to leave the full-text as a continuous
            string. If True (default), then section headers and keywords are used partition full-texts.
            If False, the full-text is left as one continuous string, including its meta-data, as
            returned by the Elsevier API.
        """
        #client = client
        directory_list = os.listdir(pii_path)


        for directory in directory_list:
            # put a file in the directory that lets us know we've been in that directory
            os.mkdir(f'{fulltext_output_path}/{directory}')
            marker = open(f'{fulltext_output_path}/{directory}/marker.txt','w')
            marker.close()

            # a file to keep track of errors
            info = open(f'{fulltext_output_path}/{directory}/info.csv','w')
            info.write('type,file,year,pub') # header

            #print(f'made marker and errors in {directory}')

            # now we have a dictionary of information in our hands. Access it via
            #journal_dict['year']['pub_number']
            json_file = f'{pii_path}/{directory}/{directory}.json'
            j_dict = self.load_journal_json(json_file)
            rem_list = ['num_authors', 'description', 'citation_count', 'keywords']
            for year in j_dict:

                if j_dict[year] is not {}:
                    for pub in j_dict[year]:

                        # the pii identification number used to get the full text
                        pii = j_dict[year][pub]['pii']

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
                        
                        # the real magic
                        if pre_partition == True:
                            partitioned_text = self.partition_fulltext(text)
                            j_dict[year][pub]['fulltext'] = partitioned_text
                            
                        if pre_partition == False:
                            j_dict[year][pub]['fulltext'] = text

                        for key in rem_list:
                            j_dict[year][pub].pop(key)

                else:
                    # the year was empty
                    info.write(f'year empty,{json_file},{year},{np.nan}')

            info.close()
            j_file = f'{fulltext_output_path}/{directory}/{directory}.json'

            with open(j_file,'w') as file:
                json.dump(j_dict,file)
                
                
    def partition_fulltext(self, fulltext):
        """
        This function calls the SciFullTextProcessor() class to separate strings that
        contain the article's full-text into a dictionary where the keys are section
        headers and the values are the text within that article section.
        
        Parameters:
            fulltext (str): An article full-text contained in a single-line string
            
        Returns:
            partitioned_text (dict): If the partition is successful, this dictionary
                contains keys pertaining to section headers and values pertaining to
                the text within that section. If the partition is unsuccessful, this
                dictionary contains a key:value pair with error codes and a key:value
                pair with the unpartitioned full-text.
        """        
                
        #Instantiate fulltext partitioner
        sfp = ftp.SciFullTextProcessor()
        
        try:
            #attempt to partition fulltext
            partitioned_text, errors = sfp.get_partitioned_full_text(fulltext)

        except:
            #if partition fails, return original fulltext and key:value pair for debugging
            partitioned_text = {'error handling fulltext':errors, 'fulltext':fulltext}
        
        return partitioned_text

                

class SciFullTextProcessor():
    """
    Class that applies a variety of cleaning and partitioning functions to
    a set of scientific full-texts.
    """
    
    def __init__(self, texts = None):
        """
        Parameters:
            texts (list, required): list of texts to be preprocessed
        """
        #pre-defined lists of section-headers. Longer headers are listed first (e.g.
        #'Methods and materials', with headers that may be contained in other headers
        #(e.g. 'Methods') being listed later. This allows post-matching filtration
        self.INTROS = [' Background ', ' Introduction ']

        self.METHODS = [' Methods and materials ', ' Materials and methods ', ' Experimental set-up' ,
                  ' Experimental methods ', ' Methods ', ' Materials ', ' Experimental ']

        self.RandD = [' Results and discussion ', ' Results ', ' Discussion ']

        self.CONCS = [' Conclusions and outlook ', ' Conclusions and future work ', ' Concluding Remarks',
                 ' Future perspectives ', ' Concluding remarks ', ' Perspectives ',
                 ' Conclusions ', ' Conlcusion ']

        self.ACKS = [' Acknowledgements ', ' Acknowledgments ']

        self.REFS = [' References', ' Reference', ' Bibliography ']
        
        if texts is None:
            self.texts = []
        elif isinstance(texts, str):
            self.texts = [texts]
        else:
            self.texts = texts
            
    def get_header_text(self, full_text):
        """
        For full-texts returned by the Elsevier, this function extracts the
        portion of the full-text string that contains the list of section
        headers
        
        Parameters:
            full_text (str): Single string containing the full-text article

        Returns:
            narrowed_string (str): A string from the meta-data section of the
                article that is meant to contain the list of headers used throughout
                the article. Depending on article formatting and keyword prevalence,
                this narrowed string is also used to determine the structure of the
                final partitioned article that is returned by this class.
        """
        
        end_header_list = full_text.index('Reference') + len('Reference')
        end_meta = full_text.index('Elsevier') + len('Elsevier')
        narrowed_string = full_text[end_meta:end_header_list]
        
        return narrowed_string


    def get_numbered_section_headers(self, full_text):
        """
        This function takes in a full-text article string and uses keywords/regex
        to identify and extract numbered section headings that are used in the
        article.

        Parameters:
            full_text (str): Single string containing the full-text article

        Returns:
            headers_list (list of str): A list containing the numbered section headers
                for the full-text
        """

        narrowed_string = self.get_header_text(full_text)

        #finds numbered headings for section titles
        number_pattern = '\s\d{1,2}\s' #No nesting
        nested_1_pattern = '\s\d{1,2}\.\d{1,2}\s' #1 level of nesting (e.g 1.1)
        nested_2_pattern = '\s\d{1,2}\.\d{1,2}\.?\d{1,2}\s' #2 levels of nesting (e.g 1.1.1)
        nested_3_pattern = '\s\d{1,2}\.\d{1,2}\.\d{1,2}\.?\d{1,2}\s' #3 levels of nesting (e.g 1.1.1.1)
        nested_4_pattern = '\s\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}\.?\d{1,2}\s' #4 levels of nesting
        nested_5_pattern = '\s\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}\.?\d{1,2}\s' #5 levels of nesting
        nums = re.findall(number_pattern, narrowed_string)
        nums.extend(re.findall(nested_1_pattern, narrowed_string))
        nums.extend(re.findall(nested_2_pattern, narrowed_string))
        nums.extend(re.findall(nested_3_pattern, narrowed_string)) 
        nums.extend(re.findall(nested_4_pattern, narrowed_string)) 
        nums = list(set(nums))

        #treat heading numbers like version numbers for sorting
        nums = natsort.natsorted(nums)

        #removes text before section headers
        header_index = narrowed_string.index(nums[0])
        headers = narrowed_string[header_index-1:]

        #gets indices to split header text into list
        index_list = []
        for i, num in enumerate(nums):
            index = headers.find(num)
            index_list.append(index)

        #remove nums and their indices if they correspond to numbers in text
        #rather than secion heading
        #have to execute twice b/c can have multiple headers containing numbers
        bad_nums = []
        for i in range(len(nums)):
            if i < len(nums)-1:
                if index_list[i] > index_list[i+1]:
                    bad_nums.append(index_list[i+1])

        for i in range(len(bad_nums)):
            bad = bad_nums[-(i+1)]
            del nums[index_list.index(bad)]
            index_list.remove(bad)

        bad_nums = []
        for i in range(len(nums)):
            if i < len(nums)-1:
                if index_list[i] > index_list[i+1]:
                    bad_nums.append(index_list[i+1])

        for i in range(len(bad_nums)):
            bad = bad_nums[-(i+1)]
            del nums[index_list.index(bad)]
            index_list.remove(bad)

        #uses gathered indices to split text into list
        header_list = []
        for i in range(len(nums)):

            #get all but final section text
            if i < len(nums)-1:
                header_list.append(headers[index_list[i]:index_list[i+1]])

            #need if/else logic for different conclusion titles
            else:
                last_entry = headers[index_list[i]:]
                satisfied = 0

                if 'Concluding statements' in last_entry:
                    satisfied = 1
                    conc_ind = last_entry.index('Concl') + len('Concluding statements')
                    finals = last_entry[conc_ind+1:].split(' ')
                    header_list.append(last_entry[:conc_ind])
                    header_list.extend(finals)

                if 'Concluding remarks ' in last_entry:
                    satisfied = 1
                    conc_ind = last_entry.index('Concl') + len('Concluding remarks ')

                    if 'and future aspects' in last_entry:
                        conc_ind = last_entry.index('aspects') + len('aspects ')

                    if 'and future perspectives' in last_entry:
                        conc_ind = last_entry.index('perspectives') + len('perspectives ')

                    finals = last_entry[conc_ind:].split(' ')
                    header_list.append(last_entry[:conc_ind])

                    if 'Conflict of interest statement' in last_entry:
                        conf_ind = last_entry.index('Conflict of')
                        conf_end = conf_ind + len('Conflict of interest statement ')
                        conf_hd = last_entry[conf_ind:conf_end]
                        header_list.append(conf_hd)
                        finals = last_entry[conf_end:].split(' ')

                    if ' Appendix ' in last_entry:
                        #finds and merges appendix numbers to 'Appendix'
                        merged_finals = []
                        for i in range(len(finals)):
                            if i < len(finals)-1:
                                if len(finals[i+1]) < 5:
                                    merge = finals[i] + ' ' + finals[i+1]
                                    merged_finals.append(merge)
                                else:
                                    pass
                            else:
                                 merged_finals.append(finals[i])

                        finals = merged_finals

                    header_list.extend(finals)

                if 'Summary and conclusion' in last_entry:
                    satisfied = 1
                    conc_ind = last_entry.index('Summary') + len('Summary and conclusion')
                    header_list.append(last_entry[:conc_ind])
                    finals = last_entry[conc_ind+1:].split(' ')
                    header_list.extend(finals)

                if 'Discussion and conclusions' in last_entry:
                    satisfied = 1
                    tmp_ind = last_entry.index('conclu') + len('conclusions ')
                    last_numbered_header = last_entry[:tmp_ind]
                    finals = last_entry[tmp_ind:].split(' ')
                    header_list.append(last_numbered_header)
                    header_list.extend(finals)

                if 'Discussion and conclusion' in last_entry:
                    if satisfied == 0:
                        satisfied = 1
                        tmp_ind = last_entry.index('conclu') + len('conclusion ')
                        last_numbered_header = last_entry[:tmp_ind]
                        finals = last_entry[tmp_ind:].split(' ')
                        header_list.append(last_numbered_header)
                        header_list.extend(finals)

                if 'Applications and conclusion' in last_entry:
                    satisfied = 1
                    tmp_ind = last_entry.index('conclusion') + len('conclusion ')
                    last_numbered_header = last_entry[:tmp_ind]
                    finals = last_entry[tmp_ind:]
                    header_list.append(last_numbered_header)
                    header_list.append(finals)


                if 'onclusion' not in last_entry:
                    if 'Perspectives' in last_entry:
                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Perspe') + len('Perspectives ')
                            header_list.append(last_entry[:tmp_ind])
                            finals = last_entry[tmp_ind:].split(' ')
                            header_list.extend(finals)

                    if 'Discussion' in last_entry:
                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Discu') + len('Discussion ')
                            header_list.append(last_entry[:tmp_ind])
                            finals = last_entry[tmp_ind:].split(' ')

                            #assumes appendix section comes after Discussion
                            if ' Appendix ' in last_entry:
                                #finds and merges appendix numbers to 'Appendix'
                                merged_finals = []
                                for i in range(len(finals)):
                                    if i < len(finals)-1:
                                        if len(finals[i+1]) < 5:
                                            merge = finals[i] + ' ' + finals[i+1]
                                            merged_finals.append(merge)
                                        else:
                                            pass
                                    else:
                                        merged_finals.append(finals[i])

                                finals = merged_finals

                            if 'Conflict of interest' in last_entry:
                                conf_ind = last_entry.index('Conflict of')
                                conf_end = conf_ind + len('Conflict of interest ')
                                conf_hd = last_entry[conf_ind:conf_end]
                                finals = last_entry[conf_end:].split(' ')
                                header_list.append(conf_hd)

                            header_list.extend(finals)

                    if 'Conflicts of interest statement' in last_entry:
                        if satisfied == 0:
                            satisfied = 1
                            conf_ind = last_entry.index('Conflicts of')
                            conf_end = conf_ind + len('Conflicts of interest statement ')
                            last_numbered_header = last_entry[:conf_ind]
                            conf_hd = last_entry[conf_ind:conf_end]
                            finals = last_entry[conf_end:].split(' ')
                            header_list.append(last_numbered_header)
                            header_list.append(conf_hd)
                            header_list.extend(finals)

                    if 'Acknowledgements' in last_entry:
                        #acks follow section header
                        if 'A' not in last_entry[:6]:
                            if satisfied == 0:
                                satisfied = 1
                                ack_ind = last_entry.index('Acknow')
                                last_numbered_header = last_entry[:ack_ind]
                                finals = last_entry[ack_ind:].split(' ')
                                header_list.append(last_numbered_header)
                                header_list.extend(finals)

                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Acknowledgement')
                            header_list.append(last_entry[:tmp_ind])
                            finals = last_entry[tmp_ind:].split(' ')
                            header_list.extend(finals)

                    #no second 'e'        
                    if 'Acknowledgment' in last_entry:
                        #acks follow section header
                        if 'A' not in last_entry[:6]:
                            if satisfied == 0:
                                satisfied = 1
                                ack_ind = last_entry.index('Acknow')
                                last_numbered_header = last_entry[:ack_ind]
                                finals = last_entry[ack_ind:].split(' ')

                                header_list.append(last_numbered_header)

                                #assumes appendix section comes after acks.
                                if ' Appendix ' in last_entry:

                                    #finds and merges appendix numbers to 'Appendix'
                                    merged_finals = []
                                    for i in range(len(finals)):
                                        if i < len(finals)-1:
                                            if len(finals[i+1]) < 5:
                                                merge = finals[i] + ' ' + finals[i+1]
                                                merged_finals.append(merge)
                                            else:
                                                pass
                                        else:
                                            merged_finals.append(finals[i])

                                    header_list.extend(merged_finals)

                                else:
                                    header_list.extend(finals)

                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Acknowledgment')
                            header_list.append(last_entry[:tmp_ind])
                            finals = last_entry[tmp_ind:].split(' ')
                            header_list.extend(finals)

                    if 'Acknowledgement' in last_entry:
                        #acks follow section header
                        if 'A' not in last_entry[:6]:
                            if satisfied == 0:
                                satisfied = 1
                                ack_ind = last_entry.index('Acknow')
                                last_numbered_header = last_entry[:ack_ind]
                                finals = last_entry[ack_ind:].split(' ')
                                header_list.append(last_numbered_header)
                                header_list.extend(finals)

                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Acknowledgement')
                            header_list.append(last_entry[:tmp_ind])
                            finals = last_entry[tmp_ind:].split(' ')
                            header_list.extend(finals)

                    if 'Reference' in last_entry:
                        if satisfied == 0:
                            satisfied = 1
                            tmp_ind = last_entry.index('Reference')
                            header_list.append(last_entry[:tmp_ind-1])
                            finals = last_entry[tmp_ind:]
                            header_list.append(finals)

                elif satisfied == 0: #header has contains 'Conclusion'

                    check_num = last_entry.index('Conclusion') - 2
                    #Conclusion is in header without a section number
                    if last_entry[check_num].isnumeric() == False:
                        satisfied = 1
                        last_numbered_header = last_entry[:check_num+2]
                        finals = last_entry[check_num+2:].split(' ')
                        header_list.append(last_numbered_header)
                        header_list.extend(finals)

                    #assumes that this header is numbered
                    if 'Conclusions and future outlook' in last_entry:
                        satisfied = 1
                        conc_ind = len(' Conclusions and future outlooks ')
                        conc_hd = last_entry[:conc_ind+2]
                        finals = last_entry[conc_ind+2:].split()
                        header_list.append(conc_hd)
                        header_list.extend(finals)

                    if ' and scope for future ' in last_entry:
                        satisfied = 1
                        conc_ind = len(' Conclusions and scope for future work ')
                        conc_hd = last_entry[:conc_ind+2]
                        finals = last_entry[conc_ind+2:].split()
                        header_list.append(conc_hd)
                        header_list.extend(finals)

                    if ' and prospect ' in last_entry:
                        satisfied = 1
                        conc_ind = len(' Conclusions and prospect ')
                        conc_hd = last_entry[:conc_ind+1]
                        finals = last_entry[conc_ind+1:].split()
                        header_list.append(conc_hd)
                        header_list.extend(finals)

                    if ' and outlook ' in last_entry:
                        satisfied = 1
                        conc_ind = len(' Conclusions and outlook ')
                        conc_hd = last_entry[:conc_ind+1]
                        finals = last_entry[conc_ind+1:].split()
                        header_list.append(conc_hd)
                        header_list.extend(finals)

                    #assumes appendix section comes after conclusion
                    if ' Appendix ' in last_entry:
                        satisfied = 1
                        conc_ind = len(' Conclusions')
                        conc_hd = last_entry[:conc_ind+2]
                        finals = last_entry[conc_ind+2:].split(' ')

                        #finds and merges appendix numbers to 'Appendix'
                        merged_finals = []
                        for i in range(len(finals)):
                            if i < len(finals)-1:
                                if len(finals[i+1]) < 4:
                                    merge = finals[i] + ' ' + finals[i+1]
                                    merged_finals.append(merge)
                                else:
                                    pass
                            else:
                                merged_finals.append(finals[i])

                        header_list.append(conc_hd)
                        header_list.extend(merged_finals)


                    #nested numbered conclusions section followed by acks. and refs
                    pattern = '\d*?\.\d+? Conclusions Ackno'
                    if len(re.findall(pattern, last_entry)) > 0:
                        if satisfied == 0:
                            satisfied = 1
                            conc_ind = last_entry.index('Conclusions') +len('Conclusions')
                            conc_hd = last_entry[:conc_ind+1]
                            finals = last_entry[conc_ind+1:].split(' ')
                            header_list.append(conc_hd)
                            header_list.extend(finals)

                    #Numbered conclusions section followed by acks. and refs
                    pattern = '[0-9] Conclusions Ackno'
                    if len(re.findall(pattern, last_entry)) > 0:
                        if satisfied == 0:
                            satisfied = 1
                            conc_ind = last_entry.index('Concl') + len(' Conclusions')
                            conc_hd = last_entry[:conc_ind]
                            finals = last_entry[conc_ind:].split(' ')
                            header_list.append(conc_hd)
                            header_list.extend(finals)

                    #Numbered conclusions section followed by acks. and refs
                    pattern = '[0-9] Conclusion Ackno'
                    if len(re.findall(pattern, last_entry)) > 0:
                        if satisfied == 0:
                            satisfied = 1
                            conc_ind = last_entry.index('Concl') + len(' Conclusion')
                            conc_hd = last_entry[:conc_ind]
                            finals = last_entry[conc_ind:].split(' ')
                            header_list.append(conc_hd)

                            if 'Conflict of interest' in last_entry:
                                conf_ind = last_entry.index('Conflict of')
                                conf_end = conf_ind + len('Conflict of interest ')
                                conf_hd = last_entry[conf_ind:conf_end]
                                finals = last_entry[conf_end:].split(' ')
                                header_list.append(conf_hd)

                            header_list.extend(finals)

                    #standard: # last header Conclusions Acknowledgements Reference
                    if satisfied == 0:

                        #assumes '# Conclusion Finan. supp. (?)Conf. of int. Ref.'
                        if 'Financial support' in last_entry:
                            satisfied = 1
                            fin_ind = last_entry.index('Financial')
                            fin_end = fin_ind + len('Financial support ')
                            last_numbered_header = last_entry[:fin_ind]
                            fin_hd = last_entry[fin_ind:fin_end]
                            header_list.append(last_numbered_header)
                            header_list.append(fin_hd)

                            finals = last_entry[fin_end].split(' ')

                            if 'Conflict of interest' in last_entry:
                                conf_end = fin_end + len('Conflict of interest ')
                                conf_hd = last_entry[fin_end:conf_end]
                                finals = last_entry[conf_end:]
                                header_list.append(conf_hd)
                                finals.split(' ')

                            if len(finals) < 4: #multiple elements
                                header_list.extend(finals)

                            else: #single element, len counts letters in that element
                                header_list.append(finals)

                        if 'Conflict of interest' in last_entry:
                            if satisfied == 0:
                                satisfied = 1
                                conf_ind = last_entry.index('Conflict of')
                                conf_end = conf_ind + len('Conflict of interest ')
                                conc_hd = last_entry[:conf_ind]
                                conf_hd = last_entry[conf_ind:conf_end]
                                header_list.append(conc_hd)
                                header_list.append(conf_hd)
                                finals = last_entry[conf_end:].split(' ')

                                if 'human rights' in last_entry:
                                    hr_ind = last_entry.index('Statement')
                                    hr_end = hr_ind + len('Statement of human rights ')
                                    hr_hd = last_entry[conf_end:hr_end]
                                    header_list.append(hr_hd)
                                    finals = last_entry[hr_end:].split(' ')

                                for i, elem in enumerate(finals):
                                    if elem == '':
                                        del finals[i]

                                header_list.extend(finals)

                        if 'Conflicts of interest' in last_entry:
                            if satisfied == 0:
                                satisfied = 1
                                conf_ind = last_entry.index('Conflicts of')
                                conf_end = conf_ind + len('Conflicts of interest ')
                                conc_hd = last_entry[:conf_ind]
                                conf_hd = last_entry[conf_ind:conf_end]
                                finals = last_entry[conf_end:].split(' ')
                                header_list.append(conc_hd)
                                header_list.append(conf_hd)
                                header_list.extend(finals)

                        # Conclusions and future trends Acks. Refs.
                        if 'and future trends' in last_entry:
                            satisfied = 1
                            ack_ind = last_entry.index('Acknow')
                            last_numbered_header = last_entry[:ack_ind]
                            finals = last_entry[ack_ind:].split(' ')
                            header_list.append(last_numbered_header)
                            header_list.extend(finals)

                        # Conclusions and future trends Acks. Refs.
                        if 'and future perspective' in last_entry:
                            satisfied = 1
                            ack_ind = last_entry.index('Acknow')
                            last_numbered_header = last_entry[:ack_ind]
                            finals = last_entry[ack_ind:].split(' ')
                            header_list.append(last_numbered_header)
                            header_list.extend(finals)

                        # Conclusions and perspectives Acks. Refs.
                        if 'and perspectives' in last_entry:
                            satisfied = 1
                            ack_ind = last_entry.index('Acknow')
                            last_numbered_header = last_entry[:ack_ind]
                            finals = last_entry[ack_ind:].split(' ')
                            header_list.append(last_numbered_header)
                            header_list.extend(finals)

                        # contains both Financial disclosure/acknowledgements
                        # and Ethical conduct of research
                        if 'Financial disclosure/' in last_entry:
                            satisfied = 1
                            fin_ind = last_entry.index('Financial')
                            fin_end = fin_ind + len('Financial disclosure/acknowledgements ')
                            conc_hd = last_entry[:fin_ind]
                            fin_hd = last_entry[fin_ind:fin_end]
                            ref_ind = last_entry.index('Reference')
                            eth_hd = last_entry[fin_end:ref_ind]
                            ref_hd = last_entry[ref_ind:]
                            header_list.append(conc_hd)
                            header_list.append(fin_hd)
                            header_list.append(eth_hd)
                            header_list.append(ref_hd)

                        else:
                            if satisfied == 0:
                                finals = last_entry.split(' ')
                                last_numbered_header = ' ' + finals[1] + ' ' + finals[2]
                                final_headers = [last_numbered_header]
                                final_headers.extend(finals[3:])
                                header_list.extend(final_headers)

        #pre-pend headers for text before article
        headers_list = [' Abstract ']

        headers_list.extend(header_list)

        return headers_list


    def get_nonnumbered_section_headers(self, full_text):
        """
        This function is called when a full text has section headers that are not
        numbered. Using pre-defined lists of section headers, it looks through the
        meta-data section containing headers to identify.
        
        Parameters:
            full_text (str): Single string containing the full-text article

        Returns:
            headers_list (list of str): A list containing the numbered section headers
                for the full-text
        """
        narrowed_string = self.get_header_text(full_text)

        headers_list = [' Abstract ']

        potential_headers = []
        for hd in self.INTROS:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass

    ###############

        potential_headers = []
        for hd in self.METHODS:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass    

    ###############

        potential_headers = []
        for hd in self.RandD:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass

    ###############

        potential_headers = []
        for hd in self.CONCS:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass 

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass

    ###############

        potential_headers = []
        for hd in self.ACKS:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass

    ###############

        potential_headers = []
        for hd in self.REFS:
            if hd in narrowed_string:
                potential_headers.append(hd)
            else:
                pass 

        if len(potential_headers) > 0:    
            headers_list.append(potential_headers[0])
        else:
            pass

        return headers_list


    def split_full_text(self, full_text, headers_list):
        """
        This function uses the output of get_section_headers to split a full-text
        string into its various subsections.

        Parameters:
            full_text (str): Single string containing the full-text article
            headers_list (list of str):  A list containing the numbered section
                headers for the full-text

        Returns:
            sectioned_text (dict): A dictionary containing the sectioned text. Keys
                correspond to section headers as defined by `headers_list`. Values
                correspond to the text within that section of the full-text article
            
        """

        sectioned_text = {}
        indices = {}
        no_abstr = False

        for i, hd in enumerate(headers_list):
            #need to replace special regex characters before matching substrings
            if '(' in hd:
                hd = hd.replace('(', '\(')

            if ')' in hd:
                hd = hd.replace(')', '\)')

            if '[' in hd:
                hd = hd.replace('[', '\[')

            if ']' in hd:
                hd = hd.replace(']', '\]')

            if '{' in hd:
                hd = hd.replace('{', '\{')

            if '}' in hd:
                hd = hd.replace('}', '\}')

            if '+' in hd:
                hd = hd.replace('+', '\+')

            if '*' in hd:
                hd = hd.replace('*', '\*')
                
            if ':' in hd:
                hd = hd.replace(':', '\:')

            if i == 0: # meta-data has no substring-matching to do
                
                inds = [m.start() for m in re.finditer(hd, full_text)]
                #Abstract can appear in text, but isn't listed w/ headers
                #Only use first instance
                if len(inds) > 0:
                    indices[hd] = inds[0]

                else: #if there is no abstract, use figures to remove meta-data
                    fig_text = [m.start() for m in re.finditer('Figure', full_text)]
                    indices[hd] = fig_text[0]
                    no_abstr = True

            else: 
                inds = [m.start() for m in re.finditer(hd, full_text)]
                #assume final instance of substring match corresponds
                #to the correct header text instance
                indices[hd] = inds[-1]


        for i, hd in enumerate(headers_list):

            if i == 0:
                if no_abstr == True:

                    #get meta-data, which has no keyword matching
                    sectioned_text['Section Headers'] = headers_list
                    end_ind = indices[' Abstract ']
                    sectioned_text['Meta-data'] = full_text[:end_ind]

                    #indicate there is no abstract
                    start_id = indices[' Abstract ']
                    end_id = indices[list(indices.keys())[1]]
                    sectioned_text[' Abstract '] = ''


                if no_abstr == False:
                    #get meta-data, which has no keyword matching
                    sectioned_text['Section Headers'] = headers_list
                    end_ind = indices[' Abstract ']
                    sectioned_text['Meta-data'] = full_text[:end_ind]

                    #get abstract
                    start_id = indices[' Abstract ']
                    end_id = indices[list(indices.keys())[1]]
                    sectioned_text[hd] = full_text[start_id : end_id]

            if i > 0 and i < len(headers_list)-1: #all setions but final section
                if i == 1:
                    if no_abstr == True:
                        start_id = indices[' Abstract ']
                        end_id = indices[list(indices.keys())[i+1]]
                        sectioned_text[hd] = full_text[start_id:end_id]
                        
                    else:
                        start_id = indices[list(indices.keys())[i]]
                        end_id = indices[list(indices.keys())[i+1]]
                        sectioned_text[hd] = full_text[start_id:end_id]
                        
                else:
                    start_id = indices[list(indices.keys())[i]]
                    end_id = indices[list(indices.keys())[i+1]]
                    sectioned_text[hd] = full_text[start_id:end_id]

            if i == len(headers_list) - 1: #final header
                start_id = indices[list(indices.keys())[i]]
                sectioned_text[hd] = full_text[start_id:]

        return sectioned_text


    def check_partition(self, sectioned_text, full_text):
        """
        Function that compares the length of the unpartitioned text to the length
        of the partitioned full text. This catches instances of bad header extraction
        that can still successfully fulfill all other functions. For example, if
        header numbers contain a typo (e.g. ' 4 Results and discussion', ' 4 
        Conclusions Acknowledgements References')

        Parameters:
            sectioned_text (dict): A dictionary containing the sectioned text. Keys
                correspond to section headers as defined by `headers_list`. Values
                correspond to the text within that section of the full-text article
                
            full_text (str): Single string containing the full-text article


        Returns
        ----------
            length_check (bool): If lengths are equivalent, length_check == True,
                else == False
        """
        
        restitched_text = ''
        for i, (key, value) in enumerate(sectioned_text.items()):
            if i > 0: #skip section headers value
                restitched_text += value

        length_check = (len(restitched_text) == len(full_text))

        return length_check

    def get_partitioned_full_text(self, full_text):
        """
        Wrapper function to partition full-text articles by the section headers
        listed in the article's metadata towards the beginning of the string.
        Calls get_section_headers and split_full_text

        Parameters:
            full_text (str):  Single string containing the full-text article

        Returns:
            sectioned_text (dict): A dictionary containing the sectioned text. Keys
                correspond to section headers as defined by `headers_list`. Values
                correspond to the text within that section of the full-text article
            
        """
        error1 = 0
        error2 = 0

        if full_text != '': #ensure that text string contains article

            try:
                #narrows string down to meta-info segment containing primarily section headers
                narrowed_string = self.get_header_text(full_text)
                
                if len(narrowed_string) > 2500:
                    #no section headers. narrowed string gets full article
                    nums = [-2]

                else:
                    #check for header numbers
                    number_pattern = '\s\d{1,2}\s' #No nesting
                    nums = re.findall(number_pattern, narrowed_string)

            except:
                pass
                
            if len(nums) > 1: #if there are numbered section headers
                headers_list = self.get_numbered_section_headers(full_text)
                sectioned_text = self.split_full_text(full_text, headers_list)
                
            elif nums == [-2]:
                headers_list = ['no section headers']
                sectioned_text = {'section_headers': headers_list, 'full text': full_text}

            else:
                header_list = self.get_nonnumbered_section_headers(full_text)
                sectioned_text = self.split_full_text(full_text, header_list)

        else:
            error1 = 1
            sectioned_text = {'article' : 'there is no text for this article'}

        if self.check_partition(sectioned_text, full_text) == False:
            error2 = 1
    #         sectioned_text = {'article': 'error in extracting headers'}

        return sectioned_text, [error1, error2]
    
    
    def partition_full_text_list(self, texts = 'default', success_only = True, show_bad_ids = True):
        """
        Wrapper function that is called to parse a list of texts
        
        Parameters:
            texts (list of str): list of texts to tokenize. if 'default', then
                self.texts will be used
            success_only (bool): If True, only the successfully parsed texts are
                returned. Otherwise, all sectioned_full_texts will contain entries
                for all attempted texts, where values pertain to error messages
            show_bad_ids (bool): If True, 
                
        Returns:
            sectioned_full_texts (dict): dictionary containing the partitioned full-
                texts. If success_only == False, then failed texts will contain
                error messages, rather than the partitioned text
            
        """
        
        if texts == 'default':
            texts = self.texts
            
        
        sectioned_full_texts = {}
        error_count = []
        header_trouble_ids = []
        empty_article_ids = []
        undf_err_ids = []

        for i in tqdm(range(len(texts))):
            try:
                sectioned_full_texts[i], errors = self.get_partitioned_full_text(texts[i])
                error_count.append(errors)

            except:
                sectioned_full_texts[i] = 'error'
                error_count.append([0, 0])

        empty_articles = 0
        bad_lengths = 0
        undefined_error = 0

        for i, (key, value) in enumerate(sectioned_full_texts.items()):
            if value == 'error':
                undefined_error +=1
                undf_err_ids.append(i)

            if error_count[i][0] == 1:
                empty_articles+=1
                empty_article_ids.append(i)

            if error_count[i][1] == 1:
                bad_lengths +=1
                header_trouble_ids.append(i)

        bad_article_ids = header_trouble_ids
        bad_article_ids.extend(undf_err_ids)
        bad_article_ids = list(set(bad_article_ids))
        bad_article_ids.sort()

        print(f'{empty_articles} full texts were empty')
        print(f'{len(bad_article_ids)} full texts had text partitioning errors')
        
        if show_bad_ids == True:
            print(bad_article_ids)
        
        if success_only == True:

            for ind in bad_article_ids:
                try:
                    del sectioned_full_texts[ind]
                except:
                    print(f'key {ind} does not exist')

        return sectioned_full_texts