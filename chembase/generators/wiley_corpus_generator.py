from contextlib import contextmanager, redirect_stderr, redirect_stdout
from crossref.restful import Members
from crossref.restful import Works
import json
import numpy as np
import os
from os import devnull
import pandas as pd
import requests
import subprocess
import sys
import time
from tqdm import tqdm

module_path = os.path.abspath(os.path.join('../modules/scipdf_parser-master/scipdf/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import scipdf


class WileyCorpusGenerator():
    """
    This class uses the CrossRef API to search the Wiley publisher database for articles
    pertaining to a given set of search terms. To get a Wiley API token, users need to
    follow the steps outlined at the following link:
    https://olabout.wiley.com/WileyCDA/Section/id-829772.html
    
    In order to work, the user needs to either be on campus, where University intranet
    automatically grants access to publishers, or use the University's Big-IP Edge VPN.
    Information for this VPN is found at:
    https://www.lib.washington.edu/help/connect/husky-onnet
    
    Note: The CrossRef API is set to retire the clickthrough-client-cr service at the end
    of 2020. After retirement, users need to contact the University's librarians to arrange
    access to the CrossRef database for Text and Data Mining (TDM). Information on this is
    found at: https://www.crossref.org/blog/evolving-our-support-for-text-and-data-mining/
    """
    
    def __init__(self, clickthrough_token, user_info, email):
        """
        Parameters:
            clickthrough_token (str): User-specific token for accessing the CrossRef API
                clickthrough service. NOTE: Service to be retired at end of 2020. Class
                may need to be updated as a result
                
            user_info (str): User-specific string denoting the user and their email address.
                follows the format 'Firstname Lastname, mailto:example@gmail.com'
                
            email (str): User-specific string denoting the email address used in the user_info
                string. This is required because url's generated for clickthrough of CrossRef
                articles include the mailto address in them.
                
        Returns:
            
        """
        self.clickthrough = clickthrough_token
        self.agent = user_info
        self.mailto = email
        
        #keyword list to search through for each article
        self.keyword_list = ['chemistry','energy','molecular','atomic','chemical','biochem',
                             'organic','polymer','chemical engineering','biotech','colloid',
                             'corrosion', 'corrosion inhibitor', 'deposition', 'Schiff',
                             'inhibit', 'corrosive', 'resistance', 'protect', 'acid', 'base',
                             'coke', 'coking', 'anti-corrosion', 'layer', 'steel',
                             'mild steel', 'coating', 'degradation', 'oxidation', 'film',
                             'photo-corrosion', 'hydrolysis']
        
        
    def meta_data_search(self, search_terms, save_path, publisher_name = 'Wiley'):
        """
        A method to collect the metadata from crossref based on specific search terms passed in.
        The results are filtered down to a specific publisher. It then saves all the URLs as a
        single list. This function writes full_publist to a json file, which contains a list of 
        meta-data for the articles identified in the query.
        
        Parameters:
            search_terms (list of str): List of search terms used to identify articles to be added
                in the corpus
            
            save_path (str): Absolute or relative filepath leading to the directory where
                the fulltexts will be saved
                
            publisher_name (str): The name of the publisher that is to be queried for articles.
                This must correspond to a publisher within the CrossRef database.
                
        Returns:
            full_publist (pd.DataFrame): A dataframe that contains a list of articles and their
                DOI's and URL's
        """
        
        works = Works()
        members = Members()

        #query publisher for articles with given search terms
        chem_subset = works.query(search_terms)
        pub = next(iter(members.query(publisher_name)))
        pub_id = pub['id']
        chem_subset = chem_subset.filter(member = pub_id)
        doc_number = chem_subset.count()
        headers = {'CR-Clickthrough-Client-Token': self.clickthrough, 
                   "User-Agent": self.agent,
                   "Connection": 'close'}
        
        url = chem_subset.url + "&select=DOI,link&rows=1000&mailto=" + self.mailto + "&cursor="
        
        #starting cursor value. It will be updated in the loop with each request we make.
        cursor = '*'
        url_list = []
        doi_list = []
        saved_docs = 0

        #make the first request before entering the while loop
        response = requests.get(url+cursor, headers = headers).json()

        while len(response['message']['items']) > 0:
            
            #Add the total number of papers from the response to the saved docs list. 
            saved_docs += len(response['message']['items'])
            pcnt_comp = 100*saved_docs/doc_number
            print(f"{pcnt_comp:.3f}% complete")

            #for every entry in the response, loop through each entry. 
            for entries in response['message']['items']:
                
                #Check to see if the response item has a link in it
                keycheck = True
                try:
                    entries['link']
                except KeyError:
                    keycheck = False
                    
                if keycheck:
                    #If the link exists, then append article meta-data
                    URL = entries['link'][0]['URL']
                    DOI = entries['DOI']
                                        
                    #Update all http to https
                    if URL[:5] != "https" and URL[:4] == 'http':
                        URL = 'https' + URL[4:]
                    
                    #Check to see if the URL format is correct. If yes, add it, 
                    #otherwise you don't add it to the list.
                    if URL[8] == 'a':
                        url_list.append(URL)
                        doi_list.append(DOI)

            #Build a dataframe of URLs and DOIs from our requests. Save checkpoint    
            full_publist = pd.DataFrame()
            full_publist['URL'] = url_list
            full_publist['DOI'] = doi_list
            full_publist.to_json(save_path + 'wiley_meta_list.json')

            #update the cursor, and make a new request for a new response. 
            cursor = response['message']['next-cursor']
            cursor = cursor.replace("+", "%2B")
            response_nojson = requests.get(url+cursor, headers = headers)
    
            #check the status code from this response
            if response_nojson.status_code == 200:
                
                response = response_nojson.json()
                
            else:
                print("Status code is bad! The response code is: " + str(response_nojson.status_code))
                
                break

        #Build a dataframe from the full list of URLs and DOIs from our requests    
        full_publist['URL'] = url_list
        full_publist['DOI'] = doi_list

        #Save the final list after dropping duplicates
        full_publist.drop_duplicates()
        full_publist.to_json(save_path + 'wiley_meta_list.json')


        return full_publist
        
    
    def get_fulltexts(self, input_filepath, save_path, checkpoint = True,
                      save_pdf = False, save_figs = False):
        """
        This function uses the list of URL's and DOI's generated by self.meta_data_search()
        to access and save the corresponding fulltexts. This function writes a json file for
        each article url that is successful. For every unsuccesfully accessed or parsed article,
        the URL, DOI, and error code is saved in a json file in the same directory as the
        correctly parsed articles. If PDF's are being saved, they are added to a sub-directory
        within the save_path directory.
        
        Error codes
        -----------
         - 200: PDF retrieval successful, no subsequent errors
         - 299: Successful PDF retrieval, failure to parse into string-format
         - 403: Article not available, based on user's subscriptions
         - 404: Broken or incorrect URL
         - 429: Too many requests to the Wiley API
         - 500: Too quick of a request rate or server connection error
         - 503, 504: CrossRef API service is under too heavy of a load. Retry link later.
        
        Parameters:
            input_filepath (str): Absolute or relative filepath leading to the json file
                that contains all of the previously identified articles
                
            save_path (str): Absolute or relative filepath leading to the directory where
                the fulltexts will be saved
                
            checkpoint (bool): Value to indicate whether to look for existing query results
                or not. True will result in calling self.find_checkpoint.
                
            save_pdf (bool): Value to indicate whether or not to save the extracted pdf file
                along with the json file for each article that is being queried. Default is
                to not save the pdf, in order to conserve memory
                
            save_figs (bool): Variable dictating whether or not to extract figures as PNG
                and save them during PDF parsing
                
        Returns:
        
        """
        
        #Make sure to keep the connection - closed bit, otherwise you get rate limited really quickly.
        headers = {'CR-Clickthrough-Client-Token': self.clickthrough, 
                   "User-Agent": self.agent,
                   "Connection": 'close'}
        
        full_df = pd.read_json(input_filepath)
        url_list = full_df['URL'].tolist()
        doi_list = full_df['DOI'].tolist()
        
        if save_pdf == True:
            pdf_save_path = save_path+'pdfs/'
                    
            #check if pdf save path exists. If not, then create it.
            if os.path.isdir(pdf_save_path) == False:
                os.mkdir(pdf_save_path)
        
        if checkpoint == True:
            error_df = pd.read_json(save_path+'error_counts.json')
            start_index = self.find_checkpoint(save_path)
            print(f'Starting at index {start_index}')
            
        else:
            error_df = pd.DataFrame(columns = ['URL', 'DOI', 'Status_Code'])
            start_index = 0

        #Need a plus one to make sure it points to the first non-filled error spot.
        error_counter = len(error_df.index) + 1
        basic_counter = 0
        successful_paper_counter = 0
        
        #initialize connection to GROBID server
        bkgnd_process = self.start_grobid()
        
        #initialize tqdm progress bar
        pbar = tqdm(total = len(url_list[start_index:]), position = 0)

        #loop through the list of URLs (from last queried index)
        for i in range(len(url_list[start_index:])):
            url = url_list[i]
            doi = doi_list[i]
            #replace the doi's '/' with '_' so it doesn't mess with saving.
            doi_no_dash = doi.replace('/', '_')
            error_status = 0

            #Query the Wiley database. 
            file = requests.get(url = url, headers = headers, allow_redirects = True)
            
            #need sleep timer to prevent rate-based over-querying 
            time.sleep(14)

            #Check now to see if the request was successful.
            if file.status_code == 200:
                
                if save_pdf == False:
                    #Save a temporary file that holds the pdf
                    with open(save_path+'file.pdf', 'wb') as f:
                        f.write(file.content)
                        f.close()
                        
                else:   
                    #Save the file that holds the pdf
                    with open(pdf_save_path + doi_no_dash + '.pdf', 'wb') as f:
                        f.write(file.content)
                        f.close()
                
                try:
                    if save_pdf == False:
                        pdf_path = save_path + 'file.pdf'

                        if save_figs == False:
                            #Pull the text out of the pdf file. 
                            article_dict = self.parse_pdf(pdf_path)

                        if save_figs == True:
                            #Pull the text out of the pdf file. 
                            article_dict = self.parse_pdf(pdf_path, save_path, figures = True)

                    else:
                        pdf_path = pdf_save_path + doi_no_dash + '.pdf'

                        if save_figs == False:
                            #Pull the text out of the pdf file. 
                            article_dict = self.parse_pdf(pdf_path)

                        if save_figs == True:
                            #Pull the text out of the pdf file. 
                            article_dict = self.parse_pdf(pdf_path, save_path, figures = True)

                    with open(f'{save_path}/{doi_no_dash}.json', 'w') as f:
                        json.dump(article_dict, f)
                        f.close()

                except:
                    #check-point save bad articles
                    error_code = 299 #successful pdf retrieval, failed to convert to txt
                    error_df.loc[error_counter] = [url, doi, error_code]
                    error_df.to_json(save_path+'error_counts.json')
                    error_counter += 1
                    
                    
            #Check to ensure the response code is not a 429, which is a too many requests error
            elif file.status_code == 429:
                print("Error! We've made too many Wiley requests!")
                
                break
                
            #If 403 we can't get the article because it is not subscribed to by UW
            #If 404, something was broken in the URL path. 
            #If 500, it's because we went too quickly or had a problem with server connection.
            elif file.status_code == 403 or file.status_code == 404 or file.status_code == 500:                
                if file.status_code == 500:
                    time.sleep(8)
                    
                error_df.loc[error_counter] = [url, doi, file.status_code]
                error_df.to_json(save_path+'error_counts.json')
                error_counter += 1

            else:
                
                #check-point save bad articles
                error_df.loc[error_counter] = [url, doi, file.status_code]
                error_df.to_json(save_path+'error_counts.json')
                error_counter += 1
                
            #update tqdm progress bar
            pbar.update()
        
        #upon completion, terminate background connection to GROBID server
        self.terminate_grobid(bkgnd_process)
        
        #upon completion, print success and error rates
        self.evaluate_corp_gen(save_path, save_pdf = save_pdf)
                                
        return
    
    
    def find_checkpoint(self, fulltext_save_path):
        """
        This function is meant to find the current number of articles that have been
        queried by the class. This is calculated by looking at successful articles
        that are saved, and links that have been marked as bad or that produce errors.
        
        Parameters:
            fulltext_save_path (str): Absolute or relative path to the directory where
                the fulltext articles are being saved. This directory should also lead
                to the directory where error-producing urls are saved as a JSON by self.
                get_fulltexts()
                
        Returns:
            arts_queried (int): This number of articles that have been queried using the self.
                get_fulltexts function looping through a list produced by self.meta_data_search.
        """
            
        with open(fulltext_save_path + 'error_counts.json', 'r') as f:
            error_arts = json.load(f)
            f.close()
            
        paths = os.listdir(fulltext_save_path)
        
        #the substring '10.' corresponds to the DOI prefix at the start of each filename
        art_jsons = [x for x in paths if '10.' in x]
        
        arts_queried = len(error_arts['URL']) + len(art_jsons)
            
        return arts_queried
    
    
    def evaluate_corp_gen(self, fulltext_save_path, save_pdf = False):
        """
        This function looks at successfully saved files and the URL's that were flagged as
        causing errors to calculate various success rates of the scraping and rates of
        occurence for different errors
        
        Parameters:
            fulltext_save_path (str):A bsolute or relative path to the directory where
                the fulltext articles are being saved. This directory should also lead
                to the directory where error-producing urls are saved as a JSON by self.
                get_fulltexts()
                
            save_pdf (bool): Value to indicate whether or not the extracted pdf files were
                saved along with the json file for each article that was queried. Default
                is False.
                
        Returns:
            
        """
        
        total_urls = self.find_checkpoint(fulltext_save_path)
        
        art_json_paths = os.listdir(fulltext_save_path)
        art_jsons = [x for x in art_json_paths if '10.' in x]
        num_jsons = len(art_jsons)
        
        if save_pdf == True:
            pdf_paths = os.listdir(fulltext_save_path+'pdfs/')
            art_pdfs = [x for x in pdf_paths if '10.' in x]
            num_pdfs = len(art_pdfs)
            
        with open(fulltext_save_path+'error_counts.json', 'r') as f:
            error_json = json.load(f)

        err299 = 0
        err400 = 0
        err403 = 0
        err404 = 0
        err500 = 0
        err503 = 0
        err504 = 0

        for val in error_json['Status_Code'].values():
            if val == 299:
                err299+=1
            elif val == 400:
                err400+=1
            elif val == 403:
                err403+=1
            elif val == 404:
                err404+=1
            elif val == 500:
                err500+=1
            elif val == 503:
                err503+=1
            elif val == 504:
                err504+=1
                
        success_rate = (num_jsons/total_urls) * 100
        total_error_rate = (len(error_json['URL'])/total_urls) * 100
        err299_rate = (err299/total_urls) * 100
        err400_rate = (err400/total_urls) * 100
        err403_rate = (err403/total_urls) * 100
        err404_rate = (err404/total_urls) * 100
        err500_rate = (err500/total_urls) * 100
        err503_rate = (err503/total_urls) * 100
        err504_rate = (err504/total_urls) * 100
        
        print(f"Total URL's queried = {total_urls}")
        print(f'Successful scrape rate = {success_rate:.2f}%')
        
        if save_pdf == True:
            pdf_success_rate = (num_pdfs / total_urls) * 100
            
            print(f'Successful PDF scrape rate = {pdf_success_rate:.2f}%')
            
        print(f'Total error rate = {total_error_rate:.2f}%')
        print(f'error299 rate = {err299_rate:.2f}%')
        print(f'error400 rate = {err400_rate:.2f}%')
        print(f'error403 rate = {err403_rate:.2f}%')
        print(f'error404 rate = {err404_rate:.2f}%')
        print(f'error500 rate = {err500_rate:.2f}%')
        print(f'error503 rate = {err503_rate:.2f}%')
        print(f'error503 rate = {err504_rate:.2f}%')
        
        return
    
    
    def restitch_text(self, article_dict):
        """
        Function that takes a partitioned text and combines the dictionary values
        to reform the initial, unpartitioned full-text.

        Parameters:
            article_dict (dict): A dictionary containing the sectioned text and meta-data.
                Keys correspond to section headers as defined by `headers_list`. Values
                correspond to the text within that section of the full-text article

        Returns:
            restitched_text (str): Single string containing the full-text article
        """
        skip_keys = ['Acknowledgements', 'Reference']
        
        restitched_text = ''

        text_dict = article_dict['full text']
        
        for key, value in text_dict.items():
            if key in skip_keys:
                pass
            else:
                restitched_text += key
                restitched_text += value

        return restitched_text
    
    
    def get_keywords(self, article_dict):
        """
        Function that identifies which of the search terms are present in a given
        article.
        
        Parameters:
            text (str): fulltext article string
            
        Returns:
            keywords (list): List of strings of keywords present in a given article
        """
        
        keywords = []
        
        fulltext = self.restitch_text(article_dict)
        
        for word in self.keyword_list:
            if word in fulltext:
                keywords.append(word)
                
        return keywords
    
    
    def start_grobid(self):
        """
        This function initializes the background subprocess of serve_grobid.sh
        
        Parameters:
            None
                
        Returns:
            p (subprocess.Popen()): The already initialized Popen object that is
                the interface with the serve_grobid.sh script, which accesses the 
                GROBID API for scipdf-parser.
        """
        #get current work directory and that of the serve_grobid.sh file 
        work_dir = os.getcwd()
        scipdf_dir = '/Users/wesleytatum/Desktop/post_doc/BETO/BETO_NLP/ipynb/scipdf_parser-master/'
        
        #change directory and execute serve_grobid as background subprocess
        os.chdir(scipdf_dir)
        p = subprocess.Popen(['./serve_grobid.sh'], shell = True)
        
        #change back to current work directory
        os.chdir(work_dir)
        
        #return background subprocess object for future termination
        return p
    
    
    def terminate_grobid(self, p):
        """
        This function terminates the background subprocess of serve_grobid.sh
        
        Parameters:
            p (subprocess.Popen()): The already initialized Popen object that is
                the interface with the serve_grobid.sh script, which accesses the 
                GROBID API for scipdf-parser.
                
        Returns:
            None
        """
        
        #terminate background subprocess of serve_grobid.sh
        p.terminate()
        
        return
    
    
    def reformat_article_json(self, article_dict):
        """
        This function reformats the extracted article json to a format that is 
        consistent with the Elsevier corpus generator class.
        
        Parameters:
            article_dict (dict): dict produced by scipdf_parser.parse_pdf_to_dict,
                which contains the partitioned article text and other information
                about that text (e.g. figure info, references per section, etc.).
                
        Returns:
            reformatted_article (dict): dict that is consistent in format and
                content with those article dictionaries that are produced by the
                Elsevier corpus generator with prepartition == True.
        """
        
        reformatted_article = {}
        
        metadata = {'DOI' : article_dict['doi'],
                    'Title' : article_dict['title'],
                    'Figures' : article_dict['figures']}
        
        fulltext = {}
        headers_list = []
        
        for i in range(len(article_dict['sections'])):
            key = article_dict['sections'][i]['heading']
            headers_list.append(key)
            text = article_dict['sections'][i]['text']
            
            fulltext[key] = text
        
        fulltext['Reference'] = article_dict['references']
        
        reformatted_article['headers_list'] = headers_list
        reformatted_article[' Meta-data '] = metadata
        reformatted_article[' Abstract '] = article_dict['abstract']
        reformatted_article['full text'] = fulltext
        
        keywords = self.get_keywords(reformatted_article)
        article_dict['keywords'] = keywords
        
        return reformatted_article
    
    
    def parse_pdf(self, pdf_path, figure_path = '', figures = False):
        """
        This function uses the scipdf-parser library to convert a PDF of an article
        into a dictionary containing strings of the text, partitioned into its different
        sections, as dictated by the section headers. This function can also extract
        figures from the pdf and saves those in the specified folder as png.
        
        This function relies on the java library GROBID and therefore requires a java 
        development kit to be installed on the user's computer. A server connection to
        the GROBID API needs to be run as a background sub-process before and during PDF
        parsing using self.start_grobid(). After all parsing is complete, terminate the 
        GROBID server connection using self.terminate_grobid().
        
        Parameters:
            pdf_path (str): String containing the path to the PDF file that is to be
                converted to a dictionary.
                
            figure_path (str): String containing the path to the directory where the
                extracted files will be saved.
                
            figures (bool): Variable dictating whether or not to extract figures as PNG
                and save them during PDF parsing 
        
        Returns:
            reformatted_article (dict): Dict that is consistent in format and
                content with those article dictionaries that are produced by the
                Elsevier corpus generator with prepartition == True.
        """
        
        article_dict = scipdf.parse_pdf_to_dict(pdf_path)
        reformatted_article = self.reformat_article_json(article_dict)
        
        if figures == True:
            if os.path.isdir(figure_path + 'figures/') == False:
                os.mkdir(figure_path + 'figures/')
                
            #suppress completion stdout in scipdf.parse_figures    
            with suppress_stdout_stderr():
                scipdf.parse_figures(pdf_path, output_folder = figure_path + 'figures/')
                
        return reformatted_article
        
    
@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull. This is used
    to silence stdouts from scipdf-parser.
    """
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

