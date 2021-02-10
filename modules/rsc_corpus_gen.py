import bs4
from bs4 import BeautifulSoup
import json
import mongo_db_interface as mongo
import os
import requests
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import sys
import time


class RscCorpusGenerator():
    """
    This class uses Selenium to web-crawl an RSC search results page and come to a full
    article HTML page. BeautifulSoup is used to scrape and parse through the HTML elements
    to identify meta-data and article components. The resulting data are saved for later
    parsing.
    
    Because web-crawling is used, all web-traffic needs to be directed through a VPN to
    ensure that institutional subscriptions are utilized, giving access to the article
    full-text. Information for the BIG-IP Edge VPN used by the University of Washington
    is found at:
    
    https://www.lib.washington.edu/help/connect/husky-onnet
    
    Before web-crawling, the user needs to notify the RSC of their intent to webcrawl, by
    doing the following:
    
    1. Notifying the RSC
    At least two (2) weeks before you wish to carry out the TDM you need to notify the RSC via
    ejournals@rsc.org that you wish to carry out the TDM and give us the following information.

    Date to start:
    Completion date:
    Institution:
    Crawler IP address:
    Crawler user agent: TDMCrawler (please set user agent to this)
    Types of content (HTML / PDF)
    Institution contact email:
    Researcher contact email:

    2.  Carrying out the TDM
    When carrying out the TDM please follow the guidelines given below:

        · Keep delays to 10-20 seconds between requests.

        · Set the user agent to TDMCrawler, adding contact and project information.

    3.  Notify your librarian
    Please inform your librarian that you are carrying out this activity and at the end
    of the process tell them the number of articles you have downloaded so that they can
    exclude these records when they are analysing the RSC usage statistics.
    """
    
    def __init__(self, url, driver_path, contact_email, project):
        """
        Selenium uses a webdriver to control your browser and access webpages and their elements.
        This class assumes you are using the browser Chrome and that you have downloaded the
        appropriate webdriver exec file. If this file is not saved in PATH, then supply it using
        the `driver_path` parameter.
        
        Parameters:
            url (str): A string specifying the starting point URL for web-crawling with Selenium.
                This should be a search results page with the search terms and filters already
                entered and applied.
                
            driver_path (str): A string specifying the directory to the chromedriver exec file.
                If file is saved to PATH, this parameter is unnecessary.
                
            contact_email (str): A string specifying the email address of the webcrawler user agent.
            
            project (str): A string specifying the project for which the scraping is being done.
        """
        try:
            self.driver_path = driver_path
        except:
            self.driver_path = ''
            
        self.search_page_url = url
        
        #add headers with RSC **required** information
        self.opts = Options()
        self.opts.add_argument('user-agent=[TDMCrawler]')
        self.opts.add_argument(f'mailto=[{contact_email}]')
        self.opts.add_argument(f'project=[{project}]')
        
        
    def navigate_search_results(self, user = '', password = '', save_path = '', continue_search = False, local = False):
        """
        This function pulls up the user-specified URL that corresponds to the RSC search results
        page. This should have the search terms already entered and any necessary filters applied.
        After navigating to an article's webpage, this function calls the self.get_article_html()
        function to extract and save the article components.
        
        Parameters:
            user (str): A whitelisted username for the MongoDB cluster
            
            password (str): The corresponding password for the MongoDB username
            
            save_path (str): A string representing a directory to which the progress reports and
                successfully scraped articles are saved.
                
            continue_search (bool): A bool specifying whether to start from beginning of search
                results, or to find a checkpoint and continue from there.
                
            local (bool): Denotes whether results and progress are to be save locally in the
                `save_path` directory, or remotely in the MongoDB cluster.
        """
        
        self.save_path = save_path
        
        if local == False:
            self.db_handler = mongo.MongoDBHandler(user, password)
        
        #get webdriver and set user-agent to 'TDMCrawler' in accordance with RSC guidelines        
        driver = webdriver.Chrome(self.driver_path, options = self.opts)
        
        #wait as much as 5 seconds for webpages to load before proceeding
        driver.implicitly_wait(5)
        
        #go to search results page
        driver.get(self.search_page_url)
        
        last_page = False
        landing_page = False
        page_num = 0
        article_count = 0
        progress_json = {'page links':{},
                         'page statuses':{},
                         'title': 'rsc_search_progress'}
        
        if continue_search == True:
            if local == True:
                progress_json = self.load_local_progress_json()
                
            if local == False:
                progress_json = self.load_remote_progress_json()
                
            start_place = self.find_checkpoint(progress_json)
            
            article_count = start_place[1]
            start_article = article_count % 25
                        
            #if last search ended at the end of a page (each page containing 25 articles)
            if start_article == 0:
                page_num = start_place[0]+1
                
            else:
                page_num = start_place[0]
            
            #need to denote if you have navigated away from starting page
            landing_page = True
            
            #navigate to correct starting page of results
            for i in range(1, page_num):
                time.sleep(10)
                driver.find_element_by_class_name('paging__btn--next').click()                
        
        #get the source HTML of current page and parse into bs4 object
        source = driver.page_source
        soup = BeautifulSoup(source, 'html.parser')
        
        #loop through until the last page is reached
        while last_page == False:
            
            if continue_search == True and landing_page == True:
                
                #if first time through loop of a continued search
                page_links = progress_json['page links'][str(page_num)]
                page_https = page_links[start_article:]
                
                successes = progress_json['page statuses'][str(page_num)]
                
                landing_page = False
                
            else:
                page_num+=1

                #for the given page, get and store all article links
                page_https = self.get_page_article_links(soup)
                progress_json['page links'][page_num] = page_https
                
                successes = []
                
            for page in page_https:
                
                if page == 'pdf only':
                    print('pdf only')
                    successes.append('pdf only')
                    article_count+=1
                    
                else:
                    #open article in new tab and switch driver to it
                    driver.execute_script("window.open('"+page+"','newTab');")
                    driver.switch_to.window(driver.window_handles[-1])

                    #follow the link and scrape contents
                    page_source = driver.page_source
                    article_json = self.get_article_html(page_source)

                    #save a JSON containing results and success code
                    if local == True:
                        self.save_article_locally(article_json)
                    else:
                        self.save_article_remotely(article_json)
                    article_count+=1
                    successes.append(article_count)

                    #save progress so far
                    progress_json[f'page statuses'][page_num] = successes
                    if local == True:
                        self.save_progress_locally(progress_json)
                    else:
                        self.save_progress_remotely(progress_json)

                    #close the newly opened tab and switch driver to results page
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

                    #delay between requests in accordance with RSC Guidelines
                    time.sleep(10)
                                                            
            #if on the last page of search results, exit loop. Else, go to next page
            try:
                driver.find_element_by_class_name('paging__btn--next').click()
                time.sleep(10)
                source = driver.page_source
                soup = BeautifulSoup(source, 'html.parser') 
                
            except:
                print('beginning of except')
                last_page = True
                
        driver.quit()
                
        return
        
        
    def get_page_article_links(self, soup):
        """
        This function is called by self.navigate_search_results to get the href hyperlinks
        to the articles listed on the current results page of an RSC advanced journal search.
        
        Parameters:
            soup (bs4.BeautifulSoup): A BeautifulSoup object containing the HTML elements of
                the article search results webpage.
            
        Returns:
            article_links (list): A list that contains all of the https links to the articles
                listed on the search results page in string format. Default length is 25 articles
                per page.
        """
        #find portion containing current page's article URL's
        article_panel_list = soup.find_all('div', {'class':'tab-content'})
        article_parent = article_panel_list[1]
        article_containers = article_parent.find_all('div', {'class':'capsule capsule--article'})

        article_links = []

        #for each article container on the page, get the url of the article HTML
        for i, art in enumerate(article_containers):

            try:
                button_element = art.find('a', {'class':'btn btn--tiny'})
                element_children = button_element.children

                for child in element_children:
                    #Sometimes there is a 'Download References' button for PDF only articles
                    if child.string == 'Article HTML':

                        href_hyperlink = button_element.get('href')

                        base_url = 'https://pubs.rsc.org'
                        full_url = base_url+href_hyperlink
                        article_links.append(full_url)

                    else:
                        article_links.append('pdf only')

            #if there is no "Article HTML" button, there is only a 'Download PDF' button
            except:
                article_links.append('pdf only')
            
        return article_links
        
        
    def get_article_html(self, source):
        """
        This function is called by self.navigate_search_results() to extract the contents of
        a single article webpage. The contents are saved in a JSON file that contains entries
        for doi, title, abstract, and source HTML.
        
        Parameters:
            source (str): A string that contains the HTML components of the article's webpage
            
        Returns:
            article (dict): A dict that contains the article metadata and HTML components
        """
        
        article = {}
                        
        soup = BeautifulSoup(source, 'html.parser')
        
        title = soup.title.text
        
        doi_start = title.find('DOI:') + 4
        doi = title[doi_start:]
        doi = doi.replace('/', '-')
        doi = doi.replace('.', '-')
        
        try:
            abstract = soup.find('p', {'class':'abstract'}).text
        except:
            abstract = 'no abstract'
        
        article['doi'] = doi
        article['title'] = title
        article['abstract'] = abstract
        article['html'] = source
                
        return  article
    
    
    def save_progress_locally(self, progress_json):
        """
        A function that saves the scraped htmls and their success status to the
        save path defined in self.navigate_results_page().
        
        Parameters:
            progress_json (dict): A dict that contains lists of article https links
                and their success status
        """
        
        with open(self.save_path+'search_progress.json', 'w') as f:
            json.dump(progress_json, f)
            f.close()
            
        return
    
    
    def save_progress_remotely(self, progress_json):
        """
        A function that saves the scraped htmls and their success status to the
        MongoDB defined in self.db_handler().
        
        Parameters:
            progress_json (dict): A dict that contains lists of article https links
                and their success status
        """
        
        self.db_handler.upload_progress_document(progress_json)
            
        return
    
    
    def load_local_progress_json(self, ):
        """
        Returns:
            progress_json (dict): The saved article progress reporter
        """
        
        with open(self.save_path+'search_progress.json', 'r') as f:
            progress_json = json.load(f)
            f.close()
            
        return progress_json
    
    
    def load_remote_progress_json(self, ):
        """
        Returns:
            progress_json (dict): The saved article progress reporter
        """
        
        progress_json = self.db_handler.retrieve_article_by_title('rsc_search_progress')
            
        return progress_json
    
    
    def find_checkpoint(self, progress_json):
        """
        This function is called if continue_search == True in self.navigate_search_results.
        The article progress saved in the JSON file at self.save_path is used to identify
        where the previous search left off.
        
        Parameters:
            progress_json (dict): The saved article progress reporter
        
        Returns:
            start_place (tuple): A tuple containing two values - the page number and the
                article number on that page that was last successfully scraped in the
                previous search.
        """
        
        page_num = len(progress_json['page links'])
        
        last_page_articles = len(progress_json['page statuses'][str(page_num)])
        
        last_article = (25 * (page_num-1)) + last_page_articles - 1
        
        start_place = (page_num, last_article)
        
        return start_place
    
    
    def save_article_locally(self, article_json):
        """
        A function that saves the scraped article and its metadata to the save path defined
        in self.navigate_results_page().
        
        Parameters:
            article_json (dict): A dict that contains lists of article https links
                and their success status
        """
        
        article_save_path = self.save_path + article_json['doi'] + '.json'
        
        with open(article_save_path, 'w') as f:
            json.dump(article_json, f)
            f.close()
            
        return
    
    
    def save_article_remotely(self, article_json):
        """
        A function that saves a scraped article and its metadata to the
        MongoDB defined in self.db_handler().
        
        Parameters:
            article_json (dict): A dict that contains an article and its metadata
        """
        
        self.db_handler.upload_single_document(article_json)
            
        return
    
    