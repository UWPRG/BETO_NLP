import os
import sys
import json
import re
from tqdm import tqdm

#################################

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import generators.elsevier_corpus_generator as ecg

#################################

module_path1 = os.path.abspath(os.path.join('../../chemdataextractor-fork/'))
if module_path1 not in sys.path:
    sys.path.append(module_path1)
    
import custom_cde as cde
from custom_cde.reader import rsc, acs, nlm, pdf

#################################

class PropertyExtractor():
    """
    This class uses ChemDataExtractor to identify and extract chemical entities and
    their properties from texts. 
    """
    
    def __init__(self, article_format = 'HTML', publisher = 'RSC'):
        """
        This function initializes the class with attributes that will determine how the
        articles are loaded and have their information extracted using ChemDataExtractor
        
        Parameters:
            article_format (str): A string denoting the format of the article(s) that 
                are being fed to the extractor. Expects 'HTML', 'JSON', 'PDF', or 'string'.
                
            publisher (str): A string denoting the publisher that is the source for the
                article(s) that text_path points to. Currently accepts: 'RSC', 'ACS',
                'Elsevier', or 'Wiley'.
                
        Returns:
            None
        """
        
        self.format = article_format
        self.publisher = publisher
        
        if self.format == 'HTML':
            if self.publisher == 'RSC':
                self.reader = rsc.RscHtmlReader()
            elif self.publisher == 'ACS':
                self.reader = acs.AcsHtmlReader()
            else:
                self.reader = nlm.NlmXmlReader()
        
        #TODO: incorporate scipdf as a reader here
        if self.format == 'PDF':
            self.reader = pdf.PdfReader()
            
        return
    
    
    def extract_from_string(self, article_string):
        """
        Converts a full text article string to a cde.Document() and extracts chemical records
        
        Parameters:
            article_string (str): string containing the article text
            
        Returns:
            records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
        """
        doc = cde.Document.from_string(article_string)
        records = doc.records.serialize()
        
        return records
        
        
        
    def extract_from_pdf(self, article_pdf):
        """
        Converts a full text article pdf to a cde.Document() and extracts chemical records
        
        Parameters:
            article_pdf (str): string containing the path to the pdf article. Reader determined by self.
                reader
            
        Returns:
            records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
        """
        
        f = open(pdf_path, 'rb')
        doc = cde.Document.from_file(f, readers = [self.reader])
        f.close()
        
        records = doc.records.serialize()
        
        return records
    
    
    def extract_from_html(self, html_string):
        """
        Converts a full text article HTML to a cde.Document() and extracts chemical records
        
        Parameters:
            html_string (str): string containing the article HTML. Reader determined by self.
                reader
            
        Returns:
            records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
        """
        html_string = html_string.encode()
        
        doc = cde.Document.from_string(html_string, readers = [self.reader])
        records = doc.records.serialize()
        
        return records
        
        
    def extract_from_json(self, article_json):
        """
        Converts a full text article json to a cde.Document() and extracts chemical records
        
        Parameters:
            html_string (json): dict containing the article text sections. Assumes parsed Elsevier
                fulltext format
            
        Returns:
            records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
        """
        if self.publisher == 'Elsevier':
            article_text = self.load_elsevier_json(article_json)
            
        else: #TODO: Build more publisher-specific json loaders
            article_text = self.load_elsevier_json(article_json)
            
        doc = cde.Document.from_string(article_text)
        records = doc.records.serialize()
        
        return records
        
        
    def load_elsevier_json(self, article_json):
        """
        Loads a parsed Elsevier article json. Stitches together sections into single string
        for easier parsing
        
        Parameters:
            article_json (json): dict containing article full text and metadata
            
        Returns:
            article_text (str): the combined article sections as a single string
        """
        sfp = ecg.SciFullTextProcessor()
        
        json_keys = list(json_texts.keys())
        drop_keys = ['Section Headers',
                     'Meta-data',
                     'Acknowledgements',
                     'Reference',
                     'errors']
        
        dict_text = {}

        for key in drop_keys:
            if key in json_keys:
                json_keys.remove(key)

        for key in json_keys:
            dict_text[key] = article_json[key]

        article_text = sfp.restitch_text(dict_text)
        

        
        return article_text
    
    
    def extract(self, article):
        """
        According to the article format (as declared in  self.__ini__()), applies appropriate
        extraction functions.
        
        Parameters:
            article (?): article of pre-specified format to be parsed
            
        Returns:
            records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
        """
        
        if self.format == 'HTML':
            records = self.extract_from_html(article)
        
        #can also incorporate scipdf as a reader here
        if self.format == 'PDF':
            records = self.extract_from_pdf(article)
            
        if self.format == 'string':
            records = self.extract_from_string(article)
            
        if self.format == 'JSON':
            records = self.extract_from_json(article)
            
        return records
    
    
    def filter_properties(self, raw_records, recs_with_props_only = 'False'):
        """
        Remove uneccesary entity information from the raw records returned by self.extract(). If
        recs_with_props_only is True, then only chemical records that have more than a name is
        added to the dictionary of results that is returned.
        
        Parameters:
            raw_records (json): serialized version of cde.Document.records() that lists entities
                and their detected properties
                
            recs_with_props_only (bool): determines whether all chemical records are returned or
                just those that also have properties.
                
        Returns:
            new_recs (json): dictionary containing the refined and maybe filtered records.
        """
        new_recs = []
        
        for rec in raw_records:
            new_rec = {}
            
            for k, v in rec.items():
                #remove roles and labels
                if k == 'roles' or k == 'labels':
                    pass
                
                #keep all other fields
                else:
                    new_rec[k] = v
            
            if recs_with_props_only == True:
                #filter out all chemical records that only have a 'names' field
                if len(new_rec.values()) > 1:
                    new_recs.append(new_rec)
                    
            else:
                new_recs.append(new_rec)

        return new_recs
    
    
    def reformat_list_of_records(self, records_list, doi):
        """
        This function converts the records dictionary to the format that will be used to store
        the extracted compound/property information in the database collection MolecularEntities
        
        Parameters:
            records_list (list of json): list of chemical records, perhaps
                for a single article. For each record, creates newly formatted dict and appends
                to the list of new records, `reformatted`
                
            doi (list of str): list of DOI that each record is associated with. This is required
                so that each property in the database has an associated reference
                
        Returns:
            reformatted (list of json): the list of newly reformatted properties json. This can
                be iterated through to directly update the MolecularEntities collection.
        """
        reformatted = []

        for rec in records_list:
            new_rec = {'Synonyms': [],
                       'Properties': []}
            prop_dict = {}
            
            for k, v in rec.items():
                if k == 'names':
                    new_rec['Synonyms'] = v
                    
                else:
                    prop_dict[k] = v
                    
            #add associated DOI
            prop_dict['AssociatedDOI'] = doi
            
            #update properties dict
            new_rec['Properties'].append(prop_dict)

            reformatted.append(new_rec)
            
        return reformatted
        
        