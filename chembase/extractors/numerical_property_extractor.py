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
        
        """
        doc = cde.Document.from_string(article_string)
        records = doc.records.serialize()
        
        return records
        
        
        
    def extract_from_pdf(self, article_pdf):
        """
        
        """
        
        f = open(pdf_path, 'rb')
        doc = cde.Document.from_file(f, readers = [self.reader])
        f.close()
        
        records = doc.records.serialize()
        
        return records
    
    
    def extract_from_html(self, html_string):
        """
        
        """
        html_string = html_string.encode()
        
        doc = cde.Document.from_string(html_string, readers = [self.reader])
        records = doc.records.serialize()
        
        return records
        
        
    def extract_from_json(self, article_json):
        """
        
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
        
        