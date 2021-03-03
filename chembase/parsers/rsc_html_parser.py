import os
import sys
from bs4 import UnicodeDammit

sys.path.append('/Users/wesleytatum/Desktop/post_doc/BETO/chemdataextractor-fork/')
from custom_cde.reader import rsc

class RscHtmlParser():
    """
    A callable HTML parser for articles published by the Royal Society of Chemistry.
    This class relies on ChemDataExtractor for conversion from HTML to a ChemDataExtractor
    Document() Object, a structured dictionary of string data. Custom functions convert
    the Document() to a full text string of the article.
    """
    
    def __init__(self,):
        """
        Initialize the HTML parser for articles published by the Royal Society of Chemistry.
        """
        
        
    def parse(self, html_string):
        """
        Convert HTML-formatted document to a single string.
        
        Paramenters:
            html_string (str): string containing the HTML for the article
            
        Returns:
            parsed_html (str): string containing the fulltext string for the article
        """
        reader = rsc.RscHtmlReader()
        
        converted = UnicodeDammit(html_string)
        cleaned_html = converted.unicode_markup
        
        article_doc = reader.readstring(cleaned_html)
        article_dict = article_doc.serialize()
        
        parsed_html = self.dict_to_string(article_dict)
        
        return parsed_html
    
    
    def dict_to_string(self, article_dict):
        """
        Convert Python dict containig ChemDataExtractor Document() object for
        the article.
        
        Parameters:
            article_dict (dict): dict containing article Document()
            
        Returns:
            article_string (str): single string containing the full text of the article
        """
        
        article_string = ''
        
        elements_list = article_dict['elements']
        
        for el in elements_list:
            for k, v in el.items():
                
                #paragraphs, headings, and references are simple key:value pairs
                if k == 'content':
                    article_string += v
                    article_string += ' '
                    
                #figure captions are contained in a nested dict
                if k == 'caption':
                    for j, c in v.items():
                        if j == 'content':
                            article_string += c
                            article_string += ' '
                    
                #table headings are contained in a list of list of dicts
                if k == 'headings':
                    for head_list in v:
                        for hl in head_list:
                            for j, c in hl.items():
                                if j == 'content':
                                    article_string += c
                                    article_string += ' '
                        
                    
                #table rows are contained in a list of list of dicts
                if k == 'rows':
                    for row_list in v:
                        for rl in row_list:
                            for j, c in rl.items():
                                if j == 'content':
                                    article_string += c 
                                    article_string += ' '
                                    
        article_string = article_string.replace('\n', '')
        article_string = article_string.replace(u'\xa0', u' ')

        return article_string
                            
                            