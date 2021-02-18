import re
import os
import sys
from glob import glob
import urllib
import subprocess
import json
import requests
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm, tqdm_notebook


GROBID_URL = 'http://localhost:8070'
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PDF_FIGURES_JAR_PATH = os.path.join(DIR_PATH, 'pdffigures2', 'pdffigures2-assembly-0.0.12-SNAPSHOT.jar')


def list_pdf_paths(pdf_folder):
    """
    list of pdf paths in pdf folder
    """
    return glob(os.path.join(pdf_folder, '*', '*', '*.pdf'))


def validate_url(path):
    """
    Validate a given ``path`` if it is URL or not
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, path) is not None


def parse_pdf(pdf_path,
              fulltext=True,
              soup=False,
              grobid_url=GROBID_URL):
    """
    Function to parse PDF to XML or BeautifulSoup using GROBID tool
    
    You can see http://grobid.readthedocs.io/en/latest/Install-Grobid/ on how to run GROBID locally
    After loading GROBID zip file, you can run GROBID by using the following
    >> ./gradlew run
    
    Parameters
    ==========
    pdf_path: str, path to publication or article
    fulltext: bool, option for parsing, if True, parse full text of the article
        if False, parse only header
    grobid_url: str, url to GROBID parser, default at 'http://localhost:8070'
    soup: bool, if True, return BeautifulSoup of the article
    
    Output
    ======
    parsed_article: if soup is False, return parsed XML in text format, 
        else return BeautifulSoup of the XML
    Example
    =======
    >> parsed_article = parse_pdf(pdf_path, fulltext=True, soup=True)
    """
    # GROBID URL
    if fulltext:
        url = '%s/api/processFulltextDocument' % grobid_url
    else:
        url = '%s/api/processHeaderDocument' % grobid_url

    if validate_url(pdf_path) and os.path.splitext(pdf_path)[-1] != '.pdf':
        print("The input URL has to have base name PDF.")
        parsed_article = None
    elif validate_url(pdf_path) and os.path.splitext(pdf_path)[-1] == '.pdf':
        page = urllib.request.urlopen(pdf_path).read()
        parsed_article = requests.post(url, files={'input': page}).text
    elif os.path.exists(pdf_path):
        parsed_article = requests.post(url, files={'input': open(pdf_path, 'rb')}).text
    else:
        parsed_article = None

    if soup and parsed_article is not None:
        parsed_article = BeautifulSoup(parsed_article, 'lxml')
    return parsed_article


def parse_abstract(article):
    """
    Parse abstract from a given BeautifulSoup of an article 
    """
    div = article.find('abstract')
    abstract = ''
    for p in list(div.children):
        if not isinstance(p, NavigableString) and len(list(p)) > 0:
            abstract += ' '.join([elem.text for elem in p if not isinstance(elem, NavigableString)])    
    return abstract


def calculate_number_of_references(div):
    """
    For a given section, calculate number of references made in the section
    """
    n_publication_ref = len([ref for ref in div.find_all('ref') if ref.attrs.get('type') == 'bibr'])
    n_figure_ref = len([ref for ref in div.find_all('ref') if ref.attrs.get('type') == 'figure'])
    return {
        'n_publication_ref': n_publication_ref,
        'n_figure_ref': n_figure_ref
    }


def parse_sections(article):
    """
    Parse list of sections from a given BeautifulSoup of an article 
    """
    article_text = article.find('text')
    divs = article_text.find_all('div', attrs={'xmlns': 'http://www.tei-c.org/ns/1.0'})
    sections = []
    for div in divs:
        div_list = list(div.children)
        if len(div_list) == 0:
            heading = ''
            text = ''
        elif len(div_list) == 1:
            if isinstance(div_list[0], NavigableString):
                heading = str(div_list[0])
                text = ''
            else:
                heading = ''
                text = div_list[0].text
        else:
            text = []
            heading = div_list[0]
            if isinstance(heading, NavigableString):
                heading = str(heading)
                p_all = list(div.children)[1:]
            else:
                heading = ''
                p_all = list(div.children)
            for p in p_all:
                if p is not None:
                    try:
                        text.append(p.text)
                    except:
                        pass
            text = ' '.join(text)
        if heading is not '' or text is not '':
            ref_dict = calculate_number_of_references(div)
            sections.append({
                'heading': heading,
                'text': text,
                'n_publication_ref': ref_dict['n_publication_ref'],
                'n_figure_ref': ref_dict['n_figure_ref']
            })
    return sections


def parse_references(article):
    """
    Parse list of references from a given BeautifulSoup of an article
    """
    reference_list = []
    references = article.find('text').find('div', attrs={'type': 'references'})
    references = references.find_all('biblstruct') if references is not None else []
    reference_list = []
    for reference in references:
        title = reference.find('title', attrs={'level': 'a'})
        if title is None:
            title = reference.find('title', attrs={'level': 'm'})
        title = title.text if title is not None else ''
        journal = reference.find('title', attrs={'level': 'j'})
        journal = journal.text if journal is not None else ''
        if journal is '':
            journal = reference.find('publisher')
            journal = journal.text if journal is not None else ''
        year = reference.find('date')
        year = year.attrs.get('when') if year is not None else ''
        authors = []
        for author in reference.find_all('author'):
            firstname = author.find('forename', {'type': 'first'})
            firstname = firstname.text.strip() if firstname is not None else ''
            middlename = author.find('forename', {'type': 'middle'})
            middlename = middlename.text.strip() if middlename is not None else ''
            lastname = author.find('surname')
            lastname = lastname.text.strip() if lastname is not None else ''
            if middlename is not '':
                authors.append(firstname + ' ' + middlename + ' ' + lastname)
            else:
                authors.append(firstname + ' ' + lastname)
        authors = '; '.join(authors)
        reference_list.append({
            'title': title,
            'journal': journal,
            'year': year,
            'authors': authors
        })
    return reference_list


def parse_figure_caption(article):
    """
    Parse list of figures/tables from a given BeautifulSoup of an article
    """
    figures_list = []
    figures = article.find_all('figure')
    for figure in figures:
        figure_type = figure.attrs.get('type') or ''
        figure_id = figure.attrs['xml:id'] or ''
        label = figure.find('label').text
        if figure_type == 'table':
            caption = figure.find('figdesc').text
            data = figure.table.text
        else:
            caption = figure.text
            data = ''
        figures_list.append({
            'figure_label': label,
            'figure_type': figure_type,
            'figure_id': figure_id,
            'figure_caption': caption,
            'figure_data': data
        })
    return figures_list


def convert_article_soup_to_dict(article):
    """
    Function to convert BeautifulSoup to JSON format 
    similar to the output from https://github.com/allenai/science-parse/

    Parameters
    ==========
    article: BeautifulSoup

    Output
    ======
    article_json: dict, parsed dictionary of a given article in the following format
        {
            'title': ..., 
            'abstract': ..., 
            'sections': [
                {'heading': ..., 'text': ...}, 
                {'heading': ..., 'text': ...},
                ...
            ],
            'references': [
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...}, 
                {'title': ..., 'journal': ..., 'year': ..., 'authors': ...},
                ...
            ], 
            'figures': [
                {'figure_label': ..., 'figure_type': ..., 'figure_id': ..., 'figure_caption': ..., 'figure_data': ...},
                ...
            ]
        }
    """
    article_dict = {}
    if article is not None:
        title = article.find('title', attrs={'type': 'main'})
        title = title.text.strip() if title is not None else ''
        article_dict['title'] = title
        article_dict['abstract'] = parse_abstract(article)
        article_dict['sections'] = parse_sections(article)
        article_dict['references'] = parse_references(article)
        article_dict['figures'] = parse_figure_caption(article)

        doi = article.find('idno', attrs={'type': 'DOI'})
        doi = doi.text if doi is not None else ''
        article_dict['doi'] = doi

        return article_dict
    else:
        return None


def parse_pdf_to_dict(pdf_path):
    """
    Parse the given 

    Parameters
    ==========
    pdf_path: str, path to publication or article

    Ouput
    =====
    article_dict: dict, dictionary of an article
    """
    parsed_article = parse_pdf(pdf_path, fulltext=True, soup=True)
    article_dict = convert_article_soup_to_dict(parsed_article)
    return article_dict


def parse_figures(pdf_folder,
                  jar_path=PDF_FIGURES_JAR_PATH,
                  resolution=300, 
                  output_folder='figures'):
    """
    Parse figures from the given scientific PDF using pdffigures2

    Parameters
    ==========
    pdf_folder: str, path to a folder that contains PDF files. A folder must contains only PDF files
    jar_path: str, default path to pdffigures2-assembly-0.0.12-SNAPSHOT.jar file
    resolution: int, resolution of the output figures
    output_folder: str, path to folder that we want to save parsed data (related to figures) and figures

    Output
    ======
    folder: making a folder of output_folder/data and output_folder/figures of parsed data and figures relatively
    """
    data_path = os.path.join(output_folder, 'data')
    figure_path = os.path.join(output_folder, 'figures')

    if os.path.isdir(output_folder):
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        if not os.path.exists(figure_path):
            os.mkdir(figure_path)

        if os.path.isdir(data_path) and os.path.isdir(figure_path):
            args = [
                'java',
                '-jar', jar_path,
                pdf_folder,
                '-i', str(resolution),
                '-d', os.path.join(os.path.abspath(data_path), ''),
                '-m', os.path.join(os.path.abspath(figure_path), '') # end path with "/"
            ]
            _ = subprocess.run(
                args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20
            )
            print('Done parsing figures from PDFs!')
    else:
        print('output_folder have to be path to folder')