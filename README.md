# BETO_NLP
An open-source toolkit for NLP to create corpora of academic texts and to extract compounds and properties from them.

# Usage:
-------------------------------

## Corpus building
-------------------------------
In order to build a corpus of journal articles from different publishers, BETO utilizes the API's that are available. Currently, these include Elsevier and Wiley (_via_ CrossRef). Each publisher requires different protocol and supporting functions, and so have different modules within BETO specialized to them. That said, the general protocol is to:

 1. Identify candidate article DOI or PII through API meta-data search
 
 2. Access full text articles
 
 3. Process full texts into JSON formatted objects
 
 4. Add the article JSON and any other files to the corpus
 
Finally, while these classes provide access to all articles for meta-data searching, access to the full text versions depend on the user's subscriptions. For this reason, it is suggested that a University-affiliated VPN is used in the background to direct web traffic and provide journal subscription information automatically (such as BIG-IP Edge Client).

### Elsevier corpus builder


### Wiley corpus builder
Usage of the `WileyCorpusGenerator()` class is described in the tutorials directory in the .ipynb 'wiley_corpus_generation.ipynb'. Considerations and prerequisites for using this class are detailed here.

As stated, access to individual articles are regulated by the user's subscription to the publisher. To get articles from the Wiley database, a user API token from the CrossRef Clickthrough API is required. To get this token, users must have a valid email address and ORCID ID. Links to appropriate sites for obtaining these credentials are described in the `WileyCorpusGenerator()` docstring.

The CrossRef API for Wiley allows access to the URL and DOI of articles during the meta-data search, and their PDF during the fulltext search. In order to convert from the PDF of the article to a JSON of the fulltext and (if the user disires) PNG of the figures in the articles, this class relies on the `scipdf-parser` package, which is a python interface for the Java API, GROBID. To use this API, *users must have a Java development kit (JDK) installed on their computers*. For this, the Oracle JDK is free and suggested: https://www.oracle.com/java/technologies/javase-downloads.html

Access to the GROBID API is run as a background process and needs to be terminated at the end of use. This is done automatically at the end of any DOI and URL dictionaries produced by the `WileyCorpusGenerator.meta_data_search()` method. This background subprocess can also be explicitly terminated by the user.

Users can specify whether or not to save the original PDF, as well as whether or not to extract any figures in the PDF. Figure information is automatically saved in the article JSON.

## Corpus processing and tokenization
-------------------------------


## Chemical entity and property extraction 
-------------------------------