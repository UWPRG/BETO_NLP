import io
import os
import sys
import re
import json
import regex
import string
import pickle
import numpy as np
import pandas as pd

import pubchempy as pcp
from chemdataextractor import Document
from chemdataextractor.doc import Paragraph

from mat2vec.processing.process import MaterialsTextProcessor


class PreProcessor():
    """
    Class that allows a variety of preprocessing functions to be applied to a set
    of scientific texts - either abstracts or full texts. Consists of functions for
        1. Cleaning - removing unwanted line breaks, tokens or words
        2. Normalizing - combining like words or phrases to single entities
            a. Chemical Entities
            b. Properties
    """
    def __init__(self, texts, type='abstract'):
        """
        Parameters:
            texts (list, required): list of texts to be preprocessed
            type (str): type of text to be processed (abstracts or full text)
        """
        self.type = type
        self.min_len = 50
        self.texts = texts

        ### Element text and regex
        self.ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                         "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                         "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                         "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                         "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                         "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
                         "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

        self.ELEMENT_NAMES = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                              "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                              "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                              "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                              "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                              "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                              "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                              "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                              "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                              "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                              "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                              "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                              "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                              "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                              "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]

        self.element_dict = {}
        for element, name in zip(self.ELEMENTS, self.ELEMENT_NAMES):
            self.element_dict[element] = name

        self.FORMULA_REGX = regex.compile(r"^("+r"|".join(self.ELEMENTS) +
                                          r"|[0-9])+$")
        self.VALENCE_REGX = regex.compile(r"^("+r"|".join(self.ELEMENTS) +
                                          r")(\(([IV|iv]|[Vv]?[Ii]{0,3})\))$")

        ### Numbers and units text and regex
        self.SPLIT_UNITS = ["K", "h", "V", "wt", "wt.", "MHz", "kHz", "GHz", "Hz", "days", "weeks",
                            "hours", "minutes", "seconds", "T", "MPa", "GPa", "at.", "mol.",
                            "at", "m", "N", "s-1", "vol.", "vol", "eV", "A", "atm", "bar",
                            "kOe", "Oe", "h.", "mWcm−2", "keV", "MeV", "meV", "day", "week", "hour",
                            "minute", "month", "months", "year", "cycles", "years", "fs", "ns",
                            "ps", "rpm", "g", "mg", "mAcm−2", "mA", "mK", "mT", "s-1", "dB",
                            "Ag-1", "mAg-1", "mAg−1", "mAg", "mAh", "mAhg−1", "m-2", "mJ", "kJ",
                            "m2g−1", "THz", "KHz", "kJmol−1", "Torr", "gL-1", "Vcm−1", "mVs−1",
                            "J", "GJ", "mTorr", "bar", "cm2", "mbar", "kbar", "mmol", "mol", "molL−1",
                            "MΩ", "Ω", "kΩ", "mΩ", "mgL−1", "moldm−3", "m2", "m3", "cm-1", "cm",
                            "Scm−1", "Acm−1", "eV−1cm−2", "cm-2", "sccm", "cm−2eV−1", "cm−3eV−1",
                            "kA", "s−1", "emu", "L", "cmHz1", "gmol−1", "kVcm−1", "MPam1",
                            "cm2V−1s−1", "Acm−2", "cm−2s−1", "MV", "ionscm−2", "Jcm−2", "ncm−2",
                            "Jcm−2", "Wcm−2", "GWcm−2", "Acm−2K−2", "gcm−3", "cm3g−1", "mgl−1",
                            "mgml−1", "mgcm−2", "mΩcm", "cm−2s−1", "cm−2", "ions", "moll−1",
                            "nmol", "psi", "mol·L−1", "Jkg−1K−1", "km", "Wm−2", "mass", "mmHg",
                            "mmmin−1", "GeV", "m−2", "m−2s−1", "Kmin−1", "gL−1", "ng", "hr", "w",
                            "mN", "kN", "Mrad", "rad", "arcsec", "Ag−1", "dpa", "cdm−2",
                            "cd", "mcd", "mHz", "m−3", "ppm", "phr", "mL", "ML", "mlmin−1", "MWm−2",
                            "Wm−1K−1", "Wm−1K−1", "kWh", "Wkg−1", "Jm−3", "m-3", "gl−1", "A−1",
                            "Ks−1", "mgdm−3", "mms−1", "ks", "appm", "ºC", "HV", "kDa", "Da", "kG",
                            "kGy", "MGy", "Gy", "mGy", "Gbps", "μB", "μL", "μF", "nF", "pF", "mF",
                            "A", "Å", "A˚", "μgL−1"]

        self.NUMBER_REGX = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
        self.UNIT_REGX = regex.compile(r"^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)", regex.DOTALL)
        self.PUNCT_REGX = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]

        ### Clean texts - different logic for abstracts and full text
        if self.type == 'abstract':
            self.clean_texts = []
            self.dropped_idxs = []
            warned = False
            too_short = False
            for i, abstract in enumerate(texts):
                if abstract is None:
                    sys.stdout.write("\r\033[K"+"WARNING: TEXT LIST CONTAINS EMPTY ABSTRACTS. NOT ALL SAMPLES CAN BE CLEANED.")
                    warned = True
                    self.dropped_idxs[i]
                else:
                    abstract = self.clean_abstract(abstract)
                    if len(abstract) < self.min_len:
                        self.dropped_idxs.append(i)
                        too_short = True
                    else:
                        self.clean_texts.append(abstract)
            if too_short:
                sys.stdout.write("\r\033[K"+"WARNING: SOME ABSTRACTS WERE DROPPED DUE TO LENGTH. CHECK dropped_idxs FOR IDX NUMBERS.")
                print('\n')
            elif warned:
                print('\n')

        self.normalized_texts = []
        self.entity_counts = {}
        self.entities_per_text = {}
        self.entity_to_cid = {}
        self.cid_to_synonyms = {}

    ########### TOKENIZING FUNCTIONS #############
    """
    Much of the tokenizing functionality has been adapted from the tokenization workflow
    in "Unsupervised word embeddings capture latent knowledge from materials science
    literature" (10.1038/s41586-019-1335-8)
    """

    def tokenize(self, keep_sentences=False, exclude_punct=False):
        """
        Takes the set of normalized texts and tokenizes them.

        Parameters:
            keep_sentences (bool): If true then will save tokenized abstracts as
                                   a list of lists where each nested list is a single
                                   sentence. If false a single list will be returned
                                   containing all tokens
            exclude_punct (bool): If true then standard punctuation will be left out
                                  of token list. If false punctuation tokens will also
                                  be returned
        """
        assert len(self.normalized_texts) > 0, "ERROR: NO NORMALIZED TEXTS FOUND. YOU MUST RUN A LOAD OR NORMALIZATION FUNCTION PRIOR TO TOKENIZING"
        if len(self.entities_per_text) == 0:
            print("WARNING: NO ENTITIES ASSOCIATED WITH THESE TEXTS. ENTITIES MAY BE SPLIT ACCIDENTALLY DURING TOKENIZATION")
        elif len(self.entities_per_text) != len(self.normalized_texts):
            print("WARNING: LENGTH OF ENTITY DICT DOES NOT MATCH NUMBER OF TEXTS. SOME TEXTS WILL BE TOKENIZED WITH NO KNOWLEDGE OF CHEMICAL ENTITIES")

        self.tokens_per_text = {}
        for i, text in enumerate(self.normalized_texts):
            cde_tokens = Paragraph(text).tokens
            tokens = []
            for sentence in cde_tokens:
                if keep_sentences:
                    tokens.append([])
                    for token in sentence:
                        token = token.text
                        token = self.process_token(token,
                                                   exclude_punct)
                        tokens[-1] += token
                else:
                    for token in sentence:
                        token = token.text
                        token = self.process_token(token,
                                                   exclude_punct)
                        tokens += token

            self.tokens_per_text[i] = tokens

    def process_token(self, token, exclude_punct):
        """
        Takes a single token and applies rules-based preprocessing

        Parameters:
            token (str, required): Token string
            exclude_punct (bool): See tokenize()

        Returns
            processed_token(s) (list): List of processed token(s)
        """
        ### Check to see if token should be split into two tokens
        unit_match = self.UNIT_REGX.match(token)
        if unit_match is not None and unit_match.group(2) in self.SPLIT_UNITS:
            token1, token2 = unit_match.group(1), unit_match.group(2)
            token1 = self.process_token(token1,
                                        exclude_punct)[0]
            token2 = self.process_token(token2,
                                        exclude_punct)[0]
            return [token1, token2]
        else:
            pass

        if exclude_punct and token in self.PUNCT_REGX:
            return []
        else:
            return [token]


    ########### LOADING FUNCTIONS ###############

    def load_search_history(self, path):
        """
        Loads a series of PubChem searches so those terms will not be searched
        again in subsequent runs

        Parameters:
            path (str, required): Path to json file containing search history
                                  dictionaries
        """
        with open(path) as f:
            search_history = json.load(f)
        self.entity_to_cid = search_history['entity_to_cid']
        self.cid_to_synonyms = search_history['cid_to_synonyms']

    def load_preprocess_history(self, path):
        """
        Loads dictionaries of entity names and counts

        Parameters:
            path (str, required): Path to json file containing preprocessing history
        """
        with open(path) as f:
            preprocess_history = json.load(f)
        self.entities_per_text = preprocess_history['entities_per_text']
        self.entity_counts = preprocess_history['entity_counts']

    def load_normalized_texts(self, path):
        """
        Loads a list of normalized texts

        Parameters:
            path (str, required): Path to numpy file containing normalized texts
        """
        self.normalized_texts = np.load(path, allow_pickle=True)

    def load_preprocessor(self, dir):
        """
        Loads all manually created preprocessor save files. Files must have the
        same name as when written by the PreProcessor object

        Parameters:
            dir (str, required): Path to folder containing save files
        """
        fns = os.listdir(dir)
        for fn in fns:
            path = os.path.join(dir, fn)
            if fn == 'normalized_texts.npy':
                load_normalized_texts(path)
            elif fn == 'search_history.json':
                load_search_history(path)
            elif fn == 'preprocess_history.json':
                load_preprocess_history(path)


    ########## CLEANING FUNCTIONS ###############

    def clean_abstract(self, abstract):
        """
        Takes an abstract and applies rules-based preprocessing to remove
        unwanted features from the raw Elsevier abstract download

        Parameters:
            abstract (str, required): The abstract which you want to clean

        Returns:
            cleaned_abstract (str): The cleaned abstract
        """
        abstract = abstract.split('\n')
        info = []
        for line in abstract:
            line = line.strip()
            if line != '':
                info.append(line)
        if len(info) == 2:
            clean_abstract = info[1]
        elif len(info) == 1:
            if info[0].split()[0].lower() == 'abstract':
                clean_abstract = ' '.join(info[0].split()[1:])
            elif info[0].split()[0].lower() == 'summary':
                clean_abstract = ' '.join(info[0].split()[1:])
            elif 'objective' in info[0].split()[0].lower():
                clean_abstract = ' '.join(info[0].split()[1:])
            else:
                clean_abstract = info[0]
        else:
            info_lower = [x.lower() for x in info]
            section_titles = ['introduction',
                              'purpose',
                              'background',
                              'scope and approach',
                              'objective',
                              'objectives',
                              'materials and methods',
                              'results',
                              'conclusion',
                              'conclusions',
                              'key findings',
                              'key findings and conclusions',
                              'methodology',
                              'methods',
                              'study design',
                              'clinical implications']
            sectioned = False
            for section_title in section_titles:
                if section_title in info_lower:
                    sectioned = True
            if sectioned:
                if info[0].lower() == 'abstract':
                    text = []
                    for entry in info[1:]:
                        if entry.lower() in section_titles:
                            pass
                        else:
                            text.append(entry)
                    clean_abstract = ' '.join(text)
                elif info[0].lower() == 'summary':
                    text = []
                    for entry in info[1::]:
                        if entry.lower() in section_titles:
                            pass
                        else:
                            text.append(entry)
                    clean_abstract = ' '.join(text)
                else:
                    text = []
                    for entry in info:
                        if entry.lower() in section_titles:
                            pass
                        else:
                            text.append(entry)
                    clean_abstract = ' '.join(text)
            else:
                if info[0].lower() == 'abstract' or info[0].lower() == 'absract' or info[0].lower() == 'abstact' or info[0].lower() == 'abstractt':
                    clean_abstract = ' '.join(info[1:])
                elif info[0].lower() == 'summary' or info[0].lower() == 'publisher summary' or info[0].lower() == '1. summary':
                    clean_abstract = ' '.join(info[1:])
                elif info[0] == 'This article has been retracted: please see Elsevier Policy on Article Withdrawal (https://www.elsevier.com/about/our-business/policies/article-withdrawal).':
                    clean_abstract = 'Retracted'
                else:
                    clean_abstract = ' '.join(info)

        clean_abstract = self.remove_copywrite(clean_abstract)

        return clean_abstract

    def remove_symbols(self, abstract):
        """
        This function takes an abstract and removes the copywrite information
        followed by the Elsevier text and publication year and returns a clean
        abstract

        Parameters:
            abstract (str, required): The abstract which has unwanted copywrite text

        Returns:
            clean_abstract (str): The clean abstract
        """

        split = abstract.split()

        if '®' in split:
            temp = list(split)
            del temp[temp.index('®')]
            split = "".join(temp)

        if '©' in split:
            if split[0] != '©':
                index = split.index('©')
                del split[index:]
                clean_abstract = ' '.join(split)
            elif split[0] == '©':
                if 'B.V.' in split:
                    new_idx = split.index('B.V.')
                    del split[0:new_idx+1]
                    clean_abstract = ' '.join(split)
                elif 'B.V..' in split:
                    new_idxs = split.index('B.V..')
                    del split[0:new_idxs+1]
                    clean_abstract = ' '.join(split)
                else:
                    del split[0:2]
                    clean_abstract = ' '.join(split)
        else:
            clean_abstract = abstract

        if clean_abstract.endswith(' Crown Copywrite'):
            clean_abstract.replace(' Crown Copywrite', '')

        return clean_abstract


    ############ NORMALIZATION FUNCTIONS ##############

    def normalize_chemical_entities(self, texts='default', remove_abbreviations=True, verbose=False, write_bold=False, save=False):
        """
        Iterates through texts, extracts chemical entities and normalizes
        them

        Parameters:
            texts (list, required): List of texts to normalize
            remove_abbreviations (bool): If true then replace abbreviated
                                         entities with full name (only those
                                         extracted by chemdataextractor)
            verbose (bool): If true then prints text pre- and post-processing
            write_bold (bool): If true then prints chemical entities bolded to terminal
            save (bool): If true then saves texts, search history and run history
                         to preprocessor_files folder. WARNING: Running this function
                         twice without moving saved files will overwrite your previous
                         saves
        """
        if texts == 'default':
            texts = self.clean_texts

        ### Some entity names are ambiguous and must be hard coded
        self.entity_to_cid['CO'] = [281, 'carbon monoxide']
        self.entity_to_cid['Co'] = [104730, 'cobalt']
        self.entity_to_cid['NO'] = [145068, 'nitric oxide']
        self.entity_to_cid['No'] = [24822, 'nobelium']
        self.entity_to_cid['sugar'] = [None, None]
        self.cid_to_synonyms[281] = ['CO']
        self.cid_to_synonyms[104730] = ['Co']
        self.cid_to_synonyms[145068] = ['NO']
        self.cid_to_synonyms[24822] = ['No']
        self.entity_counts['carbon monoxide'] = 1
        self.entity_counts['cobalt'] = 1
        self.entity_counts['nitric oxide'] = 1
        self.entity_counts['nobelium'] = 1
        self.entity_counts['sugar'] = 1
        for i, text in enumerate(texts):
            ### Remove and normalize abbreviations
            if remove_abbreviations:
                text = self.remove_abbreviations(text)
            else:
                pass

            doc = Document(text)
            if verbose:
                print('---Text {}---'.format(i+1))
                print(text+'\n')
            cems = doc.cems
            entity_list = []
            for cem in cems:
                ### Check if abbreviated valence state
                elem_valence = self.VALENCE_REGX.match(cem.text)
                if elem_valence:
                    elem = elem_valence.group(1)
                    valence = elem_valence.group(2)
                    elem = self.element_dict[elem]
                    cem.text = elem+valence

                ### Check if ambiguous formula
                if self.FORMULA_REGX.match(cem.text):
                    name = cem.text
                else:
                    name = cem.text.lower()

                ### Add entity name, start and stop indices to dictionary
                entity_list.append((name, cem.start, cem.end))

                ### Search entity in PubChem if not already done
                if name not in self.entity_to_cid.keys():
                    c = pcp.get_compounds(name, 'name')
                    if len(c) == 0:
                        self.entity_to_cid[name] = [None, None]
                        self.entity_counts[name] = 1
                    else:
                        c = c[0]
                        cid = c.cid
                        iupac_name = c.iupac_name
                        self.entity_to_cid[name] = [cid, iupac_name]
                        if cid not in self.cid_to_synonyms.keys():
                            self.cid_to_synonyms[cid] = [name]
                        else:
                            self.cid_to_synonyms[cid].append(name)
                        self.entity_counts[self.entity_to_cid[name][1]] = 1
                else:
                    if self.entity_to_cid[name][0] is None:
                        self.entity_counts[name] += 1
                    else:
                        self.entity_counts[self.entity_to_cid[name][1]] += 1

            ### Sort named entities by location in text and replace with synonym
            entity_list.sort(key=lambda x:x[1])
            index_change = 0
            self.entities_per_text[i] = []
            for entity in entity_list:
                name, start, stop = entity
                if self.entity_to_cid[name][1] is not None:
                    replacement_name = self.entity_to_cid[name][1]
                else:
                    replacement_name = name
                if write_bold:
                    replacement_delta = len(replacement_name) - (stop - start) + 8
                    text = text[:start+index_change] + '\033[1m' + replacement_name + '\033[0m' + text[stop+index_change:]
                else:
                    replacement_delta = len(replacement_name) - (stop - start)
                    text = text[:start+index_change] + replacement_name + text[stop+index_change:]
                index_change += replacement_delta
                self.entities_per_text[i].append((replacement_name, start+index_change-replacement_delta, stop+index_change, name))
            if verbose:
                print(text)
                print('\n')

            self.normalized_texts.append(text)

        if save:
            os.makedirs('preprocessor_files', exist_ok=True)
            np.save('preprocessor_files/normalized_texts.npy', np.array(self.normalized_texts)) # all texts

            for cid, synonyms in self.cid_to_synonyms.items():
                self.cid_to_synonyms[cid] = list(set(synonyms))
            search_history = {'entity_to_cid': self.entity_to_cid,
                              'cid_to_synonyms': self.cid_to_synonyms}

            preprocess_history = {'entities_per_text': self.entities_per_text,
                                  'entity_counts': self.entity_counts}

            with io.open('preprocessor_files/search_history.json', 'w', encoding='utf8') as f: # map of entity names to CID/iupac names for quick recall
                out_ = json.dumps(search_history,                                              # map of CIDs to all unique synonyms
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                f.write(str(out_))
            with io.open('preprocessor_files/preprocess_history.json', 'w', encoding='utf8') as f: # list of entities in each text with span
                out_ = json.dumps(preprocess_history,                                              # dictionary of unique entities and their number of occurrences
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                f.write(str(out_))

    def remove_abbreviations(self, abstract):
        doc = Document(abstract)
        abbvs = doc.abbreviation_definitions
        cems = doc.cems
        if len(abbvs) > 0:
            abbv_dict = {}
            for abbv in abbvs:
                cem_starts = []
                cem_ends = []
                if abbv[-1] is not None:
                    abbv_dict[abbv[0][0]] = [' '.join(abbv[1])]
                    for cem in cems:
                        if cem.text == abbv[0][0]:
                            cem_starts.append(cem.start)
                            cem_ends.append(cem.end)
                    if len(cem_starts) > 0:
                        low_idx = cem_starts[np.argmin(cem_starts)]
                    else:
                        low_idx = 0
                    abbv_dict[abbv[0][0]].append(low_idx)
            abbv_dict = {k: v for k, v in sorted(abbv_dict.items(), key=lambda item: item[1][1])}
            index_change = 0
            for abbv in abbv_dict.keys():
                non_abbv = abbv_dict[abbv][0]
                if abbv_dict[abbv][1] != 0:
                    replacement_delta = len(non_abbv) - len(abbv)
                    cem_starts = []
                    cem_ends = []
                    for cem in cems:
                        if cem.text == abbv:
                            cem_starts.append(cem.start)
                            cem_ends.append(cem.end)
                    if len(cem_starts) == 1:
                        if abstract[cem_starts[0]+index_change-1]+abstract[cem_ends[0]+index_change] == '()':
                            abstract = abstract[:cem_starts[0]-2+index_change] + abstract[cem_ends[0]+1+index_change:]
                            index_change += cem_starts[0] - cem_ends[0] - 3
                        else:
                            pass
                    else:
                        low_idx = np.argmin(cem_starts)
                        cem_start_low = cem_starts[low_idx]
                        cem_end_low = cem_ends[low_idx]
                        if abstract[cem_start_low+index_change-1]+abstract[cem_end_low+index_change] == '()':
                            abstract = abstract[:cem_start_low-2+index_change] + abstract[cem_end_low+1+index_change:]
                            index_change += cem_start_low - cem_end_low - 3
                        else:
                            pass
                    abstract = re.sub(r'([\s]){}([.,;\s]|$)'.format(abbv), r' {}\2'.format(non_abbv), abstract)
                else:
                    pass
        return abstract
