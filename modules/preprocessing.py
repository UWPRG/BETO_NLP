import io
import os
import sys
import re
import json
import regex
import pickle
import numpy as np
import pandas as pd

import pubchempy as pcp
from chemdataextractor import Document

from mat2vec.processing.process import MaterialsTextProcessor

class PreProcessor():
    def __init__(self, texts, type='abstract'):
        """
        Parameters:
            texts (list, required): list of texts to be preprocessed
            type (str): type of text to be processed (abstracts or full text)
        """
        self.type = type
        self.min_len = 50
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

        if self.type == 'abstract': ### Clean abstracts
            self.clean_abstracts = []
            self.dropped_idxs = []
            warned = False
            for i, abstract in enumerate(texts):
                if abstract is None:
                    sys.stdout.write("\r\033[K"+"WARNING: TEXT LIST CONTAINS EMPTY ABSTRACTS. NOT ALL SAMPLES CAN BE CLEANED.")
                    warned = True
                else:
                    abstract = self.clean_abstract(abstract)
                    if len(abstract) < self.min_len:
                        self.dropped_idxs.append(i)
                    else:
                        self.clean_abstracts.append(abstract)
            if warned:
                print('\n')

        self.entity_to_cid = {}
        self.cid_to_synonyms = {}

    def load_search_history(self, path):
        with open(path) as f:
            search_history = json.load(f)
        self.entity_to_cid = search_history['entity_to_cid']
        self.cid_to_synonyms = search_history['cid_to_synonyms']

    def normalize_chemical_entities(self, abstracts, remove_abbreviations=True, verbose=False, write_bold=False, save=False):
        """
        Iterates through abstracts, extracts chemical entities and normalizes
        them

        Parameters:
            abstracts (list, required): List of abstracts to normalize
            remove_abbreviations (bool): If true then replace abbreviated
                                         entities with full name (only those
                                         extracted by chemdataextractor)
            verbose (bool): If true then prints abstract pre- and post-processing
            write_bold (bool): If true then prints chemical entities bolded to terminal
            save (bool): If true then saves abstracts, search history and run history
                         to preprocessor_files folder. WARNING: Running this function
                         twice without moving saved files will overwrite your previous
                         saves
        """

        self.normalized_abstracts = []
        self.entity_counts = {}
        self.entities_per_abstract = {}

        # Some entity names are ambiguous and must be hard coded
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
        for i, abstract in enumerate(abstracts):
            # if i % 10 == 0:
            #     print('{} %'.format(round(i / len(abstracts) * 100, 3)))
            ### Remove and normalize abbreviations
            if remove_abbreviations:
                abstract = self.remove_abbreviations(abstract)
            else:
                pass

            doc = Document(abstract)
            if verbose:
                print('---Abstract {}---'.format(i+1))
                print(abstract+'\n')
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

                # Add entity name, start and stop indices to dictionary
                entity_list.append((name, cem.start, cem.end))

                # Search entity in PubChem if not already done
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

            # Sort named entities by location in abstract and replace with
            # synonym
            entity_list.sort(key=lambda x:x[1])
            index_change = 0
            self.entities_per_abstract[i] = []
            for entity in entity_list:
                name, start, stop = entity
                if self.entity_to_cid[name][1] is not None:
                    replacement_name = self.entity_to_cid[name][1]
                else:
                    replacement_name = name
                if write_bold:
                    replacement_delta = len(replacement_name) - (stop - start) + 8
                    abstract = abstract[:start+index_change] + '\033[1m' + replacement_name + '\033[0m' + abstract[stop+index_change:]
                else:
                    replacement_delta = len(replacement_name) - (stop - start)
                    abstract = abstract[:start+index_change] + replacement_name + abstract[stop+index_change:]
                index_change += replacement_delta
                self.entities_per_abstract[i].append((replacement_name, start+index_change-replacement_delta, stop+index_change))
            if verbose:
                print(abstract)
                print('\n')

            self.normalized_abstracts.append(abstract)

        if save:
            os.makedirs('preprocessor_files', exist_ok=True)
            np.save('preprocessor_files/normalized_abstracts.npy', np.array(self.normalized_abstracts)) # all abstracts

            for cid, synonyms in self.cid_to_synonyms.items():
                self.cid_to_synonyms[cid] = list(set(synonyms))
            search_history = {'entity_to_cid': self.entity_to_cid,
                              'cid_to_synonyms': self.cid_to_synonyms}

            run_history = {'entities_per_abstract': self.entities_per_abstract,
                           'entity_counts': self.entity_counts}

            with io.open('preprocessor_files/search_history.json', 'w', encoding='utf8') as f: # map of entity names to CID/iupac names for quick recall
                out_ = json.dumps(search_history,                                              # map of CIDs to all unique synonyms
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                f.write(str(out_))
            with io.open('preprocessor_files/run_history.json', 'w', encoding='utf8') as f: # list of entities in each abstract with span
                out_ = json.dumps(run_history,                                              # dictionary of unique entities and their number of occurrences
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                f.write(str(out_))


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

    def remove_copywrite(self, abstract):
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
