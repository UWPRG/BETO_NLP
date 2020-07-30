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
            type (str, required): type of text to be processed (abstracts or
                                  full text)
        """
        self.type = type
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
        if self.type == 'abstract':
            self.clean_abstracts = []
            for abstract in texts:
                assert abstract is not None, "NONETYPE"
                abstract = self.clean_abstract(abstract)
                self.clean_abstracts.append(abstract)


    def gather_chemical_entities(self, abstracts):
        for i, abstract in enumerate(abstracts):
            doc = Document(abstract)
            print('---Abstract {}---'.format(i))

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

        return clean_abstract
