import pandas as pd

class PropertyHandler():
    """
    This class is meant to parse and consolidate the property information stored in the MongoDB
    MolecularEntities collection of objects. It also facilitates unit conversion for each of the
    values in the property lists.  
    """
    
    def __init__(self):
        """
        Get this thing started.
        """
        
    def get_property_values_and_counts(self, mols_with_props):
        """
        Iterates through a list of objects returned by a list(db.collection.find()) search in the 
        MolecularEntities collection. Assumes that all objects in the list have associated properties
        in the 'Properties' field.
        
        Parameters:
            mols_with_props (list): list of MolecularEntities objects that have a 'Properties' field.
            
        Returns:
            val_df (pd.DataFrame): DataFrame object listing all the properties as rows and all their
                associated values as colums. Each cell is a tuple containing (value, units). If there
                are no associated units, then only the value is listed without the tuple.
                
            count_df (pd.DataFrame): DataFrame object containing all the properties as rows and a single
                column that lists the number of instances of that property were seen during parsing.
        """
        
        #dict of property:instance pairs
        prop_counts = {}
        #dict of property:[list of (value, unit) tuples] pairs
        prop_vals = {}

        for entity in mols_with_props:

            props = entity['Properties']

            #list of property dicts
            for entry in props:
                #property dicts where v is a list of dicts
                for k, v in entry.items():
                    #skip DOIs
                    if k != 'AssociatedDOI':
                        #initialize tracking of a property
                        if k not in prop_counts:
                            prop_counts[k] = 1

                            #only track values of properties, not spectra
                            if '_spectra' not in k:
                                prop_vals[k] = []
                                for entry_dict in v:
                                    if 'units' in entry_dict:
                                        property_value = (entry_dict['value'], entry_dict['units'])
                                    else:
                                        property_value = (entry_dict['value'])

                                    prop_vals[k].append(property_value)

                        #update a tracked property
                        if k in prop_counts:
                            prop_counts[k] += 1
                            #only track values of properties, not spectra
                            if '_spectra' not in k:

                                for entry_dict in v:
                                    if 'units' in entry_dict:
                                        property_value = (entry_dict['value'], entry_dict['units'])
                                    else:
                                        property_value = (entry_dict['value'])

                                    prop_vals[k].append(property_value)
                                    
        count_df = pd.DataFrame.from_dict(prop_counts, orient = 'index')
        val_df = pd.DataFrame.from_dict(prop_vals, orient = 'index')
        
        return val_df, count_df
    
    
    def convert_value_format(self, value):
        """
        ChemDataExtractor is built to recognize property values that can be listed as a single
        value, a range, a value with uncertainty, and more. While we can interpret that during reading,
        they are not directly convertable to numerical data. This function facilitates that conversion.
        
        Parameters:
            value (str): string representing property value in an article that was parsed 
                by ChemDataExtractor
                
        Returns:
            float_value (float): the numerical representation of `value`
        """
        float_value = 'None'

        if '≥' in value:
            value = value.replace('≥', '')

        if '≤' in value:
            value = value.replace('≤', '')

        if '>' in value:
            value = value.replace('>', '')

        if '<' in value:
            value = value.replace('<', '')

        if '~' in value:
            value = value.replace('~', '')

        if '∼' in value:
            value = value.replace('∼', '')

        if '±' in value:
            idx = value.index('±')
            value = value[:idx]

        #if a range is given, record the average
        if '–' in value:
            value = value.replace(' ', '')

            #make sure '-' is not a negative sign
            if value.count('–') == 1:
                vals = value.split('–')

                try:
                    vals.remove('')
                except:
                    pass

                vals = [float(x) for x in vals]

                if len(vals) == 1:
                    float_value = -vals[0]

                else:
                    float_value = sum(vals) / len(vals)

            if value.count('–') == 3:
                vals = value.split('–')
                vals = [-float(x) for x in vals]
                float_value = sum(vals)/len(vals)

        if '-' in value:
            value = value.replace(' ', '')

            #make sure '-' is not a negative sign
            if value.count('-') == 1:
                vals = value.split('-')
                try:
                    vals.remove('')
                except:
                    pass
                vals = [float(x) for x in vals]

                if len(vals) == 1:
                    float_value = -vals[0]

                else:
                    float_value = sum(vals) / len(vals)

            if value.count('-') == 3:
                vals = value.split('-')
                vals = [-float(x) for x in vals]
                float_value = sum(vals)/len(vals)

        if '−' in value:
            value = value.replace(' ', '')

            #make sure '-' is not a negative sign
            if value.count('−') == 1:
                vals = value.split('−')
                try:
                    vals.remove('')
                except:
                    pass
                vals = [float(x) for x in vals]

                if len(vals) == 1:
                    float_value = -vals[0]

                else:
                    float_value = sum(vals) / len(vals)

            if value.count('−') == 3:
                vals = value.split('−')
                vals = [-float(x) for x in vals]
                float_value = sum(vals)/len(vals)

        if '–' in value:
            value = value.replace(' ', '')
            #make sure '-' is not a negative sign
            if value.count('–') == 1:
                vals = value.split('–')

                try:
                    vals.remove('')
                except:
                    pass

                vals = [float(x) for x in vals]

                if len(vals) == 1:
                    float_value = -vals[0]

                else:
                    float_value = sum(vals) / len(vals)

            if value.count('-') == 3:
                vals = value.split('-')
                vals = [-float(x) for x in vals]
                float_value = sum(vals)/len(vals)

        if 'to' in value:
            value = value.replace(' ', '')
            vals = value.split('to')
            vals = [float(x) for x in vals]
            float_value = sum(vals)/len(vals)

        if 'and' in value:
            value = value.replace(' ', '')
            vals = value.split('and')
            float_value = [float(x) for x in vals]

        if float_value == 'None':
            float_value = float(value)
            
        if type(float_value) == str:
            float_value = float(float_value)
            
        return float_value
    
    
    def get_vals_and_units(self, row):
        """
        Takes in a single property row of the val_df and returns a list of the values and units.
        Because of the way values are reported in literature, some reformatting and parsing of value
        ranges, multiple instances, or approximate values are converted to single values. Uncertainties
        are dropped.
        
        Parameters:
            row (pd.Series): a Series object that contains all the (value, unit) tuples for a given
                property in the val_df that is returned by self.get_property_values_and_counts().
                
        Returns:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
        """
    
        row = row.dropna()

        row_cells = row.tolist()

        values = []
        units = []

        for cell in row_cells:
        #     print(cell)
            if type(cell) == tuple:
                str_value = cell[0]
                unit = cell[1]
            else:
                str_value = cell
                unit = None
                
            value = self.convert_value_format(str_value)
            
            if type(value) == list:
                values.append(value[0])
                values.append(value[1])
                units.append(unit)
                units.append(unit)
                
            else:
                values.append(value)
                units.append(unit)

        return values, units
    

    def convert_mol_weight_units(self, values, units):
        """
        This function converts everything in values to kDa (which == kg/mol), based on its
        corresponding units value.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []
        
        for val, un in zip(values, units):
            if un == 'Da' or un == 'g/mol':
                new_val = val / 1000
                new_un = 'kDa'

            if un == 'KDa' or un == 'kDa':
                new_val = val
                new_un = 'kDa'

            if un == 'kg/mol' or un == 'kg per mol' or un == 'kg mol-1':
                new_val = val
                new_un = 'kDa'

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_dispersity_units(self, values):
        """
        This function acts as a threshold to weed out unreasonable values. There are no units to
        convert because dispersity is inherently unitless.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []

        for val in values:
            
            if val > 100:
                pass
            
            else:
                scaled_v.append(val)

        return scaled_v
    

    def convert_voc_units(self, values, units):
        """
        This function converts everything in values to V, based on its
        corresponding units value.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if un == 'mV':
                new_val = val / 1000
                new_un = 'V'

            if un == 'eV':
                new_val = val
                new_un = 'V'
                
            if un == 'meV':
                new_val = val / 1000
                new_un = 'V'

            if un == 'V' or un == 'V.' or un == 'VOC':
                new_val = val
                new_un = 'V'
                
            if new_val > 10: #some mis-parsed (?) values have Voc listed as 100's of Volts
                new_val = new_val / 1000

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_pce_units(self, values, units):
        """
        This function ensures every element is a percent out of 100, rather than out of 1. Also filters
        out numbers beyond theoretical limits, which probably correspond to either reference numbers or
        IPCE values, rather than PCE values.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if un == None:
                new_val = val * 100
                new_un = '%'
                
            if val > 30:
                continue
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_ff_units(self, values, units):
        """
        This function acts as a threshold to weed out unreasonable values. There are no units to
        convert because fill factor is inherently unitless. Also ensures all values are out of 100,
        rather than out of 1.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if val < 0:
                continue
                
            if val > 100:
                continue
                
            if val < 1.0:
                new_val = val * 100
                new_un = un
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_jsc_units(self, values, units):
        """
        This function converts everything in values to mA/cm2, based on its
        corresponding units value.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if un == 'μA/cm2' or un == 'µA/cm2':
                new_val = val / 1000
                new_un = 'mA/cm2'
                
            if un == 'A/cm2':
                new_val = val * 1000
                new_un = 'mA/cm2'

            if un == 'mA/cm2':
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_temp_units(self, values, units):
        """
        This function converts everything in values to ºC, based on its
        corresponding units element.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if un == 'C' or un == 'ºC' or un == '°C':
                new_val = val
                new_un = un

            if un == 'F' or un == 'ºF' or un == '°F':
                new_val = (val - 32) * (5/9)
                new_un = 'C'
                
            if un == 'K':
                new_val = val - 273.15
                new_un = 'C'

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_band_gap_units(self, values, units):
        """
        This function acts as a threshold to weed out unreasonable values. Band gap values are
        consistently listed as eV
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if val > 10:
                continue
                
            if val < 0:
                new_val = val * -1
                new_un = un
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_orbital_level_units(self, values, units):
        """
        This function acts as a threshold to weed out unreasonable values. HOMO and LUMO values are
        consistently listed as eV
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if val > 10:
                continue
                
            if val < 0:
                new_val = val * -1
                new_un = un
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_corrosion_units(self, values, units):
        """
        This function ensures every element is a percent out of 100, rather than out of 1. Also filters
        out numbers beyond theoretical limits.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
                
            if val < 1.0:
                new_val = val * 100
                new_un = '%'
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_quantum_yield_units(self, values, units):
        """
        This function ensures every element is a percent out of 100, rather than out of 1. Also filters
        out numbers beyond theoretical limits.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
                
            if val < 1.0:
                new_val = val * 100
                new_un = '%'
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_fluorescence_lifetime_units(self, values, units):
        """
        This function ensures every element is in nanoseconds.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
                
            if un == 'μs':
                new_val = val * 1000
                new_un = 'ns'
                
            elif un == None:
                if val < 100:
                    new_val = val * 1000 #may not be correct...
                else:
                    new_val = val
                new_un = 'ns'
                
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u

    
    def convert_modulus_units(self, values, units):
        """
        This function converts everything in values to GPa, based on its
        corresponding units value.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
            if un == 'Pa':
                new_val = val / 1000000
                new_un = 'MPa'
                
            if un == 'kPa':
                new_val = val / 1000
                new_un = 'MPa'

            if un == 'GPa':
                new_val = val * 1000
                new_un = 'MPa'

            if un == 'MPa':
                new_val = val
                new_un = un
                
            if new_val < 1000:
                pass
            
            else:
                scaled_v.append(new_val)
                scaled_u.append(new_un)

        return scaled_v, scaled_u
    
    
    def convert_crystallinity_units(self, values, units):
        """
        This function ensures every element is a percent out of 100, rather than out of 1. Also filters
        out numbers beyond theoretical limits.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """
        scaled_v = []
        scaled_u = []

        for val, un in zip(values, units):
                
            if val < 1.0:
                new_val = val * 100
                new_un = '%'
                
            else:
                new_val = val
                new_un = un

            scaled_v.append(new_val)
            scaled_u.append(new_un)

        return scaled_v, scaled_u
        

    def convert_units(self, values, units, prop_type):
        """
        This function converts the values of a given row, based on the type of material property 
        it is. This is determined by `prop_type`, which is the index value for the row in val_df.
        Based on the prop_type, a different conversion function is called.
        
        Parameters:
            values (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus.
                
            units (list): list of strings, where each element represents the units for its 
                corresponding value in the values list.
                
        Returns:
            scaled_v (list): list of floats, where each element represents a property value identified
                by ChemDataExtractor in the corpus that is scaled to a common SI unit.
                
            scaled_u (list): list of strings, where each element represents the units for its 
                corresponding value in the values list. Should all be the same SI unit.
        """

        if prop_type == 'M_w' or prop_type == 'M_n':
            scaled_v, scaled_u = self.convert_mol_weight_units(values, units)

        if prop_type == 'dispersity':
            scaled_v = self.convert_dispersity_units(values)
            scaled_u = ['None' for x in scaled_v]

        if prop_type == 'voc':
            scaled_v, scaled_u = self.convert_voc_units(values, units)
                
        if prop_type == 'pce':
            scaled_v, scaled_u = self.convert_pce_units(values, units)
            
        if prop_type == 'ff':
            scaled_v, scaled_u = self.convert_ff_units(values, units)

        if prop_type == 'jsc':
            scaled_v, scaled_u = self.convert_jsc_units(values, units)

        if prop_type == 'HOMO_level' or prop_type == 'LUMO_level' or prop_type == 'fermi_energy':
            scaled_v, scaled_u = self.convert_orbital_level_units(values, units)

        if prop_type == 'glass_transitions' or prop_type == 'melting_points' or prop_type == 'boiling_point':
            scaled_v, scaled_u = self.convert_temp_units(values, units)
            
        if prop_type == 'band_gap':
            scaled_v, scaled_u = self.convert_band_gap_units(values, units)
            
        if prop_type == 'corrosion_inhibition':
            scaled_v, scaled_u = self.convert_corrosion_units(values, units)

        if prop_type == 'modulus':
            scaled_v, scaled_u = self.convert_modulus_units(values, units)

        if prop_type == 'electrochemical_potentials': #may need to be scaled based on electrode?
#             scaled_v, scaled_u = self.convert_ec_potential_units(values, units) #TODO
            scaled_v, scaled_u = values, units

        if prop_type == 'fluorescence_lifetimes':
            scaled_v, scaled_u = self.convert_fluorescence_lifetime_units(values, units)

        if prop_type == 'quantum_yields':
#             scaled_v, scaled_u = self.convert_quantum_yield_units(values, units)
            scaled_v, scaled_u = values, units
    
        if prop_type == 'crystallinity':
            scaled_v, scaled_u = self.convert_crystallinity_units(values, units)


        return scaled_v, scaled_u