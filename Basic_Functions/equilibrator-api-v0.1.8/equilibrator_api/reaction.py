import pkg_resources
import json
import re
import logging
import pandas as pd
from numpy import array, zeros, log, nan, inf
from numpy.linalg import inv

from equilibrator_api import settings
from equilibrator_api.compound import Compound


class Reaction(object):

    _json_str = pkg_resources.resource_stream('equilibrator_api',
                                              'data/cc_compounds.json')
    COMPOUND_JSON = json.loads(_json_str.read().decode('utf8'))

    # load formation energies from the JSON file
    COMPOUND_DICT = {}
    for cd in COMPOUND_JSON:
        kegg_id = cd.get('CID', 'unknown')
        COMPOUND_DICT[kegg_id] = cd

    REACTION_COUNTER = 0

    def __init__(self, kegg_id_to_coeff, rid=None):
        self.kegg_id_to_coeff = kegg_id_to_coeff

        # Create the relevant "Compound" objects and store in a dictionary
        self.kegg_id_to_compound = {}
        for kegg_id in self.kegg_id_to_coeff.keys():
            self.get_compound(kegg_id, create_if_missing=True)
        
        if rid is not None:
            self.reaction_id = rid
        else:
            self.reaction_id = 'R%05d' % Reaction.REACTION_COUNTER
            Reaction.REACTION_COUNTER += 1
        
    @property
    def kegg_ids(self):
        return self.kegg_id_to_coeff.keys()

    @property
    def kegg_ids_without_water(self):
        return set(self.kegg_ids) - set(['C00001'])

    def is_empty(self):
        return len(self.kegg_id_to_coeff) == 0

    def get_coeff(self, kegg_id):
        return self.kegg_id_to_coeff.get(kegg_id, 0)

    def get_compound(self, kegg_id, create_if_missing=False):
        if create_if_missing and (kegg_id not in self.kegg_id_to_compound):
            if kegg_id not in Reaction.COMPOUND_DICT:
                raise KeyError('%s is an unknown Compound ID' % kegg_id)
            compound = Compound(Reaction.COMPOUND_DICT[kegg_id])
            self.kegg_id_to_compound[kegg_id] = compound
            return compound
        else:
            return self.kegg_id_to_compound.get(kegg_id, None)

    def dG0_prime(self, pH=settings.DEFAULT_PH, pMg=settings.DEFAULT_PMG,
                  ionic_strength=settings.DEFAULT_IONIC_STRENGTH):
        """
            Calculate the standard dG'0 of reaction.
            Arguments:
                pH, pMg, ionic_strength - aqueous conditions
        """
        dG0_r_prime = 0
        for kegg_id in self.kegg_ids:
            coeff = self.get_coeff(kegg_id)
            compound = self.get_compound(kegg_id)
            dG0_f_prime = compound.dG0_prime(pH, pMg, ionic_strength)
            if dG0_f_prime is None:
                return None
            dG0_r_prime += coeff * dG0_f_prime
        return dG0_r_prime
    
    def dG_correction(self, kegg_id_to_conc):
        """
            Calculate the concentration adjustment in the dG' of reaction.
            Arguments:
                kegg_id_to_conc - a dictionary mapping KEGG compound ID 
                                 to concentration in M (default is 1M)
        """
        kegg_ids = set(self.kegg_ids_without_water).intersection(
            kegg_id_to_conc.keys())
        dG_correction = sum([self.get_coeff(c) * log(kegg_id_to_conc[c])
                             for c in kegg_ids])
        return settings.RT * dG_correction
        
    def dG_prime(self, kegg_id_to_conc, 
                 pH=settings.DEFAULT_PH, pMg=settings.DEFAULT_PMG,
                 ionic_strength=settings.DEFAULT_IONIC_STRENGTH):
        """
            Calculated the concentration adjusted dG' of reaction.
            Arguments:
                kegg_id_to_conc - a dictionary mapping KEGG compound ID 
                                 to concentration in M (default is 1M)
                pH, pMg, ionic_strength - aqueous conditions
        """
        dG0_prime = self.dG0_prime(pH=pH, pMg=pMg,
                                   ionic_strength=ionic_strength)
        return dG0_prime + self.dG_correction(kegg_id_to_conc)

    def _GetSumCoeff(self):
        """
            Calculate the sum of all coefficients (excluding water).
            This is useful for shifting the dG'0 to another set of standard
            concentrations (e.g. 1 mM)
        """
        sum_coeff = sum(map(self.get_coeff, self.kegg_ids_without_water))
        return sum_coeff

    def _GetAbsSumCoeff(self):
        """
            Calculate the sum of all coefficients (excluding water) in
            absolute value.
            This is useful for calculating the reversibility index.
        """
        abs_sum_coeff = sum(map(lambda cid: abs(self.get_coeff(cid)),
                                self.kegg_ids_without_water))
        return abs_sum_coeff     

    def dGm_correction(self):
        """
            Calculate the dG' in typical physiological concentrations (1 mM)
        """
        return settings.RT * self._GetSumCoeff() * log(1e-3)

    def dGm_prime(self):
        """
            Calculate the dG' in typical physiological concentrations (1 mM)
        """
        return self.dG0_prime() + self.dGm_correction()

    def reversibility_index(self, pH=settings.DEFAULT_PH, pMg=settings.DEFAULT_PMG,
                            ionic_strength=settings.DEFAULT_IONIC_STRENGTH):
        """
            Calculates the reversiblity index according to Noor et al. 2012:
            https://doi.org/10.1093/bioinformatics/bts317

            Returns:
                ln_RI - the natural log of the RI
        """
        dG0_prime = self.dG0_prime(pH, pMg, ionic_strength)
        return self.calculate_reversibility_index_from_dG0_prime(dG0_prime)

    def calculate_reversibility_index_from_dG0_prime(self, dG0_prime):
        """
            Calculates the reversiblity index according to Noor et al. 2012:
            https://doi.org/10.1093/bioinformatics/bts317

            Returns:
                ln_RI - the natural log of the RI
        """
        sum_coeff = self._GetSumCoeff()
        abs_sum_coeff = self._GetAbsSumCoeff()
        if abs_sum_coeff == 0:
            return inf
        dGm_prime = dG0_prime + settings.RT * sum_coeff * log(1e-3)
        ln_RI = (2.0 / abs_sum_coeff) * dGm_prime / settings.RT
        return ln_RI

    @staticmethod
    def parse_formula_side(s):
        """
            Parses the side formula, e.g. '2 C00001 + C00002 + 3 C00003'
            Ignores stoichiometry.

            Returns:
                The set of CIDs.
        """
        if s.strip() == "null":
            return {}

        compound_bag = {}
        for member in re.split('\s+\+\s+', s):
            tokens = member.split(None, 1)  # check for stoichiometric coeff
            if len(tokens) == 0:
                continue
            if len(tokens) == 1:
                amount = 1
                key = member
            else:
                try:
                    amount = float(tokens[0])
                except ValueError:
                    raise ValueError('could not parse the reaction side: %s'
                                     % s)
                key = tokens[1]
            compound_bag[key] = compound_bag.get(key, 0) + amount

        return compound_bag

    @staticmethod
    def parse_formula(formula, name_to_cid=None, rid=None):
        """
            Parses a two-sided formula such as: 2 C00001 = C00002 + C00003
            
            Args:
                formula     - a string representation of the chemical formula
                name_to_cid - (optional) a dictionary mapping names to KEGG IDs

            Return:
                The set of substrates, products and the reaction direction
        """
        tokens = []
        for arrow in settings.POSSIBLE_REACTION_ARROWS:
            if formula.find(arrow) != -1:
                tokens = formula.split(arrow, 2)
                break

        if len(tokens) < 2:
            raise ValueError('Reaction does not contain an allowed arrow sign:'
                             ' %s' % (arrow, formula))

        left = tokens[0].strip()
        right = tokens[1].strip()

        sparse_reaction = {}
        for cid, count in Reaction.parse_formula_side(left).items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) - count

        for cid, count in Reaction.parse_formula_side(right).items():
            sparse_reaction[cid] = sparse_reaction.get(cid, 0) + count

        # remove compounds that are balanced out in the reaction,
        # i.e. their coefficient is 0
        sparse_reaction = dict(filter(lambda x: x[1] != 0,
                                      sparse_reaction.items()))
        if name_to_cid is not None:
            # replace the names of the metabolites with their KEGG IDs
            # using this dictionary
            sparse_reaction = \
                dict(zip(map(name_to_cid.get, sparse_reaction.keys()),
                     sparse_reaction.values()))
            
        if 'C00080' in sparse_reaction:
            sparse_reaction.pop('C00080')
        
        return Reaction(sparse_reaction, rid=rid)

    @staticmethod
    def write_compound_and_coeff(compound_id, coeff):
        if coeff == 1:
            return compound_id
        else:
            return "%g %s" % (coeff, compound_id)

    def write_formula(self):
        """String representation."""
        left = []
        right = []
        for kegg_id in self.kegg_ids:
            coeff = self.get_coeff(kegg_id)
            if coeff < 0:
                left.append(Reaction.write_compound_and_coeff(kegg_id, -coeff))
            elif coeff > 0:
                right.append(Reaction.write_compound_and_coeff(kegg_id, coeff))
        return "%s %s %s" % (' + '.join(left), '=', ' + '.join(right))

    def __str__(self):
        return self.write_formula()
    
    def __repr__(self):
        return '<Reaction %s at 0x%x>' % (self.reaction_id, id(self))

    def _get_element_matrix(self):
        # gather the "atom bags" of all compounds in a list 'atom_bag_list'
        elements = set()
        atom_bag_list = []
        for kegg_id in self.kegg_ids:
            comp = self.get_compound(kegg_id)
            atom_bag = comp.get_atom_bag()
            if atom_bag is None:
                return None, None
            else:
                elements = elements.union(atom_bag.keys())
            atom_bag_list.append(atom_bag)
        elements = sorted(elements)

        # create the elemental matrix, where each row is a compound and each
        # column is an element (or e-)
        Ematrix = array(zeros((len(atom_bag_list), len(elements))))
        for i, atom_bag in enumerate(atom_bag_list):
            if atom_bag is None:
                Ematrix[i, :] = nan
            else:
                for j, elem in enumerate(elements):
                    Ematrix[i, j] = atom_bag.get(elem, 0)
        return elements, Ematrix

    def _get_reaction_atom_balance(self):
        cids = list(self.kegg_ids)
        coeffs = array(list(map(self.get_coeff, cids)))

        elements, Ematrix = self._get_element_matrix()
        if elements is None:
            return None
        conserved = coeffs @ Ematrix
        conserved = conserved.round(3)

        atom_balance_dict = dict([(e, c) for (e, c) in
                                  zip(elements, conserved.flat) if (c != 0)])

        return atom_balance_dict

    def check_half_reaction_balancing(self):
        """
            Returns:
                The number of electrons that are 'missing' in the half-reaction
                or None if the reaction is not atomwise-balanced.
        """
        atom_balance_dict = self._get_reaction_atom_balance()
        if atom_balance_dict is None:
            return None
        
        n_e = atom_balance_dict.pop('e-', 0)
        if not self._check_balancing(atom_balance_dict):
            return None
        else:
            return n_e

    def check_full_reaction_balancing(self):
        """
            Returns:
                True iff the reaction is balanced for all elements
                (excluding H)
        """
        atom_balance_dict = self._get_reaction_atom_balance()
        if atom_balance_dict is None:
            return None

        return self._check_balancing(atom_balance_dict)

    def _check_balancing(self, atom_balance_dict):
        """
            Use for checking if all elements are conserved.

            Returns:
                An atom_bag of the differences between the sides of the
                reaction. E.g. if there is one extra C on the left-hand
                side, the result will be {'C': -1}.
        """
        if nan in atom_balance_dict.values():
            warning_str = 'cannot test reaction balancing because of ' + \
                          'unspecific compound formulas: %s' % \
                          self.write_formula()
            raise ValueError(warning_str)

        # if there are unbalanced elements, write a full report
        if len(atom_balance_dict) == 0:
            return True
        logging.warning('unbalanced reaction: %s' % self.write_formula())
        for elem, c in atom_balance_dict.items():
            if c != 0:
                logging.warning('there are %d more %s atoms on the '
                                'right-hand side' % (c, elem))
        return False

    def balance_by_oxidation(self):
        """
            Takes an unbalanced reaction and converts it into an oxidation
            reaction, by adding O2, H2O, NH3, Pi, and CO2 to both sides.
        """
        balancing_ids = ['C00001', 'C00007', 'C00009', 'C00011', 'C00014']

        S = pd.DataFrame(columns=balancing_ids)
        for kegg_id in balancing_ids:
            comp = self.get_compound(kegg_id, create_if_missing=True)
            atom_bag = comp.get_atom_bag()
            for atom, coeff in atom_bag.items():
                S.at[atom, kegg_id] = coeff
        S.fillna(0, inplace=True)
        
        balancing_atoms = S.index
        
        atom_bag = self._get_reaction_atom_balance()
        if atom_bag is None:
            logging.warning('Cannot balance this reaction due to'
                            ' missing chemical formulas')
            return self
        atom_vector = array(list(map(lambda a: atom_bag.get(a, 0),
                                      balancing_atoms)), ndmin=2).T
        
        other_atoms = set(atom_bag.keys()).difference(balancing_atoms)
        if other_atoms:
            raise ValueError('Cannot oxidize compounds with these atoms: '
                             '%s\nFormula is %s' %
                             (str(other_atoms), self.write_formula()))
            
        imbalance = inv(S) @ atom_vector
        
        for kegg_id, coeff in zip(balancing_ids, imbalance.flat):
            self.kegg_id_to_coeff[kegg_id] = \
                self.kegg_id_to_coeff.get(kegg_id, 0) - coeff
            
        return self
        
    @staticmethod
    def get_oxidation_reaction(kegg_id):
        """
            Generate a Reaction object which represents the oxidation reaction
            of this compound using O2. If there are N atoms, the product must 
            be NH3 (and not N2) to represent biological processes.
            Other atoms other than C, N, H, and O will raise an exception.
        """
        return Reaction({kegg_id: -1}).balance_by_oxidation()
    

if __name__ == '__main__':
    import sys
    
    r = Reaction.get_oxidation_reaction('C00031')
    print(r.write_formula())
    print('standard oxidation energy of glucose: %.2f kJ/mol' % r.dG0_prime())
    
    r = Reaction.get_oxidation_reaction('C00064')
    print(r.write_formula())
    print('standard oxidation energy of acetate: %.2f kJ/mol' % r.dG0_prime())
    
    r = Reaction.parse_formula('C00031 = ').balance_by_oxidation()
    print(r.write_formula())
    print('standard oxidation energy of glucose: %.2f kJ/mol' % r.dG0_prime())
    
    print('oxidation energy of 1 mM glucose: %.2f kJ/mol' % r.dG_prime({'C00031': 1e-3}))

    print('\nNow, trying to use a compound with an unspecific formula:')
    sys.stdout.flush()
    r = Reaction.get_oxidation_reaction('C04619')
    print(r.write_formula())
    print('standard oxidation energy of (3R)-3-Hydroxydecanoyl-[acyl-carrier protein]: %s kJ/mol' % r.dG0_prime())
    
