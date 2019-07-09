#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:01:52 2017

@author: noore
"""

import logging
import re
from numpy import sqrt
from equilibrator_api import settings
from numpy import array, zeros, logaddexp

class Species(object):

    def __init__(self, d):
        try:
            self.dG0_f = d['dG0_f']
            self.phase = d['phase']
            self.nH = d.get('nH', 0)
            self.nMg = d.get('nMg', 0)
            self.z = d.get('z', 0)
        except KeyError:
            raise KeyError('missing value in JSON: ' + str(d))

    def ddG_prime(self, pH, pMg, I):
        """
            Transform this individual estimate to difference conditions.
        """

        sqrt_I = sqrt(I)
        ddG_prime = 0

        # add the potential related to the pH
        if self.nH > 0:
            ddG_prime += self.nH * settings.RTlog10 * pH

        # add the potential related to the ionic strength
        ddG_prime -= settings.DEBYE_HUECKLE_A * (self.z ** 2 - self.nH) * sqrt_I / \
            (1.0 + settings.DEBYE_HUECKLE_B * sqrt_I)

        # add the potential related to the Mg ions
        if self.nMg > 0:
            ddG_prime += self.nMg * (settings.RTlog10 * pMg - settings.MG_FORMATION_ENERGY)

        return ddG_prime

    def dG0_prime(self, pH, pMg, I):
        """
            Transform this individual estimate to difference conditions.
        """
        dG0_f_prime = self.dG0_f + self.ddG_prime(pH, pMg, I)
        logging.info('nH = %2d, nMg = %2d, z = %2d, dG0_f = %6.1f'
                     ' -> dG\'0_f = %6.1f' %
                     (self.nH, self.nMg, self.z, self.dG0_f, dG0_f_prime))
        return dG0_f_prime


class Compound(object):

    def __init__(self, d):
        self.inchi = d.get('InChI', '')
        self.kegg_id = d['CID']
        self.compound_index = d.get('compound_index', -1)
        self.group_vector = d.get('group_vector', None)
        self.formula = d.get('formula', '')
        self.mass = d.get('mass', -1)
        self.num_electrons = d.get('num_electrons', 0)
        if 'pmap' in d:
            self.no_dg_explanation = d.get('error', '')
            self.source = d['pmap'].get('source', '')
            self.species_list = list(map(Species,
                                         d['pmap'].get('species', [])))
        else:
            logging.warning('Could not find a pmap for: ' + self.kegg_id)
            self.no_dg_explanation = 'KEGG compounds with unspecific structure'
            self.species_list = []

    def __repr__(self):
        return '<equilibrator_api.compound.Compound %s>' % self.kegg_id

    def get_stoich_vector(self, Nc):
        x = array(zeros((Nc, 1)))
        i = self.compound_index
        if i is None:
            raise Exception('could not find index for ' + self.kegg_id)
        x[i, 0] = 1
        return x

    def get_group_incidence_vector(self, Ng):
        g = array(zeros((Ng, 1)))
        gv = self.group_vector
        if gv is None:
            raise Exception('could not find group vector for ' + self.kegg_id)
        for g_ind, g_count in gv:
            g[g_ind, 0] += g_count
        return g

    def dG0_prime(self, pH, pMg, I):
        """
            Get a detla-deltaG estimate for this group of species.
            I.e., this is the difference between the dG0 and the dG'0, which
            only depends on the pKa of the pseudoisomers, but not on their
            formation energies.

            Args:
                pH  - the pH to estimate at.
                pMg - the pMg to estimate at.
                I   - the ionic strength to estimate at.

            Returns:
                The estimated delta G in the given conditions or None.
        """
        # Compute per-species transforms, scaled down by R*T.
        dG0_prime_vec = array(list(map(lambda s: s.dG0_prime(pH, pMg, I),
                                       self.species_list)))

        # Numerical issues: taking a sum of exp(v) for |v| quite large.
        # Use the fact that we take a log later to offset all values by a
        # constant (the minimum value).
        if len(dG0_prime_vec) > 0:
            dG0_f_prime = -settings.RT * logaddexp.reduce((-1.0 / settings.RT) * dG0_prime_vec)
        else:
            logging.warning(self.no_dg_explanation)
            return None
        
        logging.info('KEGG_ID = %s, dG\'0_f = %.1f' %
                     (self.kegg_id, dG0_f_prime))
        return dG0_f_prime

    def get_atom_bag(self):
        if not self.formula:
            logging.warning('Cannot obtain the bag of atoms for %s, as it has'
                            ' an unspecific chemical formula' % self.kegg_id)
            return None
        
        atom_bag = {}
        for mol_formula_times in self.formula.split('.'):
            for times, mol_formula in re.findall('^(\d+)?(\w+)',
                                                 mol_formula_times):
                if not times:
                    times = 1
                else:
                    times = int(times)
                for atom, count in re.findall("([A-Z][a-z]*)([0-9]*)",
                                              mol_formula):
                    if count == '':
                        count = 1
                    else:
                        count = int(count)
                    atom_bag[atom] = atom_bag.get(atom, 0) + count * times
        atom_bag['e-'] = self.num_electrons

        # we ignore protons (hydrogens) since their number depends on the
        # protonation level, and is not a constant
        if 'H' in atom_bag:
            atom_bag.pop('H')
        return atom_bag
