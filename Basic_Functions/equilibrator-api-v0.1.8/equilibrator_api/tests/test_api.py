#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:22:11 2017

@author: noore
"""

import unittest
import os
import warnings
import inspect

from equilibrator_api import Reaction, ComponentContribution, ReactionMatcher
from equilibrator_api import Pathway

TEST_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class TestReactionParsing(unittest.TestCase):
    
    def test_atp_hydrolysis(self):
        warnings.simplefilter('ignore', ResourceWarning)
        
        formula = ' C00002 + C00001  <= C00008 +   C00009'
        kegg_ids = set(('C00002', 'C00001', 'C00008', 'C00009'))
        try:
            reaction = Reaction.parse_formula(formula)
        except ValueError as e:
            self.fail('unable to parse the formula\n' + str(e))
            
        self.assertSetEqual(set(reaction.kegg_ids), kegg_ids)
        for kegg_id in kegg_ids:
            self.assertIsNotNone(reaction.get_compound(kegg_id))
            self.assertNotEqual(reaction.get_coeff(kegg_id), 0)
        self.assertIsNone(reaction.get_compound('C00003'))
        self.assertEqual(reaction.get_coeff('C00003'), 0)
        
    def test_reaction_balancing(self):
        warnings.simplefilter('ignore', ResourceWarning)

        kegg_id_to_coeff = {'C00011' : -1, 'C00001' : -1, 'C01353' : 1}
        reaction = Reaction(kegg_id_to_coeff)
        self.assertTrue(reaction.check_full_reaction_balancing())

        kegg_id_to_coeff = {'C00036' : -1, 'C00149' : 1}  # oxaloacetate = malate
        reaction = Reaction(kegg_id_to_coeff)
        self.assertAlmostEqual(reaction.check_half_reaction_balancing(), 2.0)

        kegg_id_to_coeff = {'C00031' : -1, 'C00469' : 2}  # missing two CO2
        reaction = Reaction(kegg_id_to_coeff)
        self.assertDictEqual(reaction._get_reaction_atom_balance(),
                             {'O': -4, 'C': -2, 'e-': -44})


    def test_gibbs_energy(self):
        warnings.simplefilter('ignore', ResourceWarning)

        kegg_id_to_coeff = {'C00002' : -1, 'C00001' : -1,
                            'C00008' :  1, 'C00009' :  1} # ATP + H2O = ADP + Pi
        kegg_id_to_conc  = {'C00002' : 1e-3,
                            'C00009' :  1e-4}
        reaction = Reaction(kegg_id_to_coeff)

        cc = ComponentContribution(pH=7.0, ionic_strength=0.1)
        dG0_prime, dG0_uncertainty = cc.dG0_prime(reaction)
        
        self.assertAlmostEqual(dG0_prime, -26.4, 1)
        self.assertAlmostEqual(dG0_uncertainty, 0.6, 1)
        
        dG_prime, _ = cc.dG_prime(reaction, kegg_id_to_conc)
        self.assertAlmostEqual(dG_prime, -32.1, 1)

        dGm_prime, _ = cc.dGm_prime(reaction)
        self.assertAlmostEqual(dGm_prime, -43.5, 1)

    def test_reduction_potential(self):
        warnings.simplefilter('ignore', ResourceWarning)

        kegg_id_to_coeff = {'C00036' : -1, 'C00149' : 1}  # oxaloacetate = malate
        reaction = Reaction(kegg_id_to_coeff)

        cc = ComponentContribution(pH=7.0, ionic_strength=0.1)
        E0_prime_mV, E0_uncertainty = cc.E0_prime(reaction)
        
        self.assertAlmostEqual(E0_prime_mV, -175.2, 1)
        self.assertAlmostEqual(E0_uncertainty, 5.3, 1)

    def test_mdf(self):
        warnings.simplefilter('ignore', ResourceWarning)

        sbtab_fname = os.path.join(TEST_DIR, 'pathway_ethanol_SBtab.tsv')
        pp = Pathway.from_sbtab(sbtab_fname)
        mdf_res = pp.calc_mdf().mdf_result
        
        self.assertAlmostEqual(mdf_res.mdf, 1.69, 2)
        self.assertAlmostEqual(mdf_res.max_total_dG, -159.05, 2)
        self.assertAlmostEqual(mdf_res.min_total_dG, -181.05, 2)


        self.assertAlmostEqual(mdf_res.reaction_prices[0, 0], 0.0, 2)
        self.assertAlmostEqual(mdf_res.reaction_prices[3, 0], 0.25, 2)
        self.assertAlmostEqual(mdf_res.reaction_prices[4, 0], 0.25, 2)
        self.assertAlmostEqual(mdf_res.reaction_prices[5, 0], 0.50, 2)
        
        self.assertAlmostEqual(mdf_res.compound_prices[1, 0], 0.0, 2)
        self.assertAlmostEqual(mdf_res.compound_prices[2, 0], 1.24, 2)
        self.assertAlmostEqual(mdf_res.compound_prices[3, 0], -1.24, 2)
        
    def test_reaction_matcher(self):
        warnings.simplefilter('ignore', ResourceWarning)

        formulas = [('ATP + H2O <=> ADP + Phosphate',
                     {'C00002': -1, 'C00001': -1, 'C00008': 1, 'C00009': 1}),
                    ('O2 + 2 NADH <=> 2 NAD+ + 2 H2O',
                     {'C00007': -1, 'C00004': -2, 'C00003': 2, 'C00001': 2}),
                    ('O2 + 2 NADH <=> 2 NAD+ + 2 H2O',
                     {'C00007': -1, 'C00004': -2, 'C00003': 2, 'C00001': 2}),
                    ('ATP + D-arbino-hexulose <=> ADP + D-Fructose-1-phophte', # with some typos
                     {'C00002': -1, 'C00095': -1, 'C00008': 1, 'C01094': 1}),
                    ]
        
        rm = ReactionMatcher()
        for plaintext, kegg_id_to_coeff in formulas:
            rxn = rm.match(plaintext)
            if rxn is None:
                self.fail('unable to parse the plaintext formula\n' + plaintext)
            else:
                self.assertDictEqual(rxn.kegg_id_to_coeff, kegg_id_to_coeff)
        

if __name__ == '__main__':
    unittest.main()
