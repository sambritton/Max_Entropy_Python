#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:26:31 2017

@author: noore
"""
import csv
import logging
from numpy import array, eye, log, zeros
from scipy import linalg
from sbtab import SBtab
from equilibrator_api.reaction import Reaction
from equilibrator_api.bounds import DEFAULT_BOUNDS, Bounds
from equilibrator_api.max_min_driving_force import PathwayMDFData
from equilibrator_api.thermo_models import PathwayThermoModel
from equilibrator_api.component_contribution import ComponentContribution
from equilibrator_api.settings import RT, DEFAULT_PH, DEFAULT_IONIC_STRENGTH

class PathwayParseError(Exception):
    pass


class InvalidReactionFormula(PathwayParseError):
    pass


class UnbalancedReaction(PathwayParseError):
    pass


class ViolatesFirstLaw(PathwayParseError):
    pass


class Pathway(object):
    """A pathway parsed from user input.

    Designed for checking input prior to converting to a stoichiometric model.
    """

    def __init__(self, reactions, fluxes, dG0_r_primes, name_to_cid=None,
                 bounds=None, pH=DEFAULT_PH,
                 ionic_strength=DEFAULT_IONIC_STRENGTH):
        """Initialize.

        Args:
            reactions: a list of gibbs.reaction.Reaction objects.
            fluxes: numpy.array of relative fluxes in same order as reactions.
            dG0_r_primes: reaction energies.
            bounds: bounds on metabolite concentrations.
                Uses default bounds if None provided.
            pH: (optional) specify the pH at which the dG values are calculated
            ionic_strength: (optional) specify the I at which the dG values are calculated
        """
        assert len(reactions) == len(fluxes)
        assert len(reactions) == len(dG0_r_primes)

        self.reactions = reactions
        self.fluxes = array(fluxes, ndmin=2)
        self.dG0_r_prime = array(dG0_r_primes, ndmin=2)
        dGm_corr = array([r.dGm_correction() for r in self.reactions], ndmin=2)
        self.dGm_r_prime = self.dG0_r_prime + dGm_corr
        self.bounds = bounds or DEFAULT_BOUNDS

        self.S, self.compound_kegg_ids = self._build_stoichiometric_matrix()

        if name_to_cid is not None:
            # invert the dictionary and convert the cids back to names
            cid_to_name = dict(zip(name_to_cid.values(), name_to_cid.keys()))
            self.compound_names = list(map(cid_to_name.get,
                                           self.compound_kegg_ids))
        else:
            self.compound_names = list(self.compound_kegg_ids)
            
        nr, nc = self.S.shape

        # dGr should be orthogonal to nullspace of S
        # If not, dGr is not contained in image(S) and then there
        # is no consistent set of dGfs that generates dGr and the
        # first law of thermo is violated by the model.
        Spinv = linalg.pinv(self.S.T)
        null_proj = eye(self.S.shape[1]) - self.S.T @ Spinv
        projected = null_proj * array(self.dG0_r_prime).T
        if (projected > 1e-8).any():
            raise ViolatesFirstLaw(
                'Supplied reaction dG values are inconsistent '
                'with the stoichiometric matrix.')

    @classmethod
    def from_csv_file(cls, f, bounds=None, pH=None, ionic_strength=None):
        """Returns a pathway parsed from an input file.

        Caller responsible for closing f.

        Args:
            f: file-like object containing CSV data describing the pathway.
        """
        reactions = []
        fluxes = []

        for row in csv.DictReader(f):
            rxn_formula = row.get('ReactionFormula')

            flux = float(row.get('Flux', 0.0))
            logging.debug('formula = %f x (%s)', flux, rxn_formula)

            rxn = Reaction.parse_formula(rxn_formula)
            rxn.check_full_reaction_balancing()

            reactions.append(rxn)
            fluxes.append(flux)

        equilibrator = ComponentContribution(pH=pH, ionic_strength=ionic_strength)
        dG0_r_primes, dG0_uncertainties = zip(*map(equilibrator.dG0_prime, reactions))
        dG0_r_primes = list(dG0_r_primes)
        return Pathway(reactions, fluxes, dG0_r_primes, bounds=bounds)

    def _get_compounds(self):
        """Returns a dictionary of compounds by KEGG ID."""
        compounds = {}
        for r in self.reactions:
            for cw_coeff in r.kegg_ids:
                c = cw_coeff.compound
                compounds[c.kegg_id] = c
        return compounds

    def _build_stoichiometric_matrix(self):
        """Builds a stoichiometric matrix.

        Returns:
            Two tuple (S, compounds) where compounds is the KEGG IDs of the compounds
            in the order defining the column order of the stoichiometric matrix S.
        """
        compounds = set()
        for r in self.reactions:
            compounds.update(r.kegg_ids)
        compounds = sorted(compounds)
        
        smat = zeros((len(compounds), len(self.reactions)))
        for j, r in enumerate(self.reactions):
            for i, c in enumerate(compounds):
                smat[i, j] = r.get_coeff(c)

        return smat, compounds

    def calc_mdf(self):
        dGs = array(self.dG0_r_prime, ndmin=2).T
        model = PathwayThermoModel(self.S, self.fluxes, dGs,
                                   self.compound_kegg_ids,
                                   concentration_bounds=self.bounds)
        mdf = model.mdf_result
        return PathwayMDFData(self, mdf)

    def print_reactions(self):
        for f, r in zip(self.fluxes, self.reactions):
            print('%sx %s' % (f, r.write_formula()))

    @classmethod
    def from_sbtab(self, file_name):
        """
            read the sbtab file (can be a filename or file handel)
            and use it to initialize the Pathway
        """
        sbtabdoc = SBtab.read_csv(file_name, 'pathway')

        table_ids = ['ConcentrationConstraint', 'Reaction', 'RelativeFlux',
                     'Parameter']
        dfs = []
        
        for table_id in table_ids:
            sbtab = sbtabdoc.get_sbtab_by_id(table_id)
            if sbtab is None:
                logging.error('%s contains the following TableIDs: %s' % 
                    (file_name,
                     ', '.join(map(lambda s: s.table_id, sbtabdoc.sbtabs))))
                raise PathwayParseError('The SBtabDocument must have a table '
                                        'with the following TableID: %s'
                                        % table_id)
            dfs.append(sbtab.to_data_frame())
        
        bounds_df, reaction_df, flux_df, keqs_df = dfs
    
        bounds_unit = sbtabdoc.get_sbtab_by_id(
            'ConcentrationConstraint').get_attribute('Unit')
        bounds = Bounds.from_dataframe(bounds_df, bounds_unit)

        name_to_cid = dict(
            zip(bounds_df['Compound'],
                bounds_df['Compound:Identifiers:kegg.compound']))

        reactions = []
        for _, row in reaction_df.iterrows():
            rxn = Reaction.parse_formula(row['ReactionFormula'], name_to_cid,
                                         rid=row['ID'])
            rxn.check_full_reaction_balancing()
            reactions.append(rxn)

        reaction_ids = reaction_df['ID']
        fluxes = flux_df[flux_df['QuantityType'] == 'flux']
        reaction_fluxes = dict(zip(fluxes['Reaction'], fluxes['Value']))
        fluxes_ordered = [float(reaction_fluxes[rid]) for rid in reaction_ids]

        # grab rows containing keqs.
        keqs = keqs_df[keqs_df['QuantityType'] == 'equilibrium constant']
        reaction_keqs = dict(zip(keqs['Reaction'], keqs['Value']))
        dgs = [-RT * log(float(reaction_keqs[rid]))
               for rid in reaction_ids]

        # Manually set the delta G values on the reaction objects
        for dg, rxn in zip(dgs, reactions):
            rxn._dg0_prime = dg

        pH = sbtabdoc.get_sbtab_by_id(
            'Parameter').get_attribute('pH')
        ionic_strength = sbtabdoc.get_sbtab_by_id(
            'Parameter').get_attribute('IonicStrength')
        pp = Pathway(reactions, fluxes_ordered, dgs,
                     name_to_cid=name_to_cid,
                     bounds=bounds, pH=pH, ionic_strength=ionic_strength)
        return pp
