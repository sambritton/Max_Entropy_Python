#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 15:36:47 2017

@author: noore

A stand-alone version of Component Contribution that can calculate the
Delta-Gr'0 of any reaction (with KEGG notation, i.e. whose reactants are
already cached in our database), at a given pH and I.
"""

import argparse
import logging
import sys
from equilibrator_api import Reaction, ComponentContribution


def MakeParser():
    parser = argparse.ArgumentParser(
        description=('Estimate the Gibbs energy of a reaction. For example,'
                     'the following calculates dGr0 for ATP hydrolysis '
                     'at pH 6: calc_dGr0.py --ph 6 "C00002 + C00001 = '
                     'C00008 + C00009"'))
    parser.add_argument('--ph', type=float, help='pH level', default=7.0)
    parser.add_argument('--i', type=float,
                        help='ionic strength in M',
                        default=0.1)
    parser.add_argument('reaction', type=str, help='reaction in KEGG notation')
    return parser


###############################################################################
parser = MakeParser()
args = parser.parse_args()

logging.getLogger().setLevel(logging.WARNING)

sys.stderr.write('pH = %.1f\n' % args.ph)
sys.stderr.write('I = %.1f M\n' % args.i)
sys.stderr.write('Reaction: %s\n' % args.reaction)
sys.stderr.flush()

# parse the reaction
try:
    reaction = Reaction.parse_formula(args.reaction)
except ValueError as e:
    logging.error(str(e))
    sys.exit(-1)

equilibrator = ComponentContribution(pH=args.ph, ionic_strength=args.i)

n_e = reaction.check_half_reaction_balancing()
if n_e is None:
    logging.error('reaction is not chemically balanced')
    sys.exit(-1)
elif n_e == 0:
    dG0_prime, dG0_uncertainty = equilibrator.dG0_prime(reaction)
    sys.stdout.write(u'\u0394G\'\u00B0 = %.1f \u00B1 %.1f kJ/mol\n' %
                     (dG0_prime, dG0_uncertainty))

    ln_RI = equilibrator.reversibility_index(reaction)
    sys.stdout.write(u'ln(Reversibility Index) = %.1f\n' % ln_RI)

else:  # treat as a half-reaction
    logging.warning('This reaction isn\'t balanced, but can still be treated'
                    ' as a half-reaction')
    E0_prime_mV, E0_uncertainty = equilibrator.E0_prime(reaction)
    sys.stdout.write(
        u'E\'\u00B0 = %.1f \u00B1 %.1f mV\n' %
        (E0_prime_mV, E0_uncertainty))

sys.stdout.flush()
sys.stderr.write(r'* the range represents the 95% confidence interval'
                 ' due to Component Contribution estimation uncertainty\n')
