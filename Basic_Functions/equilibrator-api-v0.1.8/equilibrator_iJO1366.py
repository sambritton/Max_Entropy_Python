# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25th 2015

@author: flamholz
"""

from equilibrator_api import ComponentContribution, Reaction
import logging
import argparse
import csv
from numpy import sqrt, nan
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculate potentials for a number of reactions.')
    parser.add_argument(
        'outfile', type=argparse.FileType('w'),
        help='path to output file')
    parser.add_argument('--i', type=float,
                        help='ionic strength in M',
                        default=0.1)
    parser.add_argument('--ph', type=float, help='pH level', default=7.0)
    logging.getLogger().setLevel(logging.WARNING)

    args = parser.parse_args()

    sys.stderr.write('pH = %.1f\n' % args.ph)
    sys.stderr.write('I = %.1f M\n' % args.i)

    ids = []
    reactions = []
    with open('data/iJO1366_reactions.csv', 'r') as f:
        for row in csv.DictReader(f):
            ids.append(row['bigg.reaction'])
            try:
                reactions.append(Reaction.parse_formula(row['formula']))
            except ValueError as e:
                print('warning: cannot parse reaction %s because of %s' %
                      (row['bigg.reaction'], str(e)))
                reactions.append(Reaction({}))
                continue
        
    equilibrator = ComponentContribution(pH=args.ph, ionic_strength=args.i)

    dG0_prime, U = equilibrator.dG0_prime_multi(reactions)

    writer = csv.writer(args.outfile)
    header = ['reaction', 'pH', 'ionic strength [M]', 'dG\'0 [kJ/mol]',
              'uncertainty [kJ/mol]', 'ln(Reversibility Index)', 'comment']
    writer.writerow(header)
    for s, r, dg0, u in zip(ids, reactions,
                            dG0_prime.flat, U.diagonal().flat):
        row = [s, args.ph, args.i]
        if r.is_empty():
            row += [nan, nan, nan, 'reaction is empty']
        elif r.check_full_reaction_balancing():
            ln_RI = r.calculate_reversibility_index_from_dG0_prime(dg0)
            row += ['%.2f' % dg0, '%.2f' % sqrt(u), '%.2f' % ln_RI, '']
        else:
            row += [nan, nan, nan, 'reaction is not chemically balanced']
        writer.writerow(row)

    args.outfile.flush()
