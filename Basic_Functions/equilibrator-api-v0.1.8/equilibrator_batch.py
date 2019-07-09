# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25th 2015

@author: flamholz
"""

from equilibrator_api import ComponentContribution, Reaction, ReactionMatcher
import logging
import argparse
import csv
from numpy import sqrt, nan
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculate potentials for a number of reactions.')
    parser.add_argument(
        'infile', type=argparse.FileType(),
        help='path to input file containing reactions')
    parser.add_argument(
        '--plaintext', action='store_true',
        help='indicate that reactions are given in plain text (not KEGG IDs)')
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

    infile_lines = list(filter(None, map(str.strip, args.infile.readlines())))

    reaction_matcher = ReactionMatcher()
    if args.plaintext:
        reactions = list(map(reaction_matcher.match, infile_lines))
    else:
        reactions = list(map(Reaction.parse_formula, infile_lines))

    equilibrator = ComponentContribution(pH=args.ph, ionic_strength=args.i)

    dG0_prime, U = equilibrator.dG0_prime_multi(reactions)

    writer = csv.writer(args.outfile)
    header = ['reaction (text)', 'reaction (KEGG)', 'pH', 'ionic strength [M]', 'dG\'0 [kJ/mol]',
              'uncertainty [kJ/mol]', 'ln(Reversibility Index)', 'comment']
    writer.writerow(header)
    for s, r, dg0, u in zip(infile_lines, reactions,
                            dG0_prime.flat, U.diagonal().flat):
        if args.plaintext:
            row = [s, r.write_formula()]
        else:
            row = [reaction_matcher.write_text_formula(r), s]
        row += [args.ph, args.i]
        if r.check_full_reaction_balancing():
            ln_RI = r.calculate_reversibility_index_from_dG0_prime(dg0)
            row += ['%.2f' % dg0, '%.2f' % sqrt(u), '%.2f' % ln_RI, '']
        else:
            row += [nan, nan, nan, 'reaction is not chemically balanced']
        writer.writerow(row)

    args.outfile.flush()
