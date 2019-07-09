#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:24:15 2017

@author: noore
"""

import argparse
import logging
from equilibrator_api import Pathway
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Calculate the Max-min Driving Force (MDF) of a pathway.')
    parser.add_argument(
        'infile', type=argparse.FileType(),
        help='path to input file containing reactions')
    parser.add_argument(
        'outfile', type=str,
        help='path to output PDF file')
    logging.getLogger().setLevel(logging.WARNING)

    args = parser.parse_args()

    pp = Pathway.from_sbtab(args.infile)
    
    output_pdf = PdfPages(args.outfile)
    mdf_res = pp.calc_mdf()
    
    output_pdf.savefig(mdf_res.conc_plot)
    output_pdf.savefig(mdf_res.mdf_plot)
    output_pdf.close()
    
    rxn_df = pd.DataFrame(mdf_res.report_reactions)
    cpd_df = pd.DataFrame(mdf_res.report_compounds)
