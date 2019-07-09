import pkg_resources
from nltk.metrics import edit_distance
from collections import defaultdict
import logging
import itertools
import pandas as pd
from equilibrator_api import Reaction, Compound
from equilibrator_api.query_parser import QueryParser, ParseError

COMPOUND_NAME_FILE = pkg_resources.resource_filename(
    'equilibrator_api', 'data/kegg_compound_names.tsv')
COMPOUND_RENAME_FILE = pkg_resources.resource_filename(
    'equilibrator_api', 'data/kegg_compound_renaming.tsv')

class CompoundMatcher(object):
    """
        CompoundMatches uses the same approximate search as implemented by
        eQuilibrator online, essentially using a combination of 
        auto-complete and N-gram search (with N = 4)
    """
    
    def __init__(self, max_results=10, min_score=0.0):
        self._max_results = max_results
        self._min_score = min_score
        
        # read all KEGG IDs and corresponding names
        cid2names = self._load_kegg_names()
        
        # now make the inverse dictionary
        cid_name_pairs = list(itertools.chain(
                *[[(k,i) for i in v] for (k,v) in cid2names.items()]))
        self.compound_df = pd.DataFrame(data=cid_name_pairs, columns=['CID', 'compound_name'])
        self.compound_df['lowercase_name'] = self.compound_df.compound_name.str.lower()
        
        # convert the cid2names dictionary to a 1:1 mapping from the CID
        # to the first name in the list
        self.cid2name = dict([(k, v[0] if v != [] else '') for (k, v) in cid2names.items()])

    @staticmethod
    def _load_kegg_names(kegg_names_filename=COMPOUND_NAME_FILE,
                         kegg_renaming_filename=COMPOUND_RENAME_FILE):
        """
            Read compound names into a dictionary, where the keys are 
        """
        
        kegg_names_df = pd.read_csv(kegg_names_filename, index_col=0, header=0,
                                    delimiter='\t')
        cid2names = defaultdict(list, kegg_names_df['all names'].str.split('|').to_dict())
        
        # the first name on the list should be the 'common name'
        for cid, row in kegg_names_df.iterrows():
            if row['common name'] not in cid2names[cid]:
                raise ValueError('The common name \'%s\' is not in the name list for %s'
                                 % (row['common name'], cid))
            # move the common name to be first on the list of "all names"
            if cid2names[cid][0] != row['common name']:
                cid2names[cid].remove(row['common name'])
                cid2names[cid] = [row['common name']] + cid2names[cid]
                
        # update the name list according to the "renaming" file
        kegg_rename_df = pd.read_csv(kegg_renaming_filename, index_col=None,
                                     header=0, delimiter='\t')
        
        for _, row in kegg_rename_df.iterrows():
            if row.CID not in cid2names:
                raise ValueError('%s appears in the renaming file, but not in the KEGG list'
                                 % row.CID)
    
            if row.command.lower() == 'remove':
                # remove 'name' from the list of names
                try:
                    cid2names[row.CID].remove(row['name'])
                except ValueError:
                    logging.warning('The name %s is not one of the options for %s, '
                                    'so it cannot be removed' % (row['name'], row.CID))
            elif row.command.lower() == 'add':
                # put 'name' in the end of the list (or move it there if it is
                # already in the list)
                if row['name'] in cid2names[row.CID]:
                    cid2names[row.CID].remove(row['name'])
                cid2names[row.CID] = cid2names[row.CID] + [row['name']]
            elif row.command.lower() == 'delete':
                del cid2names[row.CID]
            elif row.command.lower() == 'replace':
                cid2names[row['name']] += cid2names[row.CID]
                del cid2names[row.CID]
            else:
                raise ValueError('Unknown command: %s' % row.command)
                
        for cid in cid2names.keys():
            cid2names[cid].append(cid)
        return cid2names

    def _get_score(self, query, match):
        """Get the score for a query-match pair.

        Args:
            query: the query string.
            match: the matching compound name found in compound_df

        Returns:
            A score between 0.0 and 1.0.
        """
        dist = float(edit_distance(query, match))
        return 1.0 - dist / max(len(query), len(match))

    def match(self, query):
        """Find matches for a single text query.

        Args:
            query: the string query.

        Returns:
            The closest match in KEGG format, or None.
        """
        query = query.strip().lower()
        
        # Start by looking for exact matches (ignoring case)
        matches = self.compound_df[
            self.compound_df.lowercase_name == query]
        
        if matches.shape[0] == 0:
            # Try plain old autocomplete. If it works, great.
            autocomp_matches = self.compound_df[
                self.compound_df.lowercase_name.str.match('^' + query)]
            
            matches = matches.append(autocomp_matches.iloc[:, :self._max_results])

        if matches.shape[0] == 0:
            # for string of length 4 or more, use N-grams to find more hits
            ngram_matches = []
            for i in range(len(query) - 3):
                ngram = query[i:i+4]
                ngram_matches = self.compound_df[
                    self.compound_df.lowercase_name.str.match('.*' + ngram + '.*')]
                matches = matches.append(ngram_matches.iloc[:, :self._max_results])
        
        if matches.shape[0] == 0:
            return None
        
        score = matches.lowercase_name.apply(lambda m: self._get_score(query, m))
        matches = matches.assign(score=score)
        matches = matches.drop_duplicates().sort_values('score', ascending=False)
        matches = matches[matches.score >= self._min_score]
        return matches

class ReactionMatcher(object):
    """
        ReactionMatcher is designed to emulate the behaviour of eQuilibrator's
        Search Bar, i.e. automatically mapping a textual chemical formula
        to specific chemical structures, and suggesting H2O/NAD(H) balancing
        corrections.
    """

    def __init__(self):
        """Initialize the ReactionMatcher.
        
        Args:
            compound_matcher: a matcher.Matcher object that matches
                individual compounds.
        """
        self._compound_matcher = CompoundMatcher()
        self._query_parser = QueryParser()
    
    def match(self, query):
        if not self._query_parser.is_reaction_query(query):
            raise ValueError('This query does not look like a reaction: ' + query)
        parsed_query = self._query_parser.parse_reaction_query(query)
        return self.get_best_match(parsed_query)
    
    def get_best_match(self, parsed_query):
        kegg_id_to_coeff = defaultdict(int)
        for coeff, name in parsed_query.substrates:
            comp_matches = self._compound_matcher.match(name)
            if comp_matches is None:
                raise ParseError('Cannot match this substrate at all: ' + name)
            kegg_id_to_coeff[comp_matches.CID.iat[0]] -= coeff
        for coeff, name in parsed_query.products:
            comp_matches = self._compound_matcher.match(name)
            if comp_matches is None:
                raise ParseError('Cannot match this product at all: ' + name)
            kegg_id_to_coeff[comp_matches.CID.iat[0]] += coeff
        
        # if compounds appear more than once (e.g. on both sides of a reaction
        # we need to sum up the coresponding coefficients)
        kegg_id_to_coeff = dict(filter(lambda x: x[1], kegg_id_to_coeff.items()))
        
        return Reaction(kegg_id_to_coeff)
    
    def write_compound_and_coeff(self, compound_id, coeff):
        compound_name = self._compound_matcher.cid2name.get(compound_id, '?')
        if coeff == 1:
            return compound_name
        else:
            return "%g %s" % (coeff, compound_name)

    def write_text_formula(self, reaction):
        """
            Unlike Reaction.write_formula(), this function uses the compound
            names instead of the KEGG IDs.
        """
        left = []
        right = []
        for kegg_id in reaction.kegg_ids():
            coeff = reaction.get_coeff(kegg_id)
            if coeff < 0:
                left.append(self.write_compound_and_coeff(kegg_id, -coeff))
            elif coeff > 0:
                right.append(self.write_compound_and_coeff(kegg_id, coeff))
        return "%s %s %s" % (' + '.join(left), '=', ' + '.join(right))


if __name__ == '__main__':
    rm = ReactionMatcher()
    
    # example of a standard reaction (ATP hydrolysis)
    m = rm.match('ATP + H2O = ADP + orthophosphate')
    print('ATP hydrolysis reaction: ', rm.write_text_formula(m))

    # an example of an "empty" reaction (e.g. membrane transport)
    m = rm.match('citrate + (S)-MALATE = citrate + (S)-MALATE')
    print('Empty reaction: ', rm.write_text_formula(m))
    
    # example of you to use CompoundMatcher to get a formation energy
    cm = CompoundMatcher()
    df = cm.match('ATP')
    c = Compound(Reaction.COMPOUND_DICT[df['CID'].iloc[0]])
    print('Formation energy of ATP: %.2f' % c.dG0_prime(pH=7, pMg=0, I=0.1))
