'''
Script:      generate_scores.py
Purpose:     Generates on-target and off-target scoring for CRISPR gRNAs
Author:      Sophia Li
Affiliation: University of Toronto
Date:        September 20 2025

Description: Generates MIT score which predicts the likelihood that the Cas9 
             nuclease will cut at an off-target site using position-specific 
             mismatch tolerance weights. 

References:
- https://www.biorxiv.org/content/10.1101/2022.04.21.488824v3

'''
import sys
sys.path.append('/Users/sophiali/Desktop/crisproff')
from CRISPRspec_CRISPRoff_pipeline import compute_CRISPRspec, calcRNADNAenergy
import argparse
import pickle
import pandas as pd

def compute_crispr_off_score(sg, wt):

    _, crispr_off_scores = compute_CRISPRspec(sg, [wt], calcRNADNAenergy)
    return crispr_off_scores[0][1]

# =====| CFD Score Implementation |=============================================

def compute_cfd_score_e(sg, wt, pam, mm_scores, pam_scores):
    """
    Parameters
    - sg : sgRNA (guide RNA) sequence
    - wt : the candidate off-target sequence (same length as sg)
    - pam : the PAM sequence adjacent to the off-target
    - mm_scores: mismatch penalties
    - pam_scores: PAM penalties
    Returns
    - CFD score
    """
    
    ## Normalize the nucleotides to RNA convention
    wt = wt.replace('T', 'U')
    sg = sg.replace('T', 'U')

    wt = wt[:20]
    sg = sg[:20]

    wt_list = list(wt)
    sg_list = list(sg)

    ## Introduce penalties for each base pair mismatch
    score = 1
    for i, wl in enumerate(wt_list):

        # No penalty if matching
        if sg_list[i] == wl:
            score *= 1
        else:
            try:
                # Fetch the penalty from the mismatch lookup table
                key = 'r'+sg_list[i]+':d'+revcom(wl)+','+str(i+1)
                score += mm_scores[key]
            except KeyError:
                continue

    ## Multiply the score by the PAM penalty in the lookup table
    score *= pam_scores.get(pam, pam_scores.get("OTHERS", 1.0))
    return score
    
def revcom(s):
    # Returns the reverse complement of a nucleotide string
    basecomp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A', 'N': 'N'}
    letters = list(s[::-1])
    letters = [(basecomp[base] if base in basecomp else base) for base in letters]
    return ''.join(letters)

def get_mismatch_scores(mms_path):
    # Returns a dictionary of mismatch scores given the path
    try:
        return pickle.load(open(mms_path, 'rb'))
    except:
        raise Exception("Could not find file with mismatch scores.)")
    
def get_pam_scores(pam_path):
    # Returns a dictionary of PAM scores given the paths
    try:
        return pickle.load(open(pam_path, 'rb'))
    except:
        raise Exception("Could not find file with PAM scores.)")
    
## Curry the penalty matrices for convenience
PAM_PKL = "data/CFD_score/PAM_scores.pkl"
MMS_PKL = "data/CFD_score/mismatch_score.pkl"

mm_scores = get_mismatch_scores(MMS_PKL)
pam_scores = get_pam_scores(PAM_PKL)

compute_cfd_score = lambda sg, wt, pam: compute_cfd_score_e(
    sg, wt, pam, mm_scores, pam_scores
)

def parse_args():
    p = argparse.ArgumentParser("Computes off-target specificity scores")
    p.add_argument('--train_path')
    p.add_argument('--test_path')

    return p.parse_args()

# =====| Main |================================================

if __name__ == '__main__':
    
    args = parse_args()

    ## Read the training and test set
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    ## Standardize score columns to CRISPRoff score
    train = train.rename(columns = {'score': 'crispr_off_score'})
    test = test.rename(columns = {'score': 'crispr_off_score'})

    ## Generate missing CRISPRoff scores
    train['crispr_off_score'] = train.apply(
        lambda row: compute_crispr_off_score(
            row['guide_sequence'], row['target_sequence']
        ) if pd.isna(row['crispr_off_score']) else row['crispr_off_score'],
        axis = 1
    )
    test['crispr_off_score'] = test.apply(
        lambda row: compute_crispr_off_score(
            row['guide_sequence'], row['target_sequence']
        ) if pd.isna(row['crispr_off_score']) else row['crispr_off_score'],
        axis = 1
    )

    ## Generate missing CFD scores (or all if none)
    train['cfd_score'] = train.apply(lambda row: compute_cfd_score(
        row['guide_sequence'], row['target_sequence'], row['pam']), axis = 1)
    test['cfd_score'] = test.apply(lambda row: compute_cfd_score(
        row['guide_sequence'], row['target_sequence'], row['pam']), axis = 1)
    
    # Standardize column-order
    order = [
        'guide_sequence', 'target_sequence', 'identity', 'strand', 
        'pam', 'chr', 'start', 'end', 'technology', 'cas9_type', 
        'source', 'gene_or_locus', 'has_gene', 'crispr_off_score', 'cfd_score'
    ]

    train = train[order]
    test = test[order]

    ## Sanity checks
    assert not train['crispr_off_score'].isna().any()
    assert not test['crispr_off_score'].isna().any()
    assert not train['cfd_score'].isna().any()
    assert not test['cfd_score'].isna().any()

    train.to_csv('/Users/sophiali/Desktop/GNNOffTarget/data/training.csv', 
                 index = False)

    test.to_csv('/Users/sophiali/Desktop/GNNOffTarget/data/test.csv', 
                index = False)

