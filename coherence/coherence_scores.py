import numpy as np
from collections import Counter

def compute_coherence_score_umass(terms, documents):
    term_doc_freq = Counter()
    pair_doc_freq = Counter()
    
    for doc in documents:
        unique_terms = set(doc)
        term_doc_freq.update(unique_terms & set(terms))  # Update document frequency of terms
        
        # Find all pairs in document and update their co-occurrence count
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if terms[i] in unique_terms and terms[j] in unique_terms:
                    pair_doc_freq[(terms[i], terms[j])] += 1
                    pair_doc_freq[(terms[j], terms[i])] += 1  # Include both orderings
    
    coherence = 0
    for i in range(len(terms)-1):
        for j in range(i+1, len(terms)):
            term1 = terms[i]
            term2 = terms[j]
            D_wi = term_doc_freq[term1]
            D_wj = term_doc_freq[term2]
            D_wi_wj = pair_doc_freq.get((term1, term2), 0)
            if D_wi > 0 and D_wj>0 and  D_wi_wj > 0:
                coherence += np.log((D_wi_wj + 1.0) / D_wi)
    
    return coherence


def compute_coherence_score_uci(terms, text, window_size=10):
    word_count = Counter(text)
    co_count = Counter((text[i], text[i+j+1])
                       for i in range(len(text) - window_size + 1)
                       for j in range(window_size - 1)
                       if (text[i], text[i+j+1]) in [(terms[m], terms[n]) for m in range(len(terms)) for n in range(m + 1, len(terms))])
    
    epsilon = 1e-12
    coherence = 0
    total_pairs = 0

    for f, s in [(terms[i], terms[j]) for i in range(len(terms)) for j in range(i + 1, len(terms))]:
        if (f, s) in co_count:
            joint_prob = (co_count[(f, s)] + epsilon) / (len(text) - window_size + 1)
            prob_f = (word_count[f] + epsilon) / len(text)
            prob_s = (word_count[s] + epsilon) / len(text)
            pmi = np.log(joint_prob / (prob_f * prob_s))
            coherence += pmi
            total_pairs += 1

    return coherence / max(total_pairs, 1)  # prevent division by zero