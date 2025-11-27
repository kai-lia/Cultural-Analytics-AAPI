def logodds(counter1, counter2, display=25):
    vocab=dict(counter1)
    vocab.update(dict(counter2))
    count1_sum=sum(counter1.values())
    count2_sum=sum(counter2.values())

    ranks={}
    alpha=0.01
    alphaV=len(vocab)*alpha

    for word in vocab:

        log_odds_ratio=math.log( (counter1[word] + alpha) / (count1_sum+alphaV-counter1[word]-alpha) ) - math.log( (counter2[word] + alpha) / (count2_sum+alphaV-counter2[word]-alpha) )
        variance=1./(counter1[word] + alpha) + 1./(counter2[word] + alpha)

        ranks[word]=log_odds_ratio/math.sqrt(variance)

    sorted_x = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)

    print("Most category 1:")
    for k,v in sorted_x[:display]:
        print("%.3f\t%s" % (v,k))

    print("\nMost category 2:")
    for k,v in reversed(sorted_x[-display:]):
        print("%.3f\t%s" % (v,k))


import numpy as np
import operator
from collections import Counter

def combine_counts(group_counters_dict):
    """
    Combines multiple Counter objects into a single Counter object.
    group_counters_dict: A dictionary where keys are group names and values are Counter objects.
    """
    combined = Counter()
    for group, counter in group_counters_dict.items():
        combined.update(counter)
    return combined

def log_odds_for_group(group_counters, target_group, display = 10):
    """
    Performs a one-vs-all log-odds comparison with z-scores for a target group
    against all other groups combined.

    Args:
        group_counters (dict): A dictionary where keys are group names (str) and
                               values are Counter objects representing word counts for that group.
        target_group (str): The name of the group to be compared.
        alpha (float): Dirichlet prior strength.

    Returns:
        list: A sorted list of tuples, each containing (word, z_score, target_group_count, other_groups_count).
              Sorted in descending order by z_score.
    """
    target_group_counters = {}

    print(group_counters.keys())

    print(target_group)

    for group in target_group:
      if group not in group_counters.keys():
        print(f"Group {group} not found in group_counters.")
        return
      else:
        target_group_counters[group] = group_counters.pop(group)

    logodds(combine_counts(target_group_counters), combine_counts(group_counters), display=display)

    return
