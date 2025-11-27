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