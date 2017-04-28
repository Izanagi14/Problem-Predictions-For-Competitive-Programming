from final_data import dataset2
from math import sqrt
def similarity_score(person1, person2):
    both_viewed = {}

    for item in dataset2[person1]:
        if item in dataset2[person2]:
            both_viewed[item] = 1

        if len(both_viewed) == 0:
            return 0

        sum_of_eclidean_distance = []

        for item in dataset2[person1]:
            if item in dataset2[person2]:
                sum_of_eclidean_distance.append(pow(dataset2[person1][item] - dataset2[person2][item], 2))
        sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

        return 1 / (1 + sqrt(sum_of_eclidean_distance))


# print similarity_score('User0', 'User1')

def most_similar_users(person, number_of_users):
    scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset2 if
              other_person != person]

    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]


def pearson_correlation(person1, person2):
    both_rated = {}
    #print person1 + " " + person2
    for item in dataset2[person1]:
        if item in dataset2[person2]:
            both_rated[item] = 1
    print both_rated
    number_of_ratings = len(both_rated)

    if number_of_ratings == 0:
        return 0

    person1_preferences_sum = sum([dataset2[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset2[person2][item] for item in both_rated])
    #print person1_preferences_sum
    #print person2_preferences_sum
    person1_square_preferences_sum = sum([pow(dataset2[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset2[person2][item], 2) for item in both_rated])

    product_sum_of_both_users = sum([dataset2[person1][item] * dataset2[person2][item] for item in both_rated])

    numerator_value = float(product_sum_of_both_users) - (
        person1_preferences_sum * person2_preferences_sum / (float)(number_of_ratings))
    denominator_value = sqrt(
        (float(person1_square_preferences_sum) - pow(person1_preferences_sum, 2) / (float)(number_of_ratings)) * (
            float(person2_square_preferences_sum) - pow(person2_preferences_sum, 2) / float(number_of_ratings)))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / (float)(denominator_value)
        #print r
        return r


# print pearson_correlation('User0', 'User1')


def user_recommendations(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset2:
        print other
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other)
        print sim
        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset2[other]:

            # only score movies i haven't seen yet
            if item not in dataset2[person] or dataset2[person][item] == 0:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] += dataset2[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    #print rankings
    rankings.sort()
    rankings.reverse()
    print rankings
    # returns the recommended items
    recommendataions_list = [recommend_item for score, recommend_item in rankings]
    return recommendataions_list


print most_similar_users("moejy0viiiiiv", 3)
#print user_recommendations("moejy0viiiiiv")