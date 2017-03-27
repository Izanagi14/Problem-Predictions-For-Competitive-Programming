from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render

from DataMadeJson import dataset
from math import sqrt


def similarity_score(person1, person2):
    both_viewed = {}

    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1

        if len(both_viewed) == 0:
            return 0

        sum_of_eclidean_distance = []

        for item in dataset[person1]:
            if item in dataset[person2]:
                sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
        sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

        return 1 / (1 + sqrt(sum_of_eclidean_distance))


# print similarity_score('User0', 'User1')

def most_similar_users(person, number_of_users):
    scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset if
              other_person != person]

    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]


def pearson_correlation(person1, person2):
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    if number_of_ratings == 0:
        return 0

    person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

    person1_square_preferences_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])

    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

    numerator_value = float(product_sum_of_both_users) - (
        person1_preferences_sum * person2_preferences_sum / (float)(number_of_ratings))
    denominator_value = sqrt(
        (float(person1_square_preferences_sum) - pow(person1_preferences_sum, 2) / (float)(number_of_ratings)) * (
            float(person2_square_preferences_sum) - pow(person2_preferences_sum, 2) / float(number_of_ratings)))
    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / (float)(denominator_value)
        return r


# print pearson_correlation('User0', 'User1')


def user_recommendations(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other)

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:
                # Similrity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score, recommend_item in rankings]
    return recommendataions_list


def create_json():
    crawleddata = open("CrawledData.txt", "r")
    questions = {}
    name = list()

    for columns in (raw.strip().split() for raw in crawleddata):
        questions[columns[0]] = list(columns[1:])
        name.append(columns[0])
    # print questions['QUEEN']
    # print len(questions['QUEEN'])

    # for j in range(0,5):
    # with open("new1.txt","w") as kp:
    #     for m in range(0,5):
    #     	kp.write("'")
    #         kp.write("User"+str(m))
    #         kp.write(":{")
    #         kp.write("\t")
    #         for i in range(0,len(name)):
    #         	kp.write(name[i]+" ")
    #         	kp.write(str(questions[name[i]]))
    #         	kp.write("\n")
    with open("UserNames.txt") as f:
        users = f.read().split()
    with open("DataMadeJson.py", "w") as kp:
        kp.write("dataset = " + "\\" + "\n { \n")
        for j in range(0, len(users)):
            kp.write("'" + users[j] + "':" + "{" + "\n")
            for i in range(0, len(name)):
                kp.write("\t\t" + "'" + name[i] + "':")
                k = (questions[name[i]])
                if j == 0:
                    if i == len(name) - 1:
                        k[j] = k[j].strip(",")
                    kp.write(k[j].strip("["))
                elif j == len(users) - 1:
                    if i == len(name) - 1:
                        k[j] = k[j].strip("]")
                    else:
                        k[j] = k[j].strip("]") + ","
                    kp.write(k[j])
                elif i != len(name) - 1:
                    kp.write(k[j])
                else:
                    kp.write(k[j].strip(","))
                kp.write("\n")
            if j != len(users) - 1:
                kp.write("},")
            else:
                kp.write("}")
            kp.write("\n")

        kp.write("\n" + "}")


def add_new_user(q_for_user):
    print "hahahahahahah"
    with open("CrawledData.txt", "r") as crawleddata:
        print "lklklk"
        names = {}
        questions = []
        for data in (raw.strip().split() for raw in crawleddata):
            questions.append(data[0])
            names[data[0]] = [int(data[1][1])]
            for i in range(2, len(data)):
                names[data[0]].append(int(data[i][0]))
    for i in names:
        print i, names[i]
    for i in names:
        if i in q_for_user['solved']:
            names[i].append(1)
        elif i in q_for_user['todo']:
            names[i].append(0)
        else:
            names[i].append(3)
    with open("CrawledData.txt", "w") as crawleddata:
        for i in names:
            crawleddata.write(i + " ")
            crawleddata.write(str(names[i]) + "\n")
    print "heheehheehheheh"


def get_user_data(user_name):
    import bs4 as bs
    import urllib
    print "Get user data"
    present = 0
    with open("UserNames.txt", "r") as users:
        uu = users.read()
        for u in uu.split():
            print u
            if u == user_name:
                present = 1
                break
    if present == 0:
        with open("UserNames.txt", 'a+') as users:
            users.write(" " + user_name)
    sauce = urllib.urlopen("http://www.spoj.com/users/" + user_name + "/").read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    q_for_user = {'solved': [], 'todo': []}
    for i in soup.find_all('table', attrs={'class': 'table table-condensed'}):
        q_for_user['solved'] = i.text.strip().split('\n')
        break
    solved = set()
    todo = set()
    for i in (q_for_user['solved']):
        if i != "":
            solved.add(i)
    for i in soup.find_all(lambda tag: tag.name == 'table' and tag.get('class') == ['table']):
        q_for_user['todo'] = i.text.strip().split('\n')
        break
    for i in (q_for_user['todo']):
        if i != "":
            todo.add(i)
    q_for_user['solved'] = list(solved)
    q_for_user['todo'] = list(todo)
    print q_for_user
    if present == 0:
        add_new_user(q_for_user)
        # return q_for_user


def index(request):
    return render(request, "Site/index.html")


def search(request):
    # return HttpResponse('<h1>asdasdsaasdasdasdas</h1>')
    # print "jhadhskahklhdas"
    if request.method == 'POST':
        search_id = request.POST["handles"]
        # print search_id
        pos = search_id.rfind("/")
        user_name = search_id[pos + 1:]
        try:
            html = ("<h1> %s </h1>", search_id)
            import bs4 as bs
            import urllib

            a = []
            ll = []
            str1 = "WRONG ANSWER"
            str1 = str1.lower()
            ll.append(str1.lower())
            str4 = "accepted"
            str2 = "MEMORY LIMIT EXCEEDED"
            str2 = str2.lower()
            ll.append(str2)
            str3 = "TIME LIMIT EXCEEDED"
            str3 = str3.lower()
            str5 = "COMPILATION ERROR"
            str5 = str5.lower()
            str6 = "runtime error"
            ll.append(str3)
            wa = 0
            acc = 0
            mle = 0
            tle = 0
            cme = 0
            rte = 0

            sauce = urllib.urlopen("http://www.spoj.com/status/" + user_name + "/all/start=0").read()
            soup = bs.BeautifulSoup(sauce, 'lxml')

            for i in soup.find_all('td', attrs={'class': 'statusres text-center'}):
                if str(i.text).strip() == str1:
                    wa += 1
                elif i.text.strip() in str2:
                    mle += 1
                elif i.text.strip() in str3:
                    tle += 1
                elif str4 in str(i.text).strip():
                    acc += 1
                elif i.text.strip().isnumeric == True:
                    acc += 1
                elif str5 in str(i.text).strip():
                    cme += 1
                elif str6 in str(i.text).strip():
                    rte += 1

            count = 0
            count += 20
            string = "http://www.spoj.com/status/" + user_name + "/all/start=" + str(count)
            sauce = urllib.urlopen(string).read()
            soup = bs.BeautifulSoup(sauce, 'lxml')
            while not soup.find_all('li', attrs={'class': 'disabled'}):
                string = "http://www.spoj.com/status/" + user_name + "/all/start=" + str(count)
                print string
                sauce = urllib.urlopen(string).read()
                soup = bs.BeautifulSoup(sauce, 'lxml')
                count += 20
                for i in soup.find_all('td', attrs={'class': 'statusres text-center'}):
                    if str(i.text).strip() == str1:
                        wa += 1
                    elif i.text.strip() in str2:
                        mle += 1
                    elif i.text.strip() in str3:
                        tle += 1
                    elif str4 in str(i.text).strip():
                        acc += 1
                    elif i.text.strip().isnumeric == True:
                        acc += 1
                    elif str5 in i.text.strip():
                        cme += 1
                    elif str6 in i.text.strip():
                        rte += 1

            count += 20
            string = "http://www.spoj.com/status/" + user_name + "/all/start=" + str(count)
            sauce = urllib.urlopen(string).read()
            soup = bs.BeautifulSoup(sauce, 'lxml')
            for i in soup.find_all('td', attrs={'class': 'statusres text-center'}):
                if str(i.text).strip() == str1:
                    wa += 1
                elif i.text.strip() in str2:
                    mle += 1
                elif i.text.strip() in str3:
                    tle += 1
                elif str4 in str(i.text).strip():
                    acc += 1
                elif i.text.strip().isnumeric == True:
                    acc += 1
                elif str5 in str(i.text).strip():
                    cme += 1
                elif str6 in str(i.text).strip():
                    rte += 1

            print acc
            print tle
            print tle
            print mle
            print wa
            print cme
            print rte

            q_for_user = get_user_data(user_name)
            print "haha", type(q_for_user)
            # add_new_user(q_for_user)
            create_json()
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            labels = 'WRONG ANSWER', 'TIME LIMIT EXCEEDED', 'ACCEPTED', 'COMPILATION ERROR', 'RUNTIME ERROR'
            total = wa + acc + tle + mle + cme + rte
            sizes = []
            sizes.append(float(wa / float(total)))
            sizes.append(float(tle / float(total)))
            sizes.append(float(acc / float(total)))
            sizes.append(float(cme / float(total)))
            sizes.append(float(rte / float(total)))
            explode = (0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')`

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.axis('equal')
            stri = "/home/apurv/Desktop/Project_Minor2/Site/static/Site/foo.png"
            plt.savefig(stri)
            # plt.show()
            with open("UserNames.txt", "r") as f1:
                users1 = f1.read().split()
            print most_similar_users(user_name,len(users1))
            most_similar_name = most_similar_users(user_name, len(users1))
            most_similar = most_similar_name[0][1]
            recommended_list = user_recommendations(most_similar)
            # print recommended_list
            return render(request, 'Site/view.html', {'seach': user_name ,'rec': ' '.join(recommended_list)})
        except:
            return HttpResponse("No Such User")
