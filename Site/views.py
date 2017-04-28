from math import sqrt

import bs4 as bs
from bs4 import BeautifulSoup
from django.http import HttpResponse
from django.shortcuts import render

from DataMadeJson import dataset
from final_data2 import dataset3
from final_data import dataset2


# from __future__ import absolute_import, division, print_function


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
    for item in dataset2[person1]:
        if item in dataset2[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    if number_of_ratings == 0:
        return 0

    person1_preferences_sum = sum([dataset2[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset2[person2][item] for item in both_rated])

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
        return r


# print pearson_correlation('User0', 'User1')


def user_recommendations(person,a):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset2:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other)

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
    # import bs4 as bs
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
    print "Inside Search Function"
    if request.method == 'POST':
        search_id = request.POST["handles"]
        print search_id
        pos = search_id.rfind("/")
        site_name = search_id[:pos]
        print site_name
        user_name = search_id[pos + 1:]
        if site_name == "sp":
            try:
                html = ("<h1> %s </h1>", search_id)
                import bs4 as bs
                import urllib
                print "hello urlib working"
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
                print most_similar_users(user_name, len(users1))
                most_similar_name = most_similar_users(user_name, len(users1))
                most_similar = most_similar_name[0][1]
                recommended_list = user_recommendations(most_similar)
                list_len = len(recommended_list)
                if (list_len > 100):
                    list_len = list_len % 53
                else:
                    list_len = list_len % 17
                recommended_list = recommended_list[0:list_len]
                # print recommended_list
                return render(request, 'Site/view.html', {'seach': user_name, 'rec': ' '.join(recommended_list)})
            except:
                return HttpResponse("No Such Spoj User")
        else:
            print "For Codeforces"
            try:
                print "hello"
                import urllib
                r = urllib.urlopen('http://codeforces.com/submissions/' + user_name + '/page/1').read()
                print ('http://codeforces.com/submissions/'+user_name+'/page/1')
                pages_cnt = 0
                soup = BeautifulSoup(r, 'lxml')
                for i in soup.find_all('span', attrs={'class': 'page-index'}):
                    pages_cnt = int(i.text)
                ac = tle = wa = mle = idle = 0
                lis1 = []
                if dataset3.get(user_name) != None:
                    print "hahahhahhah"
                    if pages_cnt > dataset3[user_name][0]:
                        print "Inside if"
                        for page in range(1, pages_cnt - dataset3[user_name][0] + 1):
                            d_cur = dataset3[user_name][1]
                            r = urllib.urlopen('http://codeforces.com/submissions/' + user_name + '/page/' + str(page)).read()
                            soup = BeautifulSoup(r, 'lxml')
                            s = []
                            ddd = soup.find_all('a', attrs={'class': 'view-source'})
                            if len(ddd) != 0:
                                for k in soup.find_all('tr', attrs={'data-submission-id': j.text}):
                                    x = k.text.split("\n")
                                    lis1.append((x[12].strip(), x[8].strip(), x[19].strip()))
                            else:
                                for j in soup.find_all('span', attrs={'class': 'hiddenSource'}):
                                    # lis.append(j.text)
                                    for k in soup.find_all('tr', attrs={'data-submission-id': j.text}):
                                        x = k.text.split("\n")
                                        lis1.append((x[12].strip(), x[8].strip(), x[19].strip()))
                            for ques in lis1:
                                if ques[0] in d_cur.keys():
                                    if ques[2] == "Accepted":
                                        d_cur[ques[0]][0] += 1
                                    elif ques[2].find('Time'):
                                        d_cur[ques[0]][1] += 1
                                    elif ques[2].find('Wrong'):
                                        d_cur[ques[0]][2] += 1
                                    elif ques[2].find('Memory'):
                                        d_cur[ques[0]][3] += 1
                                    elif ques[2].find('Idleness'):
                                        d_cur[ques[0]][4] += 1
                                else:
                                    if ques[2] == "Accepted":
                                        d_cur[ques[0]] = [1, 0, 0, 0, 0]
                                    elif ques[2].find('Time'):
                                        d_cur[ques[0]] = [0, 1, 0, 0, 0]
                                    elif ques[2].find('Wrong'):
                                        d_cur[ques[0]] = [0, 0, 1, 0, 0]
                                    elif ques[2].find('Memory'):
                                        d_cur[ques[0]] = [0, 0, 0, 1, 0]
                                    elif ques[2].find('Idleness'):
                                        d_cur[ques[0]] = [0, 0, 0, 0, 1]
                    for ques in dataset3[user_name][1].values():
                        ac += ques[0]
                        tle += ques[1]
                        wa += ques[2]
                        mle += ques[3]
                        idle += ques[4]
                    import matplotlib.pyplot as plt2
                    import matplotlib.patches as mpatches
                    labels = 'ACC', 'TLE', 'WA', 'MLE', 'IDLE'
                    total = wa + ac + tle + mle + idle
                    sizes = []
                    sizes.append(float(ac / float(total)))
                    sizes.append(float(tle / float(total)))
                    sizes.append(float(wa / float(total)))
                    sizes.append(float(mle / float(total)))
                    sizes.append(float(idle / float(total)))
                    explode = (0.1, 0.1, 0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')`

                    fig1, ax1 = plt2.subplots()
                    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
                    ax1.axis('equal')
                    stri = "/home/apurv/Desktop/Project_Minor2/Site/static/Site/foo.png"
                    plt2.savefig(stri)
                    recommended_list = user_recommendations(user_name,"2")
                    print recommended_list
                #return HttpResponse("Codeforces")
                return render(request, 'Site/view.html', {'seach': user_name, 'rec': ' '.join(recommended_list)})
            except:
                return HttpResponse("No Such Codeforces User")


import pandas as pd


def read_process(filname, sep=" "):
    col_names = ["user", "item", "rate"]
    print "hahaasda"
    df = pd.read_csv(filname, sep=sep, header=None, names=None, engine='python')

    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


class ShuffleIterator(object):
    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochIterator(ShuffleIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]


def inference_svd(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    with tf.device("/cpu:0"):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return infer, regularizer


def optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op


import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

np.random.seed(4)

BATCH_SIZE = 1000
USER_NUM = 5
ITEM_NUM = 17335
DIM = 15
EPOCH_MAX = 17
DEVICE = "/cpu:0"


def clip(x):
    return np.clip(x, 1.0, 5.0)


def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    print "HAHA"
    df = read_process("/home/apurv/Desktop/Project_Minor2/ratings.csv", sep=" ")
    rows = len(df)
    print rows
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def svd(train, test):
    samples_per_batch = len(train) // BATCH_SIZE

    iter_train = ShuffleIterator([train["user"],
                                  train["item"],
                                  train["rate"]],
                                 batch_size=BATCH_SIZE)

    iter_test = OneEpochIterator([test["user"],
                                  test["item"],
                                  test["rate"]],
                                 batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                       device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(EPOCH_MAX * samples_per_batch):
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    pred_batch = clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, test_err,
                                                       end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end
