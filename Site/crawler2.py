from bs4 import BeautifulSoup
import json
import urllib
import lxml

with open("../final_data2.py", 'w') as finaldata2:
    finaldata2.write("dataset3 = "+"\\\n")
with open("../final_data.py", 'w') as finaldata:
    finaldata.write("dataset2 = "+"\\\n")
collab_data = {}
user_cnt = 1
d = {}
with open("user_rating", "r") as users:
    for user in users.readlines(): 
        print (user)
        user = user.split()
        lis1 = []
        r = urllib.urlopen('http://codeforces.com/submissions/'+user[0]+'/page/1').read()
        # print ('http://codeforces.com/submissions/'+user[0]+'/page/1')
        ki = 0
        a = 0
        soup = BeautifulSoup(r, 'lxml')
        for i in soup.find_all('span',attrs={'class':'page-index'}):
            a = int(i.text)
        # print (a) 
        for kepper in range(1,2):
            r = urllib.urlopen('http://codeforces.com/submissions/'+user[0]+'/page/' + str(kepper)).read()
            soup = BeautifulSoup(r, 'lxml')
            s = []
            cnt = 1
            if user_cnt % 2 == 1:
                for j in soup.find_all('a', attrs={'class': 'view-source'}):
                    # lis.append(j.text)
                    for k in soup.find_all('tr', attrs={'data-submission-id': j.text}):
                        x = k.text.split("\n")
                        lis1.append((x[12].strip(), x[8].strip(), x[19].strip()))
            else:
                for j in soup.find_all('span', attrs={'class': 'hiddenSource'}):
                    # lis.append(j.text)
                    for k in soup.find_all('tr', attrs={'data-submission-id': j.text}):
                        x = k.text.split("\n")
                        lis1.append((x[12].strip(), x[8].strip(), x[19].strip()))
        print (lis1)
        d[user[0]] = [a,{}]
        collab_data[user[0]] = {}
        d_cur = d[user[0]][1]
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
        # print (d, d_cur)
        rating_inverse = 1.0 / float(user[1])
        cur_user_problems = d_cur.items()
        collab_data_cur_user = collab_data[user[0]]
        for ques,values in cur_user_problems:
            qtype = ques.split()[0][-1]
            prob_rating = rating_inverse
            for i in range(1, 11):
                if qtype == chr(64 + i):
                    prob_rating += i / 10.0
                    break
            solved = 0.0
            if values[0] == 0:
                solved += 1
            for i in range(1,5):
                if values[i] >= 10:
                    solved += 1
                else:
                    solved += values[i]/10
            prob_rating += solved / 5.0
            collab_data_cur_user[ques] = prob_rating
        print (collab_data)
        print d
        if user_cnt == 3:
            break
        user_cnt += 1
with open("../final_data2.py", 'a') as finaldata2:
    json.dump(d, finaldata2)
with open("../final_data.py", 'a') as finaldata:
    json.dump(collab_data,finaldata)
