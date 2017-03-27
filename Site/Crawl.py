import bs4 as bs
import urllib

a = []
ll = []
ll.append("WRONG_ANSWER")
str1 = "WRONG_ANSWER"
ll.append("OK")
str4 = "OK"
ll.append("MEMORY_LIMIT_EXCEEDED")
str2 = "MEMORY_LIMIT_EXCEEDED"
ll.append("TIME_LIMIT_EXCEEDED")
str3 = "TIME_LIMIT_EXCEEDED"
wa = 0
acc = 0
mle = 0
tle = 0

sauce = urllib.urlopen("http://codeforces.com/submissions/apurvtandon/page/1").read()
soup = bs.BeautifulSoup(sauce, 'lxml')
for i in ll:
    for k in soup.find_all('span', attrs={'submissionverdict': i}):
        a.append(i)
#print a
for i in a:
    if str1 == i:
        wa += 1
    elif str2 == i:
        mle += 1
    elif str3 == i:
        tle += 1
    elif str4 == i:
        acc += 1

str5 = "inactive"
count = 2
# wa/=2
# mle/=2
# acc/=2
# tle/=2
# print "Accepted" + str(acc)
# print " Tle " + str(tle)
# print " Memory Level Exceded " + str(mle)
# print " Wrong Answer " + str(wa)
while (soup.find_all('span', attrs={'class': 'inactive'})):
    a[:] = []
    crawlstring = "http://codeforces.com/submissions/apurvtandon/page/" + str(count)
    #print crawlstring
    sauce = urllib.urlopen(crawlstring).read()
    count += 1
    soup = bs.BeautifulSoup(sauce, 'lxml')
    for i in ll:
        for k in soup.find_all('span', attrs={"submissionverdict": i}):
            a.append(i)
    for i in a:
        if str1 == i:
            wa += 1
        elif str2 == i:
            mle += 1
        elif str3 == i:
            tle += 1
        elif str4 == i:
            acc += 1
    # wa /= 2
    # mle /= 2
    # acc /= 2
    # tle /= 2
    # print "Accepted" + str(acc)
    # print " Tle " + str(tle)
    # print " Memory Level Exceded " + str(mle)
    # print " Wrong Answer " + str(wa)
string2 = "http://codeforces.com/submissions/apurvtandon/page/" + str(count)
print string2

sauce = urllib.urlopen(string2).read()
soup = bs.BeautifulSoup(sauce, 'lxml')
a[:] = []
for i in ll:
    for k in soup.find_all('span', attrs={"submissionverdict": i}):
        a.append(i)
#print a
for i in a:
    if str1 == i:
        wa += 1
    elif str2 == i:
        mle += 1
    elif str3 == i:
        tle += 1
    elif str4 == i:
        acc += 1

#print "Accepted" + str(acc)
#print " Tle " + str(tle)
#print " Memory Level Exceded " + str(mle)
#print " Wrong Answer " + str(wa)

import matplotlib.pyplot as plt

labels ='WRONG ANSWER' , 'MEMORY LIMIT EXCEEDED', 'TIME LIMIT EXCEEDED',   'ACCEPTED'
total = wa+acc+tle+mle
sizes = []
sizes.append(float(wa/float(total)))
sizes.append(float(mle/float(total)))
sizes.append(float(tle/float(total)))
sizes.append(float(acc/float(total)))
explode = (0.1, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')`

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

