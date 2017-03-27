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

sauce = urllib.urlopen("http://www.spoj.com/status/"+"apurvtandon"+"/all/start=0").read()
soup = bs.BeautifulSoup(sauce, 'lxml')

	
for i in soup.find_all('td',attrs={'class':'statusres text-center'}):
    if str(i.text).strip() == str1:
    	wa+=1
    elif i.text.strip() in str2:
    	mle+=1
    elif i.text.strip() in str3:
    	tle+=1
    elif str4 in str(i.text).strip():
    	acc+=1
    elif i.text.strip().isnumeric==True:
    	acc+=1
    elif str5 in str(i.text).strip():
    	cme+=1
    elif str6 in str(i.text).strip():
    	rte+=1
   
count=0
count+=20
string = "http://www.spoj.com/status/"+"apurvtandon"+"/all/start="+str(count)
sauce = urllib.urlopen(string).read()
soup = bs.BeautifulSoup(sauce, 'lxml')
while not soup.find_all('li',attrs={'class':'disabled'}):
	string = "http://www.spoj.com/status/"+"apurvtandon"+"/all/start="+str(count)
	print string
	sauce = urllib.urlopen(string).read()
	soup = bs.BeautifulSoup(sauce, 'lxml')
	count+=20
	for i in soup.find_all('td',attrs={'class':'statusres text-center'}):
	    if str(i.text).strip() == str1:
	    	wa+=1
	    elif i.text.strip() in str2:
	    	mle+=1
	    elif i.text.strip() in str3:
	    	tle+=1
	    elif str4 in str(i.text).strip():
	    	acc+=1
	    elif i.text.strip().isnumeric==True:
	    	acc+=1
	    elif str5 in i.text.strip():
	    	cme+=1
	    elif str6 in i.text.strip():
	    	rte+=1
	    
count+=20
string = "http://www.spoj.com/status/"+"apurvtandon"+"/all/start="+str(count)
sauce = urllib.urlopen(string).read()
soup = bs.BeautifulSoup(sauce, 'lxml')
for i in soup.find_all('td',attrs={'class':'statusres text-center'}):
    if str(i.text).strip() == str1:
    	wa+=1
    elif i.text.strip() in str2:
    	mle+=1
    elif i.text.strip() in str3:
    	tle+=1
    elif str4 in str(i.text).strip():
    	acc+=1
    elif i.text.strip().isnumeric==True:
    	acc+=1
    elif str5 in str(i.text).strip():
    	cme+=1
    elif str6 in str(i.text).strip():
    	rte+=1

print acc
print tle
print tle
print mle
print wa
print cme
print rte

import matplotlib.pyplot as plt

labels ='WRONG ANSWER' , 'TIME LIMIT EXCEEDED',   'ACCEPTED' , 'COMPILATION ERROR' , 'RUNTIME ERROR'
total = wa+acc+tle+mle+cme+rte
sizes = []
sizes.append(float(wa/float(total)))
sizes.append(float(tle/float(total)))
sizes.append(float(acc/float(total)))
sizes.append(float(cme/float(total)))
sizes.append(float(rte/float(total)))
explode = (0.1, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')`

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('foo.png')
plt.show()



