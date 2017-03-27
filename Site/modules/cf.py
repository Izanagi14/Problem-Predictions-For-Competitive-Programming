
def create_json():

    from math import sqrt

    crawleddata = open("CrawledData.txt", "r")
    questions= {}
    name = list()
    
    for columns in ( raw.strip().split() for raw in crawleddata ):  
        questions[columns[0]] = list(columns[1:])
        name.append(columns[0])
    #print questions['QUEEN']
    #print len(questions['QUEEN'])
    
    #for j in range(0,5):
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
    with open("DataMadeJson.py","w") as kp:
    	kp.write("dataset = "+"\\"+"\n { \n")
    	for j in range(0,len(questions['QUEEN'])):
    		kp.write("'User"+str(j)+"':"+"{"+"\n")
    		for i in range(0,len(name)):
    			kp.write("\t\t"+"'"+name[i]+"':")
    			k = (questions[name[i]])
    			if j == 0:
    				if i == len(name)-1:
    					k[j] = k[j].strip(",")
    				kp.write(k[j].strip("["))
    			elif j == 4:
    				if i == len(name)-1:
    					k[j] = k[j].strip("]")
    				else:
    					k[j] = k[j].strip("]")+","
    				kp.write(k[j])
    			elif i != len(name)-1:
    				kp.write(k[j])
    			else:
    				kp.write(k[j].strip(","))
    			kp.write("\n")
    		if j!=4:
    			kp.write("},")
    		else:
    			kp.write("}")
    		kp.write("\n")
    
    	kp.write("\n"+"}")
    
