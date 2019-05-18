import os

w=[0]
age=[0]
hours=[0]
input_age=0
avg_age=0
workclass={'Private':0, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0, 'State-gov':0, 'Without-pay':0}
key_workclass=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']
education={'Bachelors':0, 'Some-college':0, '11th':0, 'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, '9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, '10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
key_education=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
maritalstatus={'Married-civ-spouse':0, 'Divorced':0, 'Never-married':0, 'Separated':0, 'Widowed':0, 'Married-spouse-absent':0, 'Married-AF-spouse':0}
key_maritalstatus=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation={'Tech-support':0, 'Craft-repair':0, 'Other-service':0, 'Sales':0, 'Exec-managerial':0, 'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
key_occupation=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
race={'White':0, 'Asian-Pac-Islander':0, 'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
key_race=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex={'Female':0, 'Male':0}
key_sex=['Female', 'Male']
count_mistake=0
input_hours=0
avg_hours=0
nativecountry={'United-States':0, 'Cambodia':0, 'England':0, 'Puerto-Rico':0, 'Canada':0, 'Germany':0, 'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 'Ireland':0, 'France':0, 'Dominican-Republic':0, 'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0, 'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
key_nativecountry=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
salary={'>50K\n':0, '<=50K\n':0}
count=0
transage=[]
transhour=[]
temp=[]
c=1
for num in range(1,95):
    w.append(0)

    with open("income.train.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                       
                workclass={'Private':0, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0, 'State-gov':0, 'Without-pay':0}
                key_workclass=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']
                education={'Bachelors':0, 'Some-college':0, '11th':0, 'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, '9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, '10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
                key_education=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
                maritalstatus={'Married-civ-spouse':0, 'Divorced':0, 'Never-married':0, 'Separated':0, 'Widowed':0, 'Married-spouse-absent':0, 'Married-AF-spouse':0}
                key_maritalstatus=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
                occupation={'Tech-support':0, 'Craft-repair':0, 'Other-service':0, 'Sales':0, 'Exec-managerial':0, 'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
                key_occupation=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
                race={'White':0, 'Asian-Pac-Islander':0, 'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
                key_race=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
                sex={'Female':0, 'Male':0}
                key_sex=['Female', 'Male']
                nativecountry={'United-States':0, 'Cambodia':0, 'England':0, 'Puerto-Rico':0, 'Canada':0, 'Germany':0, 'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 'Ireland':0, 'France':0, 'Dominican-Republic':0, 'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0, 'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
                key_nativecountry=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
                salary={'>50K\n':0, '<=50K\n':0}
                y=0
                age_y=0
                workclass_y=0
                education_y=0
                maritalstatus_y=0
                occupation_y=0
                race_y=0
                sex_y=0
                hours_y=0
                nativecountry_y=0
                count=count+1
                a=line.split(", ")
                
                input_age=int(a[0])
                
                avg_age=(avg_age*(count-1)+input_age)/count
                if input_age>avg_age:
                    age=1
                else:
                    age=0
                    '''
                print(input_age)
                print(age)
                print(avg_age)
                print("count",count)
                '''
                
                transage.append(input_age)
               
                workclass[a[1]]=1
                
                education[a[2]]=1
                
                maritalstatus[a[3]]=1
                occupation[a[4]]=1
                race[a[5]]=1
                sex[a[6]]=1
                
                
                input_hours=int(a[7])
                transhour.append(input_hours)
                
                avg_hours=(avg_hours*(count-1)+input_hours)/count
                if input_hours>avg_hours:
                    hours=1                
                else:
                    hours=0
                
                nativecountry[a[8]]=1
                salary[a[9]]=1
                c=c+1
                    
                '''print (1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+1+1+len(nativecountry))'''

            '''    age_y=w[0]*age
                for num in range(0,len(workclass)):
                    workclass_y=w[num+1]*workclass[key_workclass[num]]+workclass_y
                for num in range(0,len(education)):
                    education_y=w[num+1+len(workclass)]*education[key_education[num]]+education_y
                for num in range(0,len(maritalstatus)):
                    maritalstatus_y=w[num+1+len(workclass)+len(education)]*maritalstatus[key_maritalstatus[num]]+maritalstatus_y
                for num in range(0,len(occupation)):
                    occupation_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)]*occupation[key_occupation[num]]+occupation_y
                for num in range(0,len(race)):
                    race_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]*race[key_race[num]]+race_y
                for num in range(0,len(sex)):
                    sex_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]*sex[key_sex[num]]+sex_y
                
                hours_y=w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]*hours
                for num in range(0,len(nativecountry)):
                    nativecountry_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]*nativecountry[key_nativecountry[num]]+nativecountry_y
                y=age_y+workclass_y+education_y+maritalstatus_y+occupation_y+race_y+sex_y+hours_y+nativecountry_y
                if y>0:
                    y=1
                if y<=0:
                    y=-1            
                  
                if (y==1 and salary['<=50K\n'])or(y==-1 and salary['>50K\n']):
                    count_mistake=count_mistake+1
                    w[0]=y*-1*age+w[0]
                    for num in range(0,len(workclass)):
                        w[num+1]=y*-1*workclass[key_workclass[num]]+w[num+1]
                    for num in range(0,len(education)):
                        w[num+1+len(workclass)]=y*-1*education[key_education[num]]+w[num+1+len(workclass)]
                    for num in range(0,len(maritalstatus)):
                        w[num+1+len(workclass)+len(education)]=y*-1*maritalstatus[key_maritalstatus[num]]+w[num+1+len(workclass)+len(education)]
                    for num in range(0,len(occupation)):
                        w[num+1+len(workclass)+len(education)+len(maritalstatus)]=y*-1*occupation[key_occupation[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)]
                    for num in range(0,len(race)):
                        w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]=y*-1*race[key_race[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]
                    for num in range(0,len(sex)):
                        w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]=y*-1*sex[key_sex[num]]+w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]
                    w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]=y*-1*hours+w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]
                    for num in range(0,len(nativecountry)):
                        w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]=y*-1*nativecountry[key_nativecountry[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]
     
                
                if y==1 and salary['<=50K\n']:
                    print("in")

                if y==-1 and salary['>50K\n']:
                    print("out")
                print(salary) 
                print(y)
                print(w)
            print(w)
            print(count_mistake)'''
            count_mistake=0
transage=list(set(transage))
transage.sort()
transhour=list(set(transhour))
transhour.sort()
'''print(transage)
print(transhour)'''
'''          
            print(transage)
            with open("income.test.txt","r") as f:
                lines=f.readlines()
                for line in lines:
                
                    workclass={'Private':0, 'Self-emp-not-inc':0, 'Self-emp-inc':0, 'Federal-gov':0, 'Local-gov':0, 'State-gov':0, 'Without-pay':0, 'Never-worked':0}
                    key_workclass=['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
                    education={'Bachelors':0, 'Some-college':0, '11th':0, 'HS-grad':0, 'Prof-school':0, 'Assoc-acdm':0, 'Assoc-voc':0, '9th':0, '7th-8th':0, '12th':0, 'Masters':0, '1st-4th':0, '10th':0, 'Doctorate':0, '5th-6th':0, 'Preschool':0}
                    key_education=['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
                    maritalstatus={'Married-civ-spouse':0, 'Divorced':0, 'Never-married':0, 'Separated':0, 'Widowed':0, 'Married-spouse-absent':0, 'Married-AF-spouse':0}
                    key_maritalstatus=['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
                    occupation={'Tech-support':0, 'Craft-repair':0, 'Other-service':0, 'Sales':0, 'Exec-managerial':0, 'Prof-specialty':0, 'Handlers-cleaners':0, 'Machine-op-inspct':0, 'Adm-clerical':0, 'Farming-fishing':0, 'Transport-moving':0, 'Priv-house-serv':0, 'Protective-serv':0, 'Armed-Forces':0}
                    key_occupation=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
                    race={'White':0, 'Asian-Pac-Islander':0, 'Amer-Indian-Eskimo':0, 'Other':0, 'Black':0}
                    key_race=['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
                    sex={'Female':0, 'Male':0}
                    key_sex=['Female', 'Male']
                    nativecountry={'United-States':0, 'Cambodia':0, 'England':0, 'Puerto-Rico':0, 'Canada':0, 'Germany':0, 'Outlying-US(Guam-USVI-etc)':0, 'India':0, 'Japan':0, 'Greece':0, 'South':0, 'China':0, 'Cuba':0, 'Iran':0, 'Honduras':0, 'Philippines':0, 'Italy':0, 'Poland':0, 'Jamaica':0, 'Vietnam':0, 'Mexico':0, 'Portugal':0, 'Ireland':0, 'France':0, 'Dominican-Republic':0, 'Laos':0, 'Ecuador':0, 'Taiwan':0, 'Haiti':0, 'Columbia':0, 'Hungary':0, 'Guatemala':0, 'Nicaragua':0, 'Scotland':0, 'Thailand':0, 'Yugoslavia':0, 'El-Salvador':0, 'Trinadad&Tobago':0, 'Peru':0, 'Hong':0, 'Holand-Netherlands':0}
                    key_nativecountry=['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
                    salary={'>50K\n':0, '<=50K\n':0}
                    y=0
                    age_y=0
                    workclass_y=0
                    education_y=0
                    maritalstatus_y=0
                    occupation_y=0
                    race_y=0
                    sex_y=0
                    hours_y=0
                    nativecountry_y=0
                    count=count+1
                    input_age=int(line.split(", ")[0])
                    avg_age=(avg_age*(count-1)+input_age)/count
                    
                    if input_age>=avg_age:
                        age=1
                    else:
                        age=0
                    a=line.split(", ")
                    workclass[a[1]]=1
                    
                    education[a[2]]=1
                    
                    maritalstatus[a[3]]=1
                    occupation[a[4]]=1
                    race[a[5]]=1
                    sex[a[6]]=1
                    
                    
                    input_hours=int(a[7])
                    avg_hours=(avg_hours*(count-1)+input_hours)/count
                    if input_hours>=avg_hours:
                        hours=1
                        
                    else:
                        hours=0
                    
                    nativecountry[a[8]]=1
                    salary[a[9]]=1

                    
                        
                    

                    age_y=w[0]*age
                    for num in range(0,len(workclass)):
                        workclass_y=w[num+1]*workclass[key_workclass[num]]+workclass_y
                    for num in range(0,len(education)):
                        education_y=w[num+1+len(workclass)]*education[key_education[num]]+education_y
                    for num in range(0,len(maritalstatus)):
                        maritalstatus_y=w[num+1+len(workclass)+len(education)]*maritalstatus[key_maritalstatus[num]]+maritalstatus_y
                    for num in range(0,len(occupation)):
                        occupation_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)]*occupation[key_occupation[num]]+occupation_y
                    for num in range(0,len(race)):
                        race_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]*race[key_race[num]]+race_y
                    for num in range(0,len(sex)):
                        sex_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]*sex[key_sex[num]]+sex_y
                    
                    hours_y=w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]*hours
                    for num in range(0,len(nativecountry)):
                        nativecountry_y=w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]*nativecountry[key_nativecountry[num]]+nativecountry_y
                    y=age_y+workclass_y+education_y+maritalstatus_y+occupation_y+race_y+sex_y+hours_y+nativecountry_y
                    if y>0:
                        y=1
                    if y<=0:
                        y=-1
                    
                        
                    if (y==1 and salary['<=50K\n'])or(y==-1 and salary['>50K\n']) :
                        count_mistake=count_mistake+1
                        
                        w[0]=y*-1*age+w[0]
                        for num in range(0,len(workclass)):
                            w[num+1]=y*-1*workclass[key_workclass[num]]+w[num+1]
                        for num in range(0,len(education)):
                            w[num+1+len(workclass)]=y*-1*education[key_education[num]]+w[num+1+len(workclass)]
                        for num in range(0,len(maritalstatus)):
                            w[num+1+len(workclass)+len(education)]=y*-1*maritalstatus[key_maritalstatus[num]]+w[num+1+len(workclass)+len(education)]
                        for num in range(0,len(occupation)):
                            w[num+1+len(workclass)+len(education)+len(maritalstatus)]=y*-1*occupation[key_occupation[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)]
                        for num in range(0,len(race)):
                            w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]=y*-1*race[key_race[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)]
                        for num in range(0,len(sex)):
                            w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]=y*-1*sex[key_sex[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)]
                        w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]=y*-1*hours+w[1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)]
                        for num in range(0,len(nativecountry)):
                            w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]=y*-1*nativecountry[key_nativecountry[num]]+w[num+1+len(workclass)+len(education)+len(maritalstatus)+len(occupation)+len(race)+len(sex)+1]
                       
            print(w)
            print(count_mistake)'''
          
ag=[]
for i in range(len(transage)):
    ag.append(0)
ho=[]
for i in range(len(transhour)):
    ho.append(0)
wc=[]
for i in range(len(workclass)):
    wc.append(0)
ed=[]
for i in range(len(education)):
    ed.append(0)
ma=[]
for i in range(len(maritalstatus)):
    ma.append(0)
oc=[]
for i in range(len(occupation)):
    oc.append(0)
ra=[]
for i in range(len(race)):
    ra.append(0)
se=[]
for i in range(len(sex)):
    se.append(0)
na=[]
for i in range(len(nativecountry)):
    na.append(0)
m=len(transage)+len(transhour)+len(key_workclass)+len(key_education)+len(key_maritalstatus)+len(key_occupation)+len(key_race)+len(key_sex)+len(key_nativecountry)
'''print(len(transage))
print(len(transhour))
print(len(key_workclass))
print(len(key_education))
print(len(key_maritalstatus))
print(len(key_occupation))
print(len(key_race))
print(len(key_nativecountry))
print(m)'''
with open("income.test.txt","r") as f:
    lines=f.readlines()
    fp=open("outputest.txt","a")
    for con in range(1,m+1):
            strc=str(con)
            fp.write(strc)
            fp.write(",")
    fp.write("class\n")
    fp.close()
    for line in lines:
        fp=open("outputest.txt","a")
        for i in range(len(ag)):
            ag[i]=0
        for i in range(len(wc)):
            wc[i]=0
        for i in range(len(ed)):
            ed[i]=0
        for i in range(len(ma)):
            ma[i]=0
        for i in range(len(oc)):
            oc[i]=0
        for i in range(len(ra)):
            ra[i]=0
        for i in range(len(se)):
            se[i]=0
        
        for i in range(len(ho)):
            ho[i]=0
        for i in range(len(na)):
            na[i]=0
        a=line.split(", ")       
        input_age=int(a[0])
        for num in range(len(transage)):
            if(transage[num]==input_age):
                ag[num]=1
        for num in  range(len(ag)):
            stra=str(ag[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(wc)):
            if key_workclass[num]==a[1]:
                wc[num]=1
        for num in  range(len(wc)):
            stra=str(wc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ed)):
            if key_education[num]==a[2]:
                ed[num]=1
        for num in  range(len(ed)):
            stra=str(ed[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ma)):
            if key_maritalstatus[num]==a[3]:
                ma[num]=1
        for num in  range(len(ma)):
            stra=str(ma[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(oc)):
            if key_occupation[num]==a[4]:
                oc[num]=1
        for num in  range(len(oc)):
            stra=str(oc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ra)):
            if key_race[num]==a[5]:
                ra[num]=1
        for num in  range(len(ra)):
            stra=str(ra[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(se)):
            if key_sex[num]==a[6]:
                se[num]=1
        for num in  range(len(se)):
            stra=str(se[num])
            fp.write(stra)
            fp.write(",")     
               
                
        input_hours=int(a[7])
        for num in range(len(transhour)):
            if(transhour[num]==input_hours):
                ho[num]=1
        for num in  range(len(ho)):
            stra=str(ho[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(na)):
            if key_nativecountry[num]==a[8]:
                na[num]=1
        for num in range(len(na)):
            stra=str(na[num])
            fp.write(stra)
            fp.write(",")            
                               
        if a[9]=='>50K\n':
            fp.write("1")
        else:
            fp.write("0")
        
        fp.write("\n")
        
fp.close()

with open("income.train.txt","r") as f:
    lines=f.readlines()
    fp=open("outputrain.txt","a")
    for con in range(1,m+1):
            strc=str(con)
            fp.write(strc)
            fp.write(",")
    fp.write("class\n")
    fp.close()
    for line in lines:
        fp=open("outputrain.txt","a")
        for i in range(len(ag)):
            ag[i]=0
        for i in range(len(wc)):
            wc[i]=0
        for i in range(len(ed)):
            ed[i]=0
        for i in range(len(ma)):
            ma[i]=0
        for i in range(len(oc)):
            oc[i]=0
        for i in range(len(ra)):
            ra[i]=0
        for i in range(len(se)):
            se[i]=0
        
        for i in range(len(ho)):
            ho[i]=0
        for i in range(len(na)):
            na[i]=0
        a=line.split(", ")       
        input_age=int(a[0])
        for num in range(len(transage)):
            if(transage[num]==input_age):
                ag[num]=1
        for num in  range(len(ag)):
            stra=str(ag[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(wc)):
            if key_workclass[num]==a[1]:
                wc[num]=1
        for num in  range(len(wc)):
            stra=str(wc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ed)):
            if key_education[num]==a[2]:
                ed[num]=1
        for num in  range(len(ed)):
            stra=str(ed[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ma)):
            if key_maritalstatus[num]==a[3]:
                ma[num]=1
        for num in  range(len(ma)):
            stra=str(ma[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(oc)):
            if key_occupation[num]==a[4]:
                oc[num]=1
        for num in  range(len(oc)):
            stra=str(oc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ra)):
            if key_race[num]==a[5]:
                ra[num]=1
        for num in  range(len(ra)):
            stra=str(ra[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(se)):
            if key_sex[num]==a[6]:
                se[num]=1
        for num in  range(len(se)):
            stra=str(se[num])
            fp.write(stra)
            fp.write(",")     
               
                
        input_hours=int(a[7])
        for num in range(len(transhour)):
            if(transhour[num]==input_hours):
                ho[num]=1
        for num in  range(len(ho)):
            stra=str(ho[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(na)):
            if key_nativecountry[num]==a[8]:
                na[num]=1
        for num in range(len(na)):
            stra=str(na[num])
            fp.write(stra)
            fp.write(",")            
                               
        if a[9]=='>50K\n':
            fp.write("1")
        else:
            fp.write("0")
        
        fp.write("\n")
        
fp.close()

with open("income.dev.txt","r") as f:
    lines=f.readlines()
    fp=open("outputdev.txt","a")
    for con in range(1,m+1):
            strc=str(con)
            fp.write(strc)
            fp.write(",")
    fp.write("class\n")
    fp.close()
    for line in lines:
        fp=open("outputdev.txt","a")
        for i in range(len(ag)):
            ag[i]=0
        for i in range(len(wc)):
            wc[i]=0
        for i in range(len(ed)):
            ed[i]=0
        for i in range(len(ma)):
            ma[i]=0
        for i in range(len(oc)):
            oc[i]=0
        for i in range(len(ra)):
            ra[i]=0
        for i in range(len(se)):
            se[i]=0
        
        for i in range(len(ho)):
            ho[i]=0
        for i in range(len(na)):
            na[i]=0
        a=line.split(", ")       
        input_age=int(a[0])
        for num in range(len(transage)):
            if(transage[num]==input_age):
                ag[num]=1
        for num in  range(len(ag)):
            stra=str(ag[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(wc)):
            if key_workclass[num]==a[1]:
                wc[num]=1
        for num in  range(len(wc)):
            stra=str(wc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ed)):
            if key_education[num]==a[2]:
                ed[num]=1
        for num in  range(len(ed)):
            stra=str(ed[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ma)):
            if key_maritalstatus[num]==a[3]:
                ma[num]=1
        for num in  range(len(ma)):
            stra=str(ma[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(oc)):
            if key_occupation[num]==a[4]:
                oc[num]=1
        for num in  range(len(oc)):
            stra=str(oc[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(ra)):
            if key_race[num]==a[5]:
                ra[num]=1
        for num in  range(len(ra)):
            stra=str(ra[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(se)):
            if key_sex[num]==a[6]:
                se[num]=1
        for num in  range(len(se)):
            stra=str(se[num])
            fp.write(stra)
            fp.write(",")     
               
                
        input_hours=int(a[7])
        for num in range(len(transhour)):
            if(transhour[num]==input_hours):
                ho[num]=1
        for num in  range(len(ho)):
            stra=str(ho[num])
            fp.write(stra)
            fp.write(",")
        for num in range(len(na)):
            if key_nativecountry[num]==a[8]:
                na[num]=1
        for num in range(len(na)):
            stra=str(na[num])
            fp.write(stra)
            fp.write(",")            
                               
        if a[9]=='>50K\n':
            fp.write("1")
        else:
            fp.write("0")
        
        fp.write("\n")
        
fp.close()


        
            
        

        
        
        
            
            
            
           
         
            
            
       
        
       
