#             INbreast Release 1.0
#Breast Research Group, INESC Porto, Portugal
#http://medicalresearch.inescporto.pt/breastresearch/
#         medicalresearch@inescporto.pt
import numpy as np
xlsdata = read_mixed_csv('../INBreast.csv',';')
biradsclass = xlsdata(np.arange(2,end()+1),8)
biradsclassFilename = xlsdata(np.arange(2,end()+1),6)
date = xlsdata(np.arange(2,end()+1),5)
sortedFilenames,idx = __builtint__.sorted(biradsclassFilename)
biradsclass = biradsclass(idx)
date = date(idx)
DIR = dir('.\\*.dcm')
nn = np.asarray(DIR).size
INbreast = cell(nn,8)
for k in np.arange(1,nn+1).reshape(-1):
    print(num2str(k))
    info = dicominfo(DIR(k).name)
    INbreast[k,1] = k
    lineData = textscan(DIR(k).name,'%s','Delimiter','_')
    lineData = lineData[0]
    #INbreast{k,2} is empty - name
    INbreast[k,3] = lineData[2]
    INbreast[k,4] = lineData[3]
    INbreast[k,5] = lineData[5]
    #INbreast{k,6} is empty - date
    INbreast[k,7] = DIR(k).name
    #INbreast{k,8} is empty - birads

dicomFilename = INbreast(:,7)
sortedFilenames,idx = __builtint__.sorted(dicomFilename)
INbreast = INbreast(idx,:)
INbreast[:,8] = biradsclass
INbreast[:,6] = date
#organize by cases: a case is the set of all views for a single patient for
#a single day. We have 117 cases.
#IMPORTANT NOTE: in the Academic Radiology publication the number is wrongly presented as 115.
PatientInfo = strcat(INbreast(:,3),INbreast(:,6))
uniquePatientInfo = unique(PatientInfo)
for n in np.arange(1,len(uniquePatientInfo)+1).reshape(-1):
    idx = find(ismember(PatientInfo,uniquePatientInfo(n)) == 1)
    INbreastCase[n] = idx

#just usage example
caseBirads = - 10 * np.ones((len(INbreastCase),1))
caseLen = np.zeros((len(INbreastCase),1))
for n in np.arange(1,len(INbreastCase)+1).reshape(-1):
    print(np.array(['case ',num2str(n)]))
    curr_case = INbreastCase[n]
    curr_biradsSet = []
    caseLen[n] = len(curr_case)
    for i in np.arange(1,len(curr_case)+1).reshape(-1):
        filename = INbreast[curr_case(i),7]
        birads = INbreast[curr_case(i),8]
        curr_biradsSet = np.array([curr_biradsSet,sscanf(birads,'%g')])
        #img=dicomread(filename);
#imshow(img,[]);
    caseBirads[n] = np.amax(curr_biradsSet)

#[caseLen caseBirads]
figure
hist(caseLen,np.arange(1,8+1))
figure
hist(caseBirads,np.arange(0,6+1))