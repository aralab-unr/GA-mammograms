
##-------------------------------------------
import numpy as np
    
def read_mixed_csv(fileName = None,delimiter = None): 
    fid = open(fileName,'r')
    
    lineArray = cell(100,1)
    
    ##   larger than is needed)
    lineIndex = 1
    
    nextLine = fgetl(fid)
    
    while not nextLine==- 1 :

        lineArray[lineIndex] = nextLine
        lineIndex = lineIndex + 1
        nextLine = fgetl(fid)

    
    fid.close()
    
    lineArray = lineArray(np.arange(1,lineIndex - 1+1))
    
    for iLine in np.arange(1,lineIndex - 1+1).reshape(-1):
        lineData = textscan(lineArray[iLine],'%s','Delimiter',delimiter)
        lineData = lineData[0]
        if str(lineArray[iLine](end())) == str(delimiter):
            lineData[end() + 1] = ''
        lineArray[iLine,np.arange[1,np.asarray[lineData].size+1]] = lineData
    
    return lineArray
    
    return lineArray