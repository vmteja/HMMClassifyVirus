#kindly replace the path with your pwd

import numpy
with open('C:/Users/Krishna Teja/Documents/Study Material/223 - Bio/project/code_dir/banana_all', 'r') as myfile:
    data=myfile.read().replace('\n', '')

n = float(len(data))
print "The total number of nucleotides are", n
c = 0.0
for i in list("actg"):
   c = float(data.count(i))
   print i, (c/n)

  
catch = numpy.arange(0, len(data), 3)  
startCodonPositions = []
stopCodonPositions = []

for i in catch:
        codon = data[i:i + 3]
        if codon == 'atg':
            startCodonPositions.append(i + 1)
        if codon == 'taa' or codon == 'tag' or codon == 'tga':
            stopCodonPositions.append(i + 1)
        
        

print "Frequency of start codon :- ", len(startCodonPositions)/(n/3)
            
print "Frequency of stop codon :- ", len(stopCodonPositions)/(n/3)
            

   
