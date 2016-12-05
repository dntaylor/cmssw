from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from httplib import HTTPException


import pickle
import sys,os,subprocess
import string


### Fabrice Couderc: script in development to hadd jobs and create datasetdefinion
### one arg = crab project area
### eos dir should be mounted first in current directory: use  eosmount eos

### log
#       13 July 2016 : first commit
#                      no protection, you should know what you are doing
#                      only work on eos
#                      campaign is named after crab project area (to be improved)

dirToCheck = sys.argv[1]

inputTnPCrab = pickle.load(open(dirToCheck+'.requestcache','rb'))

config = None
if inputTnPCrab.has_key('OriginalConfig'):
    config = inputTnPCrab['OriginalConfig'] 

print config


### mount eos directory
#print os.environ
#subprocess.call(['bash','-i','-c','eosumount eos'])
#subprocess.call(['bash','-i','-c','eosmount eos'])
#os.rmdir('eos')
#os.mkdir('eos')

dasDataset = string.split(config.Data.inputDataset,'/')[1]
crabName   = 'crab_' + config.General.requestName


proc = subprocess.Popen('ls eos/cms/%s/%s/%s/*/*/*root'%(config.Data.outLFNDirBase,dasDataset,crabName), shell=True, 
                        stdout=subprocess.PIPE)
out,err = proc.communicate()
filelistTmp = string.split(out,'\n')
#print filelist

filelist = {}
for f in filelistTmp:
    if len(f) > 0:
        jobIdTmp = string.split(f,'_')
        jobId = int(jobIdTmp[len(jobIdTmp)-1].split('.root')[0])
        filelist[jobId] = f


###################################################################
##############  Check STATUS
###################################################################
checkStatus = True
statusOut   = None
if checkStatus:
    try:
        statusOut = crabCommand('status', dir = dirToCheck )
    except:
        print "Crab command status failed ... "

if statusOut != None:
    filelistOnlyFinished = []
    for job in statusOut['jobs']:
        jobId = int(job)
        state = statusOut['jobs'][job]['State']
        print ' job %d : status: %s ' % (jobId,state) 
        if state == 'finished' :
            print '   -> adding  file jobID:  ', jobId
            filelistOnlyFinished.append( filelist[jobId] )
#        if state == 'transferring' and jobId in filelist.keys() : 
#            filelistOnlyFinished.append( filelist[jobId] )
#            print 'WARNING adding transferring file... you should remove them unless you have a good reason to.'


###################################################################
##############  Check REPORT
###################################################################
checkReport = True
reportOut   = None
nEvtsRead = 0
lumiProccessedFile = None
if checkReport:
    try:
        reportOut = crabCommand('report', dir = dirToCheck )
    except:
        print "Crab command report failed ... "
    nEvtsRead          = reportOut['numEventsRead']
    lumiProccessedFile = '%s/%s/%s' % (os.getcwd(),dirToCheck,'results/processedLumis.json')
    
#sys.exit(0)

###################################################################
##############  hAdd
###################################################################
outDir  = dirToCheck 
outFile = '%s/TnPTree_%s_%s.root' % (outDir,dasDataset,config.General.requestName)
print 'hadd will be saved to %s ' % outFile
print ' - if file is moved properly to eos one should remove it (not automated for now)'

dataset = {}
dataset['campaign'] = config.General.workArea
dataset['dataset']  = '%s_%s' % ( dasDataset, config.General.requestName )
dataset['file' ]    = '%s/%s' % (config.Data.outLFNDirBase,os.path.basename(outFile))
dataset['nEvts']    = nEvtsRead
dataset['lumiProcessedFile' ]  = lumiProccessedFile
dataset['lumi' ]    = -1
print dataset

haddCommand = ['hadd','-f',outFile]
haddCommand += filelistOnlyFinished

subprocess.call(haddCommand)
subprocess.call(['mv',outFile,'eos/cms/%s'%config.Data.outLFNDirBase])

