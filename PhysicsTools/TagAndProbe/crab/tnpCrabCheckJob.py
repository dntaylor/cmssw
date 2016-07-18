from CRABClient.UserUtilities import config
from CRABAPI.RawCommand import crabCommand
from CRABClient.ClientExceptions import ClientException
from httplib import HTTPException


import pickle
import sys,os,subprocess
import string

#dirToCheck = 'crab_projects_elev2/crab_DYToLL_madgraph/'
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

#        print ' job %d : status: %s ' % (jobId,state) 
        if state == 'finished' :
#            print '   -> adding  file ', filelist[jobId] 
            filelistOnlyFinished.append( filelist[jobId] )


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
    


###################################################################
##############  hAdd
###################################################################
outDir  = dirToCheck 
outFile = '%s/TnPTree_%s_%s.root' % (outDir,dasDataset,config.General.requestName)
print outFile
#sys.exit(0)
haddCommand = ['hadd','-f',outFile]
haddCommand += filelistOnlyFinished

subprocess.call(haddCommand)
subprocess.call(['cp',outFile,'eos/cms/%s'%config.Data.outLFNDirBase])

dataset = {}
dataset['campaign'] = config.General.workArea
dataset['dataset']  = '%s_%s' % (dasDataset,config.General.requestName)
dataset['file' ]    = '%s/%s' % (config.Data.outLFNDirBase,os.path.basename(outFile))
dataset['nEvts']    = nEvtsRead
dataset['lumiProcessedFile' ]  = lumiProccessedFile
dataset['lumi' ]   = -1
print dataset

