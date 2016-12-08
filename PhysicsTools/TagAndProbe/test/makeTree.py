import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
import sys

process = cms.Process("tnp")

###################################################################
## argument line options
###################################################################
varOptions = VarParsing('analysis')
varOptions.register(
    "isMC", True,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Compute MC efficiencies"
    )

varOptions.register(
    "doEleID", True,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Include tree for photon ID SF"
    )

varOptions.register(
    "doPhoID", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Include tree for photon ID SF"
    )

varOptions.register(
    "doTrigger", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Include tree for Trigger SF"
    )

varOptions.register(
    "doRECO", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Include tree for Reco SF"
    )

varOptions.register(
    "useCalibEn", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "use EGM smearer to calibrate photon and electron energy"
    )

varOptions.register(
    "isAOD", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "switch to run other AOD (for RECO SFs)"
    )

#### HLTname is HLT2 in reHLT samples
varOptions.register(
    "HLTname", "HLT",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "HLT process name (default HLT)"
    )

varOptions.register(
    "GT","auto",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Global Tag to be used"
    )



varOptions.parseArguments()


###################################################################
## Define TnP inputs 
###################################################################

options = dict()
options['useAOD']               = cms.bool(varOptions.isAOD)

options['HLTProcessName']       = varOptions.HLTname

### set input collections
options['ELECTRON_COLL']        = "slimmedElectrons"
options['PHOTON_COLL']          = "slimmedPhotons"
options['SUPERCLUSTER_COLL']    = "reducedEgamma:reducedSuperClusters" ### not used in AOD
if options['useAOD']:
    options['ELECTRON_COLL']        = "gedGsfElectrons"


options['ELECTRON_CUTS']        = "ecalEnergy*sin(superClusterPosition.theta)>5.0 &&  (abs(-log(tan(superClusterPosition.theta/2)))<2.5)"
options['SUPERCLUSTER_CUTS']    = "abs(eta)<2.5 &&  et>5.0"
options['PHOTON_CUTS']          = "(abs(-log(tan(superCluster.position.theta/2)))<=2.5) && pt> 10"
options['ELECTRON_TAG_CUTS']    = "(abs(-log(tan(superCluster.position.theta/2)))<=2.1) && !(1.4442<=abs(-log(tan(superClusterPosition.theta/2)))<=1.566) && pt >= 30.0"

options['MAXEVENTS']            = cms.untracked.int32(varOptions.maxEvents) 
options['DoTrigger']            = cms.bool( varOptions.doTrigger )
options['DoRECO']               = cms.bool( varOptions.doRECO    )
options['DoEleID']              = cms.bool( varOptions.doEleID   )
options['DoPhoID']              = cms.bool( varOptions.doPhoID   )

options['OUTPUTEDMFILENAME']    = 'edmFile.root'
options['DEBUG']                = cms.bool(False)
options['isMC']                 = cms.bool(False)
options['EVENTSToPROCESS']      = cms.untracked.VEventRange()
options['UseCalibEn']           = varOptions.useCalibEn

if (varOptions.isMC):
    options['isMC']                = cms.bool(True)
    options['OUTPUT_FILE_NAME']    = "TnPTree_mc.root"
#    options['TnPPATHS']            = cms.vstring("HLT*")
#    options['TnPHLTTagFilters']    = cms.vstring()
#    options['TnPHLTProbeFilters']  = cms.vstring()
#    options['HLTFILTERTOMEASURE']  = cms.vstring("")
    options['TnPPATHS']            = cms.vstring("HLT_Ele27_eta2p1_WPTight_Gsf_v*")
    options['TnPHLTTagFilters']    = cms.vstring("hltEle27erWPTightGsfTrackIsoFilter")
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("hltEle27erWPTightGsfTrackIsoFilter")   
    options['GLOBALTAG']           = 'auto:run2_mc'
else:
    options['OUTPUT_FILE_NAME']    = "TnPTree_data.root"
    options['TnPPATHS']            = cms.vstring("HLT_Ele27_eta2p1_WPTight_Gsf_v*")
    options['TnPHLTTagFilters']    = cms.vstring("hltEle27erWPTightGsfTrackIsoFilter")
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("hltEle27erWPTightGsfTrackIsoFilter")
    options['GLOBALTAG']           = 'auto:run2_data'

if varOptions.GT != "auto" :
    options['GLOBALTAG'] = varOptions.GT


###################################################################
## Inputs for test
###################################################################
filesMC =  cms.untracked.vstring(
    '/store/mc/RunIISpring16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14_ext1-v1/20000/00071E92-6F55-E611-B68C-0025905A6066.root',
    '/store/mc/RunIISpring16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14_ext1-v1/20000/00384BBA-D455-E611-B32C-0CC47A4D7600.root',
       '/store/mc/RunIISpring16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14_ext1-v1/20000/00684732-3155-E611-B794-0CC47A4D7640.root',
    )

filesData =  cms.untracked.vstring( 
    '/store/data/Run2016B/SingleElectron/MINIAOD/23Sep2016-v2/80000/08A02DC3-608C-E611-ADA5-0025905B85B6.root',
    '/store/data/Run2016B/SingleElectron/MINIAOD/23Sep2016-v2/80000/0C9C7188-708C-E611-A4D7-0025907DE266.root',
    '/store/data/Run2016B/SingleElectron/MINIAOD/23Sep2016-v2/80000/14415255-6B8C-E611-87D3-002590E3A212.root',
    )


if options['useAOD']:
    filesMC = cms.untracked.vstring(
        '/store/mc/RunIISummer16DR80Premix/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/110000/00389155-1FB1-E611-A88A-001E674FB207.root',
        '/store/mc/RunIISummer16DR80Premix/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/110000/003C5386-7DB1-E611-9FD3-A0000420FE80.root',
        '/store/mc/RunIISummer16DR80Premix/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/110000/009A908B-9BB1-E611-936F-848F69FD4CB2.root',
        '/store/mc/RunIISummer16DR80Premix/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/AODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v1/110000/00B371BA-8BB1-E611-8D83-24BE05CEEC21.root',
       )
    filesData = cms.untracked.vstring(
        '/store/data/Run2016B/SingleElectron/AOD/23Sep2016-v2/80000/0220DA0C-648C-E611-A9AA-0CC47A78A2F6.root',
        '/store/data/Run2016B/SingleElectron/AOD/23Sep2016-v2/80000/022664AC-618C-E611-B6D4-0CC47A78A3EC.root',
        '/store/data/Run2016B/SingleElectron/AOD/23Sep2016-v2/80000/02DE6A4E-6A8C-E611-9241-00259048AC76.root',
        )
    
options['INPUT_FILE_NAME'] = filesData
if varOptions.isMC:
    options['INPUT_FILE_NAME'] = filesMC


###################################################################
## import TnP tree maker pythons and configure for AODs
###################################################################
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff') 

process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options['GLOBALTAG'] , '')


if options['useAOD']:
    import PhysicsTools.TagAndProbe.treeMakerOptionsAOD_cfi as tnpTreeMaker
else: 
    import PhysicsTools.TagAndProbe.treeMakerOptions_cfi as tnpTreeMaker

tnpTreeMaker.setModules(process,options)


import PhysicsTools.TagAndProbe.treeContent_cfi as tnpVars
if options['useAOD']:
    tnpVars.setupTnPVariablesForAOD()

if not varOptions.isMC:
    tnpVars.mcTruthCommonStuff = cms.PSet(
        isMC = cms.bool(False)
        )

    



###################################################################
## Init and Load
###################################################################

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.source = cms.Source("PoolSource",
                            fileNames = options['INPUT_FILE_NAME'],
                            eventsToProcess = options['EVENTSToPROCESS']
                            )

process.maxEvents = cms.untracked.PSet( input = options['MAXEVENTS'])



###################################################################
## ID
###################################################################
print 'Input electron collection: ', options['ELECTRON_COLL']
print 'Input photon   collection: ', options['PHOTON_COLL'  ]

import PhysicsTools.TagAndProbe.electronIDModules_cfi as egmEleID
import PhysicsTools.TagAndProbe.photonIDModules_cfi   as egmPhoID
egmEleID.setIDs(process, options)
egmPhoID.setIDs(process, options)
        


###################################################################
## SEQUENCES
###################################################################
tnpTreeMaker.setSequences(process,options)
process.cand_sequence = cms.Sequence( process.init_sequence + process.tag_sequence )

if (options['DoEleID']):
    process.cand_sequence += process.ele_sequence
    print "  -- Producing electron SF tree    -- "

if (options['DoPhoID']):
    process.cand_sequence += process.pho_sequence
    print "  -- Producing photon SF tree      -- "

if (options['DoTrigger']):
    process.cand_sequence += process.hlt_sequence
    print "  -- Producing HLT efficiency tree -- "

if (options['DoRECO']):
    process.cand_sequence += process.sc_sequence
    print "  -- Producing RECO SF tree        -- "



###################################################################
## TnP PAIRS
###################################################################
process.allTagsAndProbes = cms.Sequence()

if (options['DoTrigger']):
    process.allTagsAndProbes *= process.tagTightHLT

if (options['DoRECO']):
    process.allTagsAndProbes *= process.tagTightSC

if (options['DoEleID']):
    process.allTagsAndProbes *= process.tagTightEleID

if (options['DoPhoID']):
    process.allTagsAndProbes *= process.tagTightPhoID




process.mc_sequence = cms.Sequence()


##########################################################################
## TREE MAKER OPTIONS
##########################################################################    
process.GsfElectronToTrigger = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                              tnpVars.CommonStuffForGsfElectronProbe, tnpVars.mcTruthCommonStuff,
                                              tagProbePairs = cms.InputTag("tagTightHLT"),
                                              arbitration   = cms.string("HighestPt"),
                                              flags         = cms.PSet(passingHLT     = cms.InputTag("goodElectronsMeasureHLT"),
                                                                       passingLoose   = cms.InputTag("goodElectronsPROBECutBasedLoose" ),
                                                                       passingMedium  = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                       passingTight   = cms.InputTag("goodElectronsPROBECutBasedTight" ),
                                                                       passingHLTsafe = cms.InputTag("goodElectronsPROBEHLTsafe"),
                                                                       ),                                               
                                              allProbes     = cms.InputTag("goodElectronsProbeMeasureHLT"),
                                              )

process.GsfElectronToSC = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                         tnpVars.mcTruthCommonStuff, tnpVars.CommonStuffForSuperClusterProbe, 
                                         tagProbePairs = cms.InputTag("tagTightSC"),
                                         arbitration   = cms.string("HighestPt"),
                                         flags         = cms.PSet(passingRECO   = cms.InputTag("GsfMatchedSuperClusterCands", "superclusters")
                                                                  ),                                               
                                         allProbes     = cms.InputTag("goodSuperClustersHLT"),
                                         )

process.GsfElectronToEleID = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                            tnpVars.mcTruthCommonStuff, tnpVars.CommonStuffForGsfElectronProbe,
                                            tagProbePairs = cms.InputTag("tagTightEleID"),
                                            arbitration   = cms.string("HighestPt"),
                                            flags         = cms.PSet(passingVeto    = cms.InputTag("goodElectronsPROBECutBasedVeto"  ),
                                                                     passingLoose   = cms.InputTag("goodElectronsPROBECutBasedLoose" ),
                                                                     passingMedium  = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                     passingTight   = cms.InputTag("goodElectronsPROBECutBasedTight" ),
                                                                     passingVeto80X    = cms.InputTag("goodElectronsPROBECutBasedVeto80X"  ),
                                                                     passingLoose80X   = cms.InputTag("goodElectronsPROBECutBasedLoose80X" ),
                                                                     passingMedium80X  = cms.InputTag("goodElectronsPROBECutBasedMedium80X"),
                                                                     passingTight80X   = cms.InputTag("goodElectronsPROBECutBasedTight80X" ),
                                                                     passingMVA80Xwp90 = cms.InputTag("goodElectronsPROBEMVA80Xwp90" ),
                                                                     passingMVA80Xwp80 = cms.InputTag("goodElectronsPROBEMVA80Xwp80" ),
                                                                     passingHLTsafe = cms.InputTag("goodElectronsPROBEHLTsafe"),
                                                                     ),                                               
                                            allProbes     = cms.InputTag("goodElectronsProbeHLT"),
                                            )

process.GsfElectronToPhoID = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                            tnpVars.mcTruthCommonStuff, tnpVars.CommonStuffForPhotonProbe,
                                            tagProbePairs = cms.InputTag("tagTightPhoID"),
                                            arbitration   = cms.string("HighestPt"),
                                            flags         = cms.PSet(passingLoose     = cms.InputTag("goodPhotonsPROBECutBasedLoose"),
                                                                     passingMedium    = cms.InputTag("goodPhotonsPROBECutBasedMedium"),
                                                                     passingTight     = cms.InputTag("goodPhotonsPROBECutBasedTight"),
                                                                     passingLoose80X  = cms.InputTag("goodPhotonsPROBECutBasedLoose80X"),
                                                                     passingMedium80X = cms.InputTag("goodPhotonsPROBECutBasedMedium80X"),
                                                                     passingTight80X  = cms.InputTag("goodPhotonsPROBECutBasedTight80X"),
                                                                     passingMVA           = cms.InputTag("goodPhotonsPROBEMVA"),
                                                                     passingMVA80Xwp90    = cms.InputTag("goodPhotonsPROBEMVA80Xwp90"),
                                                                     passingMVA80Xwp80    = cms.InputTag("goodPhotonsPROBEMVA80Xwp80"),
                                                                     ),                                                                                           
                                            allProbes     = cms.InputTag("goodPhotonsProbeHLT"),
                                            )

if (varOptions.isMC):
    process.GsfElectronToTrigger.eventWeight = cms.InputTag("generator")
    process.GsfElectronToEleID.eventWeight   = cms.InputTag("generator")
    process.GsfElectronToPhoID.eventWeight   = cms.InputTag("generator")
    process.GsfElectronToSC.eventWeight      = cms.InputTag("generator")
    process.GsfElectronToTrigger.PUWeightSrc = cms.InputTag("pileupReweightingProducer","pileupWeights")
    process.GsfElectronToEleID.PUWeightSrc   = cms.InputTag("pileupReweightingProducer","pileupWeights")
    process.GsfElectronToPhoID.PUWeightSrc   = cms.InputTag("pileupReweightingProducer","pileupWeights")
    process.GsfElectronToSC.PUWeightSrc      = cms.InputTag("pileupReweightingProducer","pileupWeights")


process.tree_sequence = cms.Sequence()
if (options['DoTrigger']):
    process.tree_sequence *= process.GsfElectronToTrigger

if (options['DoRECO']):
    process.tree_sequence *= process.GsfElectronToSC

if (options['DoEleID']):
    process.tree_sequence *= process.GsfElectronToEleID

if (options['DoPhoID']):
    process.tree_sequence *= process.GsfElectronToPhoID

##########################################################################
## PATHS
##########################################################################

process.out = cms.OutputModule("PoolOutputModule", 
                               fileName = cms.untracked.string(options['OUTPUTEDMFILENAME']),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
                               )
process.outpath = cms.EndPath(process.out)
if (not options['DEBUG']):
    process.outpath.remove(process.out)

if (varOptions.isMC):
    process.p = cms.Path(
        process.sampleInfo       +
        process.hltFilter        +
        process.cand_sequence    +
        process.allTagsAndProbes +
        process.pileupReweightingProducer +
        process.mc_sequence      +
        process.eleVarHelper     +
        process.tree_sequence
        )
else:
    process.p = cms.Path(
        process.sampleInfo       +
        process.hltFilter        +
        process.cand_sequence    +
        process.allTagsAndProbes +
        process.mc_sequence      +
        process.eleVarHelper     +
        process.tree_sequence
        )

process.TFileService = cms.Service(
    "TFileService", fileName = cms.string(options['OUTPUT_FILE_NAME']),
    closeFileFast = cms.untracked.bool(True)
    )
