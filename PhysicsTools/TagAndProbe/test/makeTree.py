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
    "isAOD", False,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "switch to run other AOD (for RECO SFs)"
    )


varOptions.parseArguments()


###################################################################
## Define TnP inputs 
###################################################################

options = dict()
options['useAOD']               = cms.bool(varOptions.isAOD)

options['HLTProcessName']       = "HLT"

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


if (varOptions.isMC):
    options['OUTPUT_FILE_NAME']    = "TnPTree_mc.root"
    options['TnPPATHS']            = cms.vstring("HLT*")
    options['TnPHLTTagFilters']    = cms.vstring()
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("")
    options['GLOBALTAG']           = 'auto:run2_mc'
    options['EVENTSToPROCESS']     = cms.untracked.VEventRange()
else:
    options['OUTPUT_FILE_NAME']    = "TnPTree_data.root"
    options['TnPPATHS']            = cms.vstring("HLT_Ele27_eta2p1_WPLoose_Gsf_v*")
    options['TnPHLTTagFilters']    = cms.vstring("hltEle27erWPLooseGsfTrackIsoFilter")
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("hltEle27erWPLooseGsfTrackIsoFilter")
    options['GLOBALTAG']           = 'auto:run2_data'
    options['EVENTSToPROCESS']     = cms.untracked.VEventRange()

###################################################################
## Inputs for test
###################################################################
filesMC =  cms.untracked.vstring(
    '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/00271303-C80D-E611-A98C-0002C94D552A.root',
    '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/026FA9AF-DB0D-E611-9CFB-A0000420FE80.root',
    '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/02FEA941-F10D-E611-BB9E-A0000420FE80.root',
    )

filesData =  cms.untracked.vstring( 
    '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/150/00000/0A6284C7-D719-E611-93E6-02163E01421D.root',
    '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/158/00000/06277EC1-181A-E611-870F-02163E0145E5.root',
    '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/158/00000/0A7BD549-131A-E611-8287-02163E0134FC.root',
    )


if options['useAOD']:
    filesMC = cms.untracked.vstring(
        '/store/mc/RunIISpring16DR80/DYToEE_NNPDF30_13TeV-powheg-pythia8/AODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/000AB373-AD0C-E611-8261-0002C94CD120.root',
        '/store/mc/RunIISpring16DR80/DYToEE_NNPDF30_13TeV-powheg-pythia8/AODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/0020B9BE-D20C-E611-AD17-A0369F310374.root',
        '/store/mc/RunIISpring16DR80/DYToEE_NNPDF30_13TeV-powheg-pythia8/AODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/004A2D66-D30C-E611-A844-0CC47A009258.root',
        )
    filesData = cms.untracked.vstring(
       '/store/data/Run2016B/SingleElectron/AOD/PromptReco-v2/000/273/158/00000/100B4FC6-1D1A-E611-AADB-02163E0118D5.root',
       '/store/data/Run2016B/SingleElectron/AOD/PromptReco-v2/000/273/158/00000/1032C8FE-211A-E611-BE71-02163E01387F.root',
       '/store/data/Run2016B/SingleElectron/AOD/PromptReco-v2/000/273/158/00000/12E3D1D6-0A1A-E611-85A9-02163E011BF0.root',
       '/store/data/Run2016B/SingleElectron/AOD/PromptReco-v2/000/273/158/00000/18C2A248-101A-E611-9906-02163E01414F.root',
       '/store/data/Run2016B/SingleElectron/AOD/PromptReco-v2/000/273/158/00000/18CB30C7-181A-E611-BA7E-02163E014573.root',
        )

options['INPUT_FILE_NAME'] = filesData
if varOptions.isMC:
    options['INPUT_FILE_NAME'] = filesMC

###################################################################
## import TnP tree maker pythons and configure for AODs
###################################################################
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
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options['GLOBALTAG'] , '')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
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
import PhysicsTools.TagAndProbe.electronIDModules_cfi as egmEleID
import PhysicsTools.TagAndProbe.photonIDModules_cfi   as egmPhoID
egmEleID.setIDs(process, options)
egmPhoID.setIDs(process, options)
process.egmGsfElectronIDs.physicsObjectSrc = cms.InputTag(options['ELECTRON_COLL'])
process.egmPhotonIDs.physicsObjectSrc      = cms.InputTag(options['PHOTON_COLL'])


###################################################################
## SEQUENCES
###################################################################
tnpTreeMaker.setSequences(process,options)
process.cand_sequence = cms.Sequence( process.tag_sequence )

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
                                              flags         = cms.PSet(passingHLT    = cms.InputTag("goodElectronsMeasureHLT"),
                                                                       passingLoose  = cms.InputTag("goodElectronsPROBECutBasedLoose"),
                                                                       passingMedium = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                       passingTight  = cms.InputTag("goodElectronsPROBECutBasedTight")
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
                                            flags         = cms.PSet(passingVeto   = cms.InputTag("goodElectronsPROBECutBasedVeto"),
                                                                     passingLoose  = cms.InputTag("goodElectronsPROBECutBasedLoose"),
                                                                     passingMedium = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                     passingTight  = cms.InputTag("goodElectronsPROBECutBasedTight"),
                                                                     ),                                               
                                            allProbes     = cms.InputTag("goodElectronsProbeHLT"),
                                            )

process.GsfElectronToPhoID = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                            tnpVars.mcTruthCommonStuff, tnpVars.CommonStuffForPhotonProbe,
                                            tagProbePairs = cms.InputTag("tagTightPhoID"),
                                            arbitration   = cms.string("HighestPt"),
                                            flags         = cms.PSet(passingLoose  = cms.InputTag("goodPhotonsPROBECutBasedLoose"),
                                                                     passingMedium = cms.InputTag("goodPhotonsPROBECutBasedMedium"),
                                                                     passingTight  = cms.InputTag("goodPhotonsPROBECutBasedTight"),
                                                                     passingMVA    = cms.InputTag("goodPhotonsPROBEMVA"),
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
        process.sampleInfo    +
        process.hltFilter     +
        process.cand_sequence + 
        process.allTagsAndProbes +
        process.pileupReweightingProducer +
        process.mc_sequence +
        process.eleVarHelper +
        process.tree_sequence
        )
else:
    process.p = cms.Path(
        process.sampleInfo    +
        process.hltFilter     +
        process.cand_sequence +
        process.allTagsAndProbes +
        process.mc_sequence  +
        process.eleVarHelper +
        process.tree_sequence
        )

process.TFileService = cms.Service(
    "TFileService", fileName = cms.string(options['OUTPUT_FILE_NAME']),
    closeFileFast = cms.untracked.bool(True)
    )
