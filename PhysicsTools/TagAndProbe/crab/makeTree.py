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
    "doPhoID", True,
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

varOptions.parseArguments()


###################################################################
## Define TnP inputs 
###################################################################

options = dict()
options['HLTProcessName']       = "HLT"
options['ELECTRON_COLL']        = "slimmedElectrons"
options['ELECTRON_CUTS']        = "ecalEnergy*sin(superClusterPosition.theta)>5.0 &&  (abs(-log(tan(superClusterPosition.theta/2)))<2.5)"; # && ecalEnergy*sin(superClusterPosition.theta)>10.0) && (abs(superCluster.eta)<2.5)"
options['ELECTRON_TAG_CUTS']    = "(abs(-log(tan(superClusterPosition.theta/2))) <= 2.5) && !(1.4442<=abs(-log(tan(superClusterPosition.theta/2)))<=1.566) && pt >= 30.0"

options['SUPERCLUSTER_COLL']    = "reducedEgamma:reducedSuperClusters"
options['SUPERCLUSTER_CUTS']    = "abs(eta)<2.5 && !(1.4442< abs(eta) <1.566) && et>10.0"

options['PHOTON_COLL']          = "slimmedPhotons"
options['PHOTON_CUTS']          = "(abs(-log(tan(superCluster.position.theta/2)))<=2.5) && pt> 10"


options['MAXEVENTS']            = cms.untracked.int32(varOptions.maxEvents) 
options['useAOD']               = cms.bool(False)
options['DoTrigger']            = cms.bool( varOptions.doTrigger )
options['DoRECO']               = cms.bool( varOptions.doRECO    )
options['DoEleID']              = cms.bool( varOptions.doEleID   )
options['DoPhoID']              = cms.bool( varOptions.doPhoID   )

options['OUTPUTEDMFILENAME']    = 'edmFile.root'
options['DEBUG']                = cms.bool(False)


if (varOptions.isMC):
    options['INPUT_FILE_NAME']  = cms.untracked.vstring( 
        '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/00271303-C80D-E611-A98C-0002C94D552A.root',
        '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/026FA9AF-DB0D-E611-9CFB-A0000420FE80.root',
        '/store/mc/RunIISpring16MiniAODv1/DYToEE_NNPDF30_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_v3-v1/00000/02FEA941-F10D-E611-BB9E-A0000420FE80.root',
        )
    options['OUTPUT_FILE_NAME']    = "TnPTree_mc.root"
    options['TnPPATHS']            = cms.vstring("HLT*")#HLT_Ele27_WPLoose*")
    options['TnPHLTTagFilters']    = cms.vstring()#hltEle23WPLooseGsfTrackIsoFilter")
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("")#hltEle23WPLooseGsfTrackIsoFilter")
    options['GLOBALTAG']           = 'auto:run2_mc'
    options['EVENTSToPROCESS']     = cms.untracked.VEventRange()
else:
    options['INPUT_FILE_NAME']     = cms.untracked.vstring(    '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/150/00000/0A6284C7-D719-E611-93E6-02163E01421D.root',
                                                               '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/158/00000/06277EC1-181A-E611-870F-02163E0145E5.root',
                                                               '/store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/158/00000/0A7BD549-131A-E611-8287-02163E0134FC.root',
                                                               )
                                                           
    options['OUTPUT_FILE_NAME']    = "TnPTree_data.root"
    options['TnPPATHS']            = cms.vstring("HLT_Ele27_eta2p1_WPLoose_Gsf_v*")
    options['TnPHLTTagFilters']    = cms.vstring("hltEle27erWPLooseGsfTrackIsoFilter")
    options['TnPHLTProbeFilters']  = cms.vstring()
    options['HLTFILTERTOMEASURE']  = cms.vstring("hltEle27erWPLooseGsfTrackIsoFilter")
    options['GLOBALTAG']           = 'auto:run2_data'
    options['EVENTSToPROCESS']     = cms.untracked.VEventRange()

###################################################################

from PhysicsTools.TagAndProbe.treeMakerOptions_cfi import *
setModules(process, options)



from PhysicsTools.TagAndProbe.treeContent_cfi import *

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

import PhysicsTools.TagAndProbe.electronIDModules_cfi
PhysicsTools.TagAndProbe.electronIDModules_cfi.setIDs(process, options)

import PhysicsTools.TagAndProbe.photonIDModules_cfi
PhysicsTools.TagAndProbe.photonIDModules_cfi.setIDs(process, options)

###################################################################
## SEQUENCES
###################################################################

process.egmGsfElectronIDs.physicsObjectSrc = cms.InputTag(options['ELECTRON_COLL'])
process.tag_sequence = cms.Sequence(
    process.goodElectrons                    +
    process.egmGsfElectronIDSequence         +
    process.goodElectronsTAGCutBasedLoose    +
    process.goodElectronsTAGCutBasedTight    +
    process.goodElectronsTagHLT              
    )

process.ele_sequence = cms.Sequence(
    process.goodElectronsPROBECutBasedVeto   +
    process.goodElectronsPROBECutBasedLoose  +
    process.goodElectronsPROBECutBasedMedium +
    process.goodElectronsPROBECutBasedTight  +
    process.goodElectronsProbeHLT            
    )

process.hlt_sequence = cms.Sequence(
    process.goodElectronsProbeMeasureHLT     +
    process.goodElectronsMeasureHLT
    )


process.egmPhotonIDs.physicsObjectSrc = cms.InputTag(options['PHOTON_COLL'])
process.pho_sequence = cms.Sequence(
    process.goodPhotons                    +
    process.egmPhotonIDSequence            +
    process.photonIDValueMapProducer       +
    process.goodPhotonsPROBECutBasedLoose  +
    process.goodPhotonsPROBECutBasedMedium +
    process.goodPhotonsPROBECutBasedTight  +
    process.goodPhotonsPROBEMVA            + 
    process.goodPhotonsProbeHLT
    )


process.sc_sequence = cms.Sequence(process.superClusterCands +
                                   process.goodSuperClusters +
                                   process.goodSuperClustersHLT +
                                   process.GsfMatchedSuperClusterCands
                                   )

process.cand_sequence = cms.Sequence( process.tag_sequence )

if (options['DoEleID']):
    process.cand_sequence += process.ele_sequence

if (options['DoPhoID']):
    process.cand_sequence += process.pho_sequence

if (options['DoTrigger']):
    process.cand_sequence += process.hlt_sequence

if (options['DoRECO']):
    process.cand_sequence += process.sc_sequence





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
if (not varOptions.isMC):
    mcTruthCommonStuff = cms.PSet(
        isMC = cms.bool(False)
        )

process.GsfElectronToTrigger = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                              CommonStuffForGsfElectronProbe, mcTruthCommonStuff,
                                              tagProbePairs = cms.InputTag("tagTightHLT"),
                                              arbitration   = cms.string("HighestPt"),
                                              flags         = cms.PSet(passingHLT    = cms.InputTag("goodElectronsMeasureHLT"),
                                                                       passingLoose  = cms.InputTag("goodElectronsPROBECutBasedLoose"),
                                                                       passingMedium = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                       passingTight  = cms.InputTag("goodElectronsPROBECutBasedTight")
                                                                       ),                                               
                                              allProbes     = cms.InputTag("goodElectronsProbeMeasureHLT"),
                                              )

if (varOptions.isMC):
    process.GsfElectronToTrigger.eventWeight    = cms.InputTag("generator")
    process.GsfElectronToTrigger.PUWeightSrc    = cms.InputTag("pileupReweightingProducer","pileupWeights")

process.GsfElectronToSC = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                         CommonStuffForSuperClusterProbe, mcTruthCommonStuff,
                                         tagProbePairs = cms.InputTag("tagTightSC"),
                                         arbitration   = cms.string("HighestPt"),
                                         flags         = cms.PSet(passingRECO   = cms.InputTag("GsfMatchedSuperClusterCands", "superclusters")
                                                                  ),                                               
                                         allProbes     = cms.InputTag("goodSuperClustersHLT"),
                                         )

if (varOptions.isMC):
    process.GsfElectronToSC.eventWeight   = cms.InputTag("generator")
    process.GsfElectronToSC.PUWeightSrc   = cms.InputTag("pileupReweightingProducer","pileupWeights")

process.GsfElectronToEleID = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                            mcTruthCommonStuff, CommonStuffForGsfElectronProbe,
                                            tagProbePairs = cms.InputTag("tagTightEleID"),
                                            arbitration   = cms.string("HighestPt"),
                                            flags         = cms.PSet(passingVeto   = cms.InputTag("goodElectronsPROBECutBasedVeto"),
                                                                     passingLoose  = cms.InputTag("goodElectronsPROBECutBasedLoose"),
                                                                     passingMedium = cms.InputTag("goodElectronsPROBECutBasedMedium"),
                                                                     passingTight  = cms.InputTag("goodElectronsPROBECutBasedTight"),
                                                                     ),                                               
                                            allProbes     = cms.InputTag("goodElectronsProbeHLT"),
                                            )

if (varOptions.isMC):
    process.GsfElectronToEleID.eventWeight   = cms.InputTag("generator")
    process.GsfElectronToEleID.PUWeightSrc   = cms.InputTag("pileupReweightingProducer","pileupWeights")

process.GsfElectronToPhoID = cms.EDAnalyzer("TagProbeFitTreeProducer",
                                            mcTruthCommonStuff, CommonStuffForPhotonProbe,
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
    process.GsfElectronToPhoID.eventWeight   = cms.InputTag("generator")
    process.GsfElectronToPhoID.PUWeightSrc   = cms.InputTag("pileupReweightingProducer","pileupWeights")






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
