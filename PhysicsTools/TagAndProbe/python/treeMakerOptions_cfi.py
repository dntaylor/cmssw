import FWCore.ParameterSet.Config as cms



def calibrateEGM(process, options ):
    process.selectElectronsBase = cms.EDFilter("PATElectronSelector",
                                               src = cms.InputTag( options['ELECTRON_COLL'] ),
                                               cut = cms.string(options['ELECTRON_CUTS']),
                                               filter = cms.bool(True),
                                               )

    process.selectPhotonsBase   = cms.EDFilter("PATPhotonSelector",
                                               src = cms.InputTag( options['PHOTON_COLL'] ),
                                               cut = cms.string(options['PHOTON_CUTS']),
                                               filter = cms.bool(True),
                                               )


    process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
                                                       calibratedPatElectrons  = cms.PSet( initialSeed = cms.untracked.uint32(81),
                                                                                           engineName = cms.untracked.string('TRandom3'),                
                                                                                           ),
                                                       calibratedPatPhotons    = cms.PSet( initialSeed = cms.untracked.uint32(81),
                                                                                           engineName = cms.untracked.string('TRandom3'),                
                                                                                           ),
                                                       )
    process.load('EgammaAnalysis.ElectronTools.calibratedElectronsRun2_cfi')
    process.load('EgammaAnalysis.ElectronTools.calibratedPhotonsRun2_cfi')


    process.calibratedPatElectrons.electrons = cms.InputTag('selectElectronsBase')
    process.calibratedPatPhotons.photons     = cms.InputTag('selectPhotonsBase')
    if options['isMC']:
        process.calibratedPatElectrons.isMC = cms.bool(True)
        process.calibratedPatPhotons.isMC   = cms.bool(True)

    ### change the input collection to be the calibrated energy one for all other modules from now on
    options['ELECTRON_COLL'] = 'calibratedPatElectrons'
    options['PHOTON_COLL']   = 'calibratedPatPhotons'


def setModules(process, options):
    
    process.sampleInfo = cms.EDProducer("tnp::SampleInfoTree",
                                        #vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                        genInfo = cms.InputTag("generator")
                                        )
    
    process.eleVarHelper = cms.EDProducer("PatElectronVariableHelper",
                                          probes           = cms.InputTag(options['ELECTRON_COLL']),
                                          l1EGColl         = cms.InputTag('caloStage2Digis:EGamma'),
                                          vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices")
                                          )
    
    from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
    process.hltFilter = hltHighLevel.clone()
    process.hltFilter.throw = cms.bool(True)
    process.hltFilter.HLTPaths = options['TnPPATHS']
    process.hltFilter.TriggerResultsTag = cms.InputTag("TriggerResults","",options['HLTProcessName'])    

    from PhysicsTools.TagAndProbe.pileupConfiguration_cfi import pileupProducer
    process.pileupReweightingProducer = pileupProducer.clone()
    

    if options['UseCalibEn']:
        calibrateEGM( process, options )

###################################################################                                                                               
## Electron/Photon MODULES                                                                                                                                    
###################################################################                                        
    process.goodElectrons = cms.EDFilter("PATElectronRefSelector",
                                         src = cms.InputTag( options['ELECTRON_COLL'] ),
                                         cut = cms.string(   options['ELECTRON_CUTS'] )
                                         )
    process.goodPhotons =  cms.EDFilter("PATPhotonRefSelector",
                                        src = cms.InputTag( options['PHOTON_COLL'] ),
                                        cut = cms.string(   options['PHOTON_CUTS'] )
                                        )

    
###################################################################                                                                     
## SUPERCLUSTER MODULES                                                     
###################################################################         
    
    process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                               src = cms.InputTag(options['SUPERCLUSTER_COLL']),
                                               particleType = cms.int32(11),
                                               )
    
    process.goodSuperClusters = cms.EDFilter("RecoEcalCandidateRefSelector",
                                             src = cms.InputTag("superClusterCands"),
                                             cut = cms.string(options['SUPERCLUSTER_CUTS']),
                                             filter = cms.bool(True)
                                             )
    
    process.GsfMatchedSuperClusterCands = cms.EDProducer("PatElectronMatchedCandidateProducer",
                                                         src     = cms.InputTag("superClusterCands"),
                                                         ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
                                                         cut = cms.string(options['SUPERCLUSTER_CUTS'])
                                                         )
    
###################################################################
## TRIGGER MATCHING
###################################################################
    
    process.goodElectronsTagHLT = cms.EDProducer("PatElectronTriggerCandProducer",
                                                 filterNames = cms.vstring(options['TnPHLTTagFilters']),
                                                 inputs      = cms.InputTag("goodElectronsTAGCutBasedTight"),
                                                 bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName'] ),
                                                 objects     = cms.InputTag('selectedPatTrigger'),
                                                 dR          = cms.double(0.3),
                                                 isAND       = cms.bool(True)
                                                 )
    
    process.goodElectronsProbeHLT = cms.EDProducer("PatElectronTriggerCandProducer",
                                                   filterNames = cms.vstring(options['TnPHLTProbeFilters']),
                                                   inputs      = cms.InputTag("goodElectrons"),
                                                   bits        = cms.InputTag('TriggerResults::'+ options['HLTProcessName']),
                                                   objects     = cms.InputTag('selectedPatTrigger'),
                                                   dR          = cms.double(0.3),
                                                   isAND       = cms.bool(True)
                                                   )
    
    process.goodElectronsProbeMeasureHLT = cms.EDProducer("PatElectronTriggerCandProducer",
                                                          filterNames = cms.vstring(options['TnPHLTProbeFilters']),
                                                          inputs      = cms.InputTag("goodElectrons"),
                                                          bits        = cms.InputTag('TriggerResults::'+ options['HLTProcessName']),
                                                          objects     = cms.InputTag('selectedPatTrigger'),
                                                          dR          = cms.double(0.3),
                                                          isAND       = cms.bool(True)
                                                          )
    
    process.goodElectronsMeasureHLT = cms.EDProducer("PatElectronTriggerCandProducer",
                                                     filterNames = cms.vstring(options['HLTFILTERTOMEASURE']),
                                                     inputs      = cms.InputTag("goodElectronsProbeMeasureHLT"),
                                                     bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName'] ),
                                                     objects     = cms.InputTag('selectedPatTrigger'),
                                                     dR          = cms.double(0.3),
                                                     isAND       = cms.bool(False)
                                                     )
    
    process.goodPhotonsProbeHLT = cms.EDProducer("PatPhotonTriggerCandProducer",
                                                 filterNames = options['TnPHLTProbeFilters'],
                                                 inputs      = cms.InputTag("goodPhotons"),
                                                 bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName'] ),
                                                 objects     = cms.InputTag('selectedPatTrigger'),
                                                 dR          = cms.double(0.3),
                                                 isAND       = cms.bool(True)
                                                 )


    process.goodSuperClustersHLT = cms.EDProducer("RecoEcalCandidateTriggerCandProducer",
                                                  filterNames  = cms.vstring(options['TnPHLTProbeFilters']),
                                                  inputs       = cms.InputTag("goodSuperClusters"),
                                                  bits         = cms.InputTag('TriggerResults::' + options['HLTProcessName'] ),
                                                  objects      = cms.InputTag('selectedPatTrigger'),
                                                  dR           = cms.double(0.3),
                                                  isAND        = cms.bool(True)
                                                  )


###################################################################
## TnP PAIRS
###################################################################
    masscut = cms.string("60<mass<120")
    if (options['isMC']) :
        ## keep larger mass window for MC for template fitting
        masscut = cms.string("40<mass<140")
    process.tagTightHLT   = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay = cms.string("goodElectronsTagHLT@+ goodElectronsProbeMeasureHLT@-"), 
                                           checkCharge = cms.bool(True),
                                           cut = masscut,
                                           )
    
    process.tagTightSC    = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay = cms.string("goodElectronsTagHLT goodSuperClustersHLT"), 
                                           checkCharge = cms.bool(False),
                                           cut = masscut,
                                           )
    
    process.tagTightEleID = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay = cms.string("goodElectronsTagHLT goodElectronsProbeHLT"), 
                                           checkCharge = cms.bool(False),
                                           cut = masscut,
                                           )
    
    process.tagTightPhoID = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay = cms.string("goodElectronsTagHLT goodPhotonsProbeHLT"), 
                                           checkCharge = cms.bool(False),
                                           cut = masscut,
                                           )
    
def setSequences(process, options):
###################################################################
## SEQUENCES
###################################################################
    process.init_sequence = cms.Sequence()
    if options['UseCalibEn']:
        process.enCalib_sequence = cms.Sequence(
            process.selectElectronsBase    +
            process.selectPhotonsBase      +
            process.calibratedPatElectrons +
            process.calibratedPatPhotons   
            )
        process.init_sequence += process.enCalib_sequence

    process.tag_sequence = cms.Sequence(
        process.goodElectrons                    +
        process.egmGsfElectronIDSequence         +
#        process.goodElectronsTAGCutBasedVeto     +
        process.goodElectronsTAGCutBasedTight    +
        process.goodElectronsTagHLT
        )

    process.ele_sequence = cms.Sequence(
        process.goodElectronsPROBEHLTsafe           +
        process.goodElectronsPROBECutBasedVeto      +
        process.goodElectronsPROBECutBasedLoose     +
        process.goodElectronsPROBECutBasedMedium    +
        process.goodElectronsPROBECutBasedTight     +
        process.goodElectronsPROBECutBasedVeto80X   +
        process.goodElectronsPROBECutBasedLoose80X  +
        process.goodElectronsPROBECutBasedMedium80X +
        process.goodElectronsPROBECutBasedTight80X  +
        process.goodElectronsPROBEMVA80Xwp90        +
        process.goodElectronsPROBEMVA80Xwp80        +
        process.goodElectronsProbeHLT
        )

    process.hlt_sequence = cms.Sequence(
        process.goodElectronsProbeMeasureHLT     +
        process.goodElectronsMeasureHLT
        )

    process.pho_sequence = cms.Sequence(
        process.goodPhotons                       +
        process.egmPhotonIDSequence               +
        process.goodPhotonsPROBECutBasedLoose     +
        process.goodPhotonsPROBECutBasedMedium    +
        process.goodPhotonsPROBECutBasedTight     +
        process.goodPhotonsPROBEMVA               +
        process.goodPhotonsPROBECutBasedLoose80X  +
        process.goodPhotonsPROBECutBasedMedium80X +
        process.goodPhotonsPROBECutBasedTight80X  +
        process.goodPhotonsPROBEMVA80Xwp90       +
        process.goodPhotonsPROBEMVA80Xwp80       +
        process.goodPhotonsProbeHLT
        )

    process.sc_sequence = cms.Sequence( 
        process.superClusterCands +
        process.goodSuperClusters +
        process.goodSuperClustersHLT +
        process.GsfMatchedSuperClusterCands
        )
