import FWCore.ParameterSet.Config as cms

def setupHLT(process,options) :
    process.hltVarHelper = cms.EDProducer("GsfElectronHLTVariableHelper",
                                          probes = cms.InputTag(options['ELECTRON_COLL']),
                                          hltCandidateCollection = cms.InputTag("hltEgammaCandidates"),
                                          mapOutputNames = cms.vstring("hltsieie",
                                                                       "hltecaliso",
                                                                       "hlthcaliso",
                                                                       "hlthoe",
                                                                       "hlttkiso",
                                                                       "hltdeta",
                                                                       "hltdetaseed",
                                                                       "hltdphi",
                                                                       "hlteop",
                                                                       "hltmishits"),
                                          mapInputTags = cms.VInputTag("hltEgammaClusterShape:sigmaIEtaIEta5x5",
                                                                       "hltEgammaEcalPFClusterIso",
                                                                       "hltEgammaHcalPFClusterIso",
                                                                       "hltEgammaHoverE", 
                                                                       "hltEgammaEleGsfTrackIso",
                                                                       "hltEgammaGsfTrackVars:Deta",
                                                                       "hltEgammaGsfTrackVars:DetaSeed",
                                                                       "hltEgammaGsfTrackVars:Dphi",
                                                                       "hltEgammaGsfTrackVars:OneOESuperMinusOneOP",
                                                                       "hltEgammaGsfTrackVars:MissingHits")
                                          )





def setModules(process, options):

    process.sampleInfo = cms.EDProducer("tnp::SampleInfoTree",
                                        #vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                        genInfo = cms.InputTag("generator")
                                        )

    process.eleVarHelper = cms.EDProducer("GsfElectronVariableHelper",
                                          probes = cms.InputTag(options['ELECTRON_COLL']),
                                          vertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                          l1EGColl = cms.InputTag("caloStage2Digis:EGamma")
                                          )

#    setupHLT()

    from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
    process.hltFilter = hltHighLevel.clone()
    process.hltFilter.throw = cms.bool(True)
    process.hltFilter.HLTPaths = options['TnPPATHS']
    process.hltFilter.TriggerResultsTag = cms.InputTag("TriggerResults","",options['HLTProcessName'])

    from PhysicsTools.TagAndProbe.pileupConfiguration_cfi import pileupProducer
    process.pileupReweightingProducer = pileupProducer.clone()
    process.pileupReweightingProducer.pileupInfoTag = "addPileupInfo"

###################################################################                                                                               
## ELECTRON MODULES                                                                                                                                    
###################################################################                                    
    
    process.goodElectrons = cms.EDFilter("GsfElectronRefSelector",
                                         src = cms.InputTag(options['ELECTRON_COLL']),
                                         cut = cms.string(options['ELECTRON_CUTS'])
                                         )


    ### dummy in AOD (use miniAOD for photons)
    process.goodPhotons =  cms.EDFilter("PhotonRefSelector",
                                        src = cms.InputTag( options['PHOTON_COLL'] ),
                                        cut = cms.string(   options['PHOTON_CUTS'] )
                                        )

    process.goodPhotonsProbeHLT = process.goodPhotons.clone()
    
###################################################################                                                                     
## SUPERCLUSTER MODULES                                                     
###################################################################         
    process.superClusterMerger =  cms.EDProducer("EgammaSuperClusterMerger",
                                                 src = cms.VInputTag(cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
                                                                     cms.InputTag("particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower"),
#                                                                     cms.InputTag("particleFlowEGamma"),
                                                                     ),
                                                 )
    
    
    process.superClusterCands = cms.EDProducer("ConcreteEcalCandidateProducer",
                                               src = cms.InputTag("superClusterMerger"),
                                               particleType = cms.int32(11),
                                               )
    
    process.goodSuperClusters = cms.EDFilter("RecoEcalCandidateRefSelector",
                                             src = cms.InputTag("superClusterCands"),
                                             cut = cms.string(options['SUPERCLUSTER_CUTS']),
                                             filter = cms.bool(True)
                                             )
    
    process.GsfMatchedSuperClusterCands = cms.EDProducer("GsfElectronMatchedCandidateProducer",
                                                         src     = cms.InputTag("superClusterCands"),
                                                         ReferenceElectronCollection = cms.untracked.InputTag("goodElectrons"),
                                                         cut = cms.string(options['SUPERCLUSTER_CUTS'])
                                                         )


    process.recoEcalCandidateHelper = cms.EDProducer("RecoEcalCandidateVariableHelper",
                                                     probes = cms.InputTag("superClusterCands"),
                                                     countTracks = cms.bool( False ),
                                                     trkIsoPtMin = cms.double( 0.5 ),
                                                     trkIsoStripEndcap = cms.double( 0.03 ),
                                                     trackProducer = cms.InputTag( "generalTracks" ),
                                                     trkIsoStripBarrel = cms.double( 0.03 ),
                                                     trkIsoConeSize = cms.double( 0.4 ),
                                                     trkIsoVetoConeSize = cms.double( 0.06 ),
                                                     trkIsoRSpan = cms.double( 999999.0 ),
                                                     trkIsoZSpan = cms.double( 999999. )
                                                     )
    
###################################################################
## TRIGGER MATCHING
###################################################################

    print 'Trigger collections: ','TriggerResults::' + options['HLTProcessName'] 
    
    process.goodElectronsTagHLT = cms.EDProducer("GsfElectronTriggerCandProducer",
                                                 filterNames = cms.vstring(options['TnPHLTTagFilters']),
                                                 inputs      = cms.InputTag("goodElectronsTAGCutBasedTight"),
                                                 bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName']),
                                                 objects     = cms.InputTag('hltTriggerSummaryAOD'),
                                                 dR          = cms.double(0.3),
                                                 isAND       = cms.bool(True)
                                                 )
    
    process.goodElectronsProbeHLT = cms.EDProducer("GsfElectronTriggerCandProducer",
                                                   filterNames = cms.vstring(options['TnPHLTProbeFilters']),
                                                   inputs      = cms.InputTag("goodElectrons"),
                                                   bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName']),
                                                   objects     = cms.InputTag('hltTriggerSummaryAOD'),
                                                   dR          = cms.double(0.3),
                                                   isAND       = cms.bool(True)
                                                   )
    
    process.goodElectronsProbeMeasureHLT = cms.EDProducer("GsfElectronTriggerCandProducer",
                                                          filterNames = cms.vstring(options['TnPHLTProbeFilters']),
                                                          inputs      = cms.InputTag("goodElectrons"),
                                                          bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName']),
                                                          objects     = cms.InputTag('hltTriggerSummaryAOD'),
                                                          dR          = cms.double(0.3),
                                                          isAND       = cms.bool(True)
                                                          )
    
    process.goodElectronsMeasureHLT = cms.EDProducer("GsfElectronTriggerCandProducer",
                                                     filterNames = cms.vstring(options['HLTFILTERTOMEASURE']),
                                                     inputs      = cms.InputTag("goodElectronsProbeMeasureHLT"),
                                                     bits        = cms.InputTag('TriggerResults::' + options['HLTProcessName']),
                                                     objects     = cms.InputTag('hltTriggerSummaryAOD'),
                                                     dR          = cms.double(0.3),
                                                     isAND       = cms.bool(False)
                                                     )
    
    process.goodSuperClustersHLT = cms.EDProducer("RecoEcalCandidateTriggerCandProducer",
                                                  filterNames  = cms.vstring(options['TnPHLTProbeFilters']),
                                                  inputs       = cms.InputTag("goodSuperClusters"),
                                                  bits         = cms.InputTag('TriggerResults::' + options['HLTProcessName']),
                                                  objects      = cms.InputTag('hltTriggerSummaryAOD'),
                                                  dR           = cms.double(0.3),
                                                  isAND        = cms.bool(True)
                                                  )
    
###################################################################
## TnP PAIRS
###################################################################
    
    process.tagTightHLT   = cms.EDProducer("CandViewShallowCloneCombiner",
                                         decay = cms.string("goodElectronsTagHLT@+ goodElectronsProbeMeasureHLT@-"), 
                                         checkCharge = cms.bool(True),
                                         cut = cms.string("60<mass<120"),
                                         )
    
    process.tagTightSC    = cms.EDProducer("CandViewShallowCloneCombiner",
                                        decay = cms.string("goodElectronsTagHLT goodSuperClustersHLT"), 
                                        checkCharge = cms.bool(False),
                                        cut = cms.string("60<mass<120"),
                                        )
    
    process.tagTightEleID = cms.EDProducer("CandViewShallowCloneCombiner",
                                          decay = cms.string("goodElectronsTagHLT@+ goodElectronsProbeHLT@-"), 
                                          checkCharge = cms.bool(True),
                                          cut = cms.string("60<mass<120"),
                                          )
    

def setSequences(process, options):
###################################################################
## SEQUENCES
###################################################################
    process.init_sequence = cms.Sequence( )

    process.tag_sequence = cms.Sequence(
        process.goodElectrons                    +
        process.egmGsfElectronIDSequence         +
#        process.goodElectronsTAGCutBasedLoose    +
        process.goodElectronsTAGCutBasedTight    +
        process.goodElectronsTagHLT
        )

    process.ele_sequence = cms.Sequence( )

    process.hlt_sequence = cms.Sequence( )

    process.pho_sequence = cms.Sequence( )

    process.sc_sequence = cms.Sequence(
        process.superClusterMerger      +
        process.superClusterCands       +
        process.recoEcalCandidateHelper +
        process.goodSuperClusters       +
        process.goodSuperClustersHLT    +
        process.GsfMatchedSuperClusterCands
        )

