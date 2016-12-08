import FWCore.ParameterSet.Config as cms

###################################################################
## ID MODULES
###################################################################

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

def setIDs(process, options):

    dataFormat = DataFormat.MiniAOD
    eleProducer = "PatElectronSelectorByValueMap"
    if (options['useAOD']):
        dataFormat = DataFormat.AOD
        eleProducer = "GsfElectronSelectorByValueMap"
        
    switchOnVIDElectronIdProducer(process, dataFormat)

    # define which IDs we want to produce
    my_id_modules = [
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Spring15_25ns_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Summer16_80X_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.heepElectronID_HEEPV60_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring15_25ns_nonTrig_V1_cff',
        'RecoEgamma.ElectronIdentification.Identification.mvaElectronID_Spring16_GeneralPurpose_V1_cff',
        ]

    ### add only miniAOD supported IDs
    if not options['useAOD'] :
        my_id_modules.append( 'RecoEgamma.ElectronIdentification.Identification.cutBasedElectronHLTPreselecition_Summer16_V1_cff' )
                 
    for idmod in my_id_modules:
        setupAllVIDIdsInModule(process, idmod, setupVIDElectronSelection)

    process.egmGsfElectronIDs.physicsObjectSrc     = cms.InputTag(options['ELECTRON_COLL'])
    process.electronMVAValueMapProducer.srcMiniAOD = cms.InputTag(options['ELECTRON_COLL'])

    process.goodElectronsPROBECutBasedVeto = cms.EDProducer(eleProducer,
                                                            input     = cms.InputTag("goodElectrons"),
                                                            cut       = cms.string(options['ELECTRON_CUTS']),
                                                            selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto"),
                                                            id_cut    = cms.bool(True)
                                                            )
    
    ######################################################################################
    ## MODULES FOR N-1 CUT BASED STUDIES
    ######################################################################################
    # List of cuts
    #0 MinPtCut
    #1 GsfEleSCEtaMultiRangeCut
    #2 GsfEleDEtaInCut
    #3 GsfEleDPhiInCut
    #4 GsfEleFull5x5SigmaIEtaIEtaCut
    #5 GsfEleHadronicOverEMCut
    #6 GsfEleDxyCut 
    #7 GsfEleDzCut
    #8 GsfEleEInverseMinusPInverseCut
    #9 GsfEleEffAreaPFIsoCut
    #10 GsfEleConversionVetoCut
    #11 GsfEleMissingHitsCut
    
    #process.goodElectronsPROBECutBasedVeto = cms.EDProducer("PatElectronNm1Selector",
    #                                                    input     = cms.InputTag("goodElectrons"),
    #                                                    cut       = cms.string(options['ELECTRON_CUTS']),
    #                                                    selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto"),
    #                                                    #cutIndicesToMask = cms.vuint32(2, 3),
    #                                                    cutNamesToMask = cms.vstring("GsfEleDEtaInCut_0", "GsfEleDPhiInCut_0")
    #                                                    )
    
    process.goodElectronsPROBEHLTsafe = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBEHLTsafe.selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronHLTPreselection-Summer16-V1")

    process.goodElectronsPROBECutBasedLoose  = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedMedium = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedTight  = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedLoose.selection  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-loose")
    process.goodElectronsPROBECutBasedMedium.selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-medium")
    process.goodElectronsPROBECutBasedTight.selection  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-tight")

    process.goodElectronsPROBECutBasedVeto80X   = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedLoose80X  = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedMedium80X = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedTight80X  = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBEMVA80Xwp90        = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBEMVA80Xwp80        = process.goodElectronsPROBECutBasedVeto.clone()
    process.goodElectronsPROBECutBasedVeto80X.selection   = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-veto"  )
    process.goodElectronsPROBECutBasedLoose80X.selection  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-loose" )
    process.goodElectronsPROBECutBasedMedium80X.selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-medium")
    process.goodElectronsPROBECutBasedTight80X.selection  = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight" )
    process.goodElectronsPROBEMVA80Xwp90.selection        = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp90" )
    process.goodElectronsPROBEMVA80Xwp80.selection        = cms.InputTag("egmGsfElectronIDs:mvaEleID-Spring16-GeneralPurpose-V1-wp80" )
    
    process.goodElectronsTAGCutBasedTight = cms.EDProducer(eleProducer,
                                                          input     = cms.InputTag("goodElectrons"),
                                                          cut       = cms.string(options['ELECTRON_TAG_CUTS']),
                                                          selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Spring15-25ns-V1-standalone-veto"),
                                                          id_cut    = cms.bool(True)
                                                          )
    
    process.goodElectronsTAGCutBasedTight.selection = cms.InputTag("egmGsfElectronIDs:cutBasedElectronID-Summer16-80X-V1-tight")
