import FWCore.ParameterSet.Config as cms

##########################################################################
## TREE CONTENT
#########################################################################
    
ZVariablesToStore = cms.PSet(
    eta = cms.string("eta"),
    abseta = cms.string("abs(eta)"),
    pt  = cms.string("pt"),
    mass  = cms.string("mass"),
    )   

SCProbeVariablesToStore = cms.PSet(
    probe_sc_eta    = cms.string("eta"),
    probe_sc_abseta = cms.string("abs(eta)"),
    probe_sc_pt     = cms.string("pt"),
    probe_sc_et     = cms.string("et"),
    probe_sc_e      = cms.string("energy"),
    )

EleProbeVariablesToStore = cms.PSet(
    probe_Ele_eta    = cms.string("eta"),
    probe_Ele_abseta = cms.string("abs(eta)"),
    probe_Ele_pt     = cms.string("pt"),
    probe_Ele_et     = cms.string("et"),
    probe_Ele_e      = cms.string("energy"),
    probe_Ele_q      = cms.string("charge"),
    
    ## super cluster quantities
    probe_sc_energy   = cms.string("superCluster.energy"),
    probe_sc_et       = cms.string("superCluster.energy*sin(superClusterPosition.theta)"),    
    probe_sc_eta      = cms.string("-log(tan(superCluster.position.theta/2))"),
    probe_sc_abseta   = cms.string("abs(-log(tan(superCluster.position.theta/2)))"),
    
    #id based
    probe_Ele_dEtaOut       = cms.string("deltaEtaSeedClusterTrackAtCalo"),
    probe_Ele_dEtaIn        = cms.string("deltaEtaSuperClusterTrackAtVtx"),
    probe_Ele_dPhiIn        = cms.string("deltaPhiSuperClusterTrackAtVtx"),
    probe_Ele_sigmaIEtaIEta = cms.string("sigmaIetaIeta"),
    probe_Ele_r9            = cms.string("r9"),
    probe_Ele_r9_5x5        = cms.string("full5x5_r9"),
    probe_Ele_sieie_5x5     = cms.string("full5x5_sigmaIetaIeta"),
    probe_Ele_hoe           = cms.string("hadronicOverEm"),
    probe_Ele_ooemoop       = cms.string("(1.0/ecalEnergy - eSuperClusterOverP/ecalEnergy)"),

    probe_Ele_mHits         = cms.InputTag("eleVarHelper:missinghits"),
    probe_Ele_dz            = cms.InputTag("eleVarHelper:dz"),
    probe_Ele_dxy           = cms.InputTag("eleVarHelper:dxy"),
    probe_Ele_nonTrigMVA    = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15NonTrig25nsV1Values"),
    probe_Ele_trigMVA       = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15Trig25nsV1Values"),
     
    #isolation
    probe_Ele_chIso         = cms.string("pfIsolationVariables().sumChargedHadronPt"),
    probe_Ele_phoIso        = cms.string("pfIsolationVariables().sumPhotonEt"),
    probe_Ele_neuIso        = cms.string("pfIsolationVariables().sumNeutralHadronEt"),

    # tracker
    probe_Ele_trkPt         = cms.string("gsfTrack().ptMode") 

    )

PhoProbeVariablesToStore = cms.PSet(
    probe_Pho_eta    = cms.string("eta"),
    probe_Pho_abseta = cms.string("abs(eta)"),
    probe_Pho_et     = cms.string("et"),
    probe_Pho_e      = cms.string("energy"),

## super cluster quantities
    probe_sc_energy = cms.string("superCluster.energy"),
    probe_sc_et     = cms.string("superCluster.energy*sin(superCluster.position.theta)"),    
    probe_sc_eta    = cms.string("-log(tan(superCluster.position.theta/2))"),
    probe_sc_abseta = cms.string("abs(-log(tan(superCluster.position.theta/2)))"),


#id based
    probe_Pho_full5x5x_r9   = cms.string("full5x5_r9"),
    probe_Pho_r9            = cms.string("r9"),
    probe_Pho_sieie         = cms.string("full5x5_sigmaIetaIeta"),
    probe_Pho_sieip          = cms.InputTag("photonIDValueMapProducer:phoFull5x5SigmaIEtaIPhi"),
    probe_Pho_ESsigma       = cms.InputTag("photonIDValueMapProducer:phoESEffSigmaRR"),
    probe_Pho_hoe           = cms.string("hadronicOverEm"),

#iso
    probe_Pho_chIso    = cms.InputTag("photonIDValueMapProducer:phoChargedIsolation"),
    probe_Pho_neuIso   = cms.InputTag("photonIDValueMapProducer:phoNeutralHadronIsolation"),
    probe_Pho_phoIso   = cms.InputTag("photonIDValueMapProducer:phoPhotonIsolation"),
    probe_Pho_chWorIso = cms.InputTag("photonIDValueMapProducer:phoWorstChargedIsolation"), 

#pho mva
    probe_mva          = cms.InputTag("photonMVAValueMapProducer:PhotonMVAEstimatorRun2Spring15NonTrig25nsV2p1Values"),
)




TagVariablesToStore = cms.PSet(
    Ele_eta    = cms.string("eta"),
    Ele_phi    = cms.string("phi"),
    Ele_abseta = cms.string("abs(eta)"),
    Ele_pt     = cms.string("pt"),
    Ele_et     = cms.string("et"),
    Ele_e      = cms.string("energy"),
    Ele_q      = cms.string("charge"),
    
    ## super cluster quantities
    sc_energy = cms.string("superCluster.energy"),
    sc_et     = cms.string("superCluster.energy*sin(superClusterPosition.theta)"),    
    sc_eta    = cms.string("-log(tan(superClusterPosition.theta/2))"),
    sc_abseta = cms.string("abs(-log(tan(superCluster.position.theta/2)))"),
 #   
    Ele_mHits         = cms.InputTag("eleVarHelper:missinghits"),
    Ele_dz            = cms.InputTag("eleVarHelper:dz"),
    Ele_dxy           = cms.InputTag("eleVarHelper:dxy"),
    Ele_nonTrigMVA    = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15NonTrig25nsV1Values"),
    Ele_trigMVA       = cms.InputTag("electronMVAValueMapProducer:ElectronMVAEstimatorRun2Spring15Trig25nsV1Values"),

    )

CommonStuffForGsfElectronProbe = cms.PSet(
    addEventVariablesInfo   =  cms.bool(True),

    variables        = cms.PSet(EleProbeVariablesToStore),
    pairVariables    =  cms.PSet(ZVariablesToStore),
    tagVariables   =  cms.PSet(TagVariablesToStore),

    ignoreExceptions =  cms.bool (True),
    addRunLumiInfo   =  cms.bool (True),
    pileupInfoTag    = cms.InputTag("slimmedAddPileupInfo"),
    vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices"),
    beamSpot         = cms.InputTag("offlineBeamSpot"),
    pfMet            = cms.InputTag("slimmedMETsPuppi"),
    rho              = cms.InputTag("fixedGridRhoFastjetAll"),
    #    pfMet = cms.InputTag("slimmedMETsNoHF"),
    #    rho = cms.InputTag("fixedGridRhoAll"),

    pairFlags     =  cms.PSet(
        mass60to120 = cms.string("60 < mass < 120")
        ),
    tagFlags       =  cms.PSet(),    

    )

CommonStuffForPhotonProbe = CommonStuffForGsfElectronProbe.clone()
CommonStuffForPhotonProbe.variables = cms.PSet(PhoProbeVariablesToStore)

CommonStuffForSuperClusterProbe = CommonStuffForGsfElectronProbe.clone()
CommonStuffForSuperClusterProbe.variables = cms.PSet(SCProbeVariablesToStore)

mcTruthCommonStuff = cms.PSet(
    isMC = cms.bool(True),
    #tagMatches = cms.InputTag("McMatchTag"),
    motherPdgId = cms.vint32(),
    #motherPdgId = cms.vint32(22,23),
    #motherPdgId = cms.vint32(443), # JPsi
    #motherPdgId = cms.vint32(553), # Yupsilon
    #makeMCUnbiasTree = cms.bool(False),
    #checkMotherInUnbiasEff = cms.bool(False),
    genParticles = cms.InputTag("prunedGenParticles"),
    useTauDecays = cms.bool(False),
    checkCharge = cms.bool(False),
    pdgId = cms.int32(11),
    mcVariables = cms.PSet(
        probe_eta = cms.string("eta"),
        probe_abseta = cms.string("abs(eta)"),
        probe_et  = cms.string("et"),
        probe_e  = cms.string("energy"),
        ),
    mcFlags     =  cms.PSet(
        probe_flag = cms.string("pt>0")
        ),      
    )

