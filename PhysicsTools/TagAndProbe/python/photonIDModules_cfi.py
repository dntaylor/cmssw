import FWCore.ParameterSet.Config as cms

###################################################################
## ID MODULES
###################################################################

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

def setIDs(process, options):

    dataFormat = DataFormat.MiniAOD
    if (options['useAOD']):
        dataFormat = DataFormat.AOD
        
    switchOnVIDPhotonIdProducer(process, dataFormat)
        
    # define which IDs we want to produce
    my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff',
                     'RecoEgamma.PhotonIdentification.Identification.mvaPhotonID_Spring15_25ns_nonTrig_V2p1_cff',
                     'RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring16_V1_cff']
                 
    for idmod in my_id_modules:
        setupAllVIDIdsInModule(process, idmod, setupVIDPhotonSelection)

    process.egmPhotonIDs.physicsObjectSrc        = cms.InputTag(options['PHOTON_COLL'])
    process.photonIDValueMapProducer.srcMiniAOD  = cms.InputTag(options['PHOTON_COLL'])
    process.photonMVAValueMapProducer.srcMiniAOD = cms.InputTag(options['PHOTON_COLL'])
    process.photonMVAValueMapProducer.src        = cms.InputTag(options['PHOTON_COLL'])


    process.goodPhotonsPROBECutBasedLoose = cms.EDProducer("PatPhotonSelectorByValueMap",
                                                           input     = cms.InputTag("goodPhotons"),
                                                           cut       = cms.string(options['PHOTON_CUTS']),
                                                           selection = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring15-25ns-V1-standalone-loose"),
                                                           id_cut    = cms.bool(True)
                                                           )

    process.goodPhotonsPROBECutBasedMedium = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBECutBasedMedium.selection = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring15-25ns-V1-standalone-medium")
    process.goodPhotonsPROBECutBasedTight = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBECutBasedTight.selection = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring15-25ns-V1-standalone-tight")

    process.goodPhotonsPROBECutBasedLoose80X  = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBECutBasedMedium80X = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBECutBasedTight80X  = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBECutBasedLoose80X.selection  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring16-V1-loose"  )
    process.goodPhotonsPROBECutBasedMedium80X.selection = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring16-V1-medium" )
    process.goodPhotonsPROBECutBasedTight80X.selection  = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring16-V1-tight"  )


    process.goodPhotonsPROBEMVA = process.goodPhotonsPROBECutBasedLoose.clone()
    process.goodPhotonsPROBEMVA.selection = cms.InputTag("egmPhotonIDs:mvaPhoID-Spring15-25ns-nonTrig-V2p1-wp90")    

