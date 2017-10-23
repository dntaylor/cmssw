import FWCore.ParameterSet.Config as cms

recoMuonsForJetCleaning = cms.EDFilter('MuonRefSelector',
    src = cms.InputTag('muons'),
    cut = cms.string('pt > 3.0 && isPFMuon && (isGlobalMuon || isTrackerMuon)'),
)

ak4PFJetsMuonCleaned = cms.EDProducer(
    'MuonCleanedJetProducer',
    jetSrc = cms.InputTag("ak4PFJets"),
    muonSrc = cms.InputTag("recoMuonsForJetCleaning"),
    pfCandSrc = cms.InputTag("particleFlow"),
)
