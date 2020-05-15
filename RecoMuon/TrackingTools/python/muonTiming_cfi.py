import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonTimingFiller_cfi import *

muontiming = cms.EDProducer('MuonTimingProducer',
  TimingFillerBlock,
  MuonCollection = cms.InputTag("muons"),
)
