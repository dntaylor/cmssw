import FWCore.ParameterSet.Config as cms

CSCTimingExtractorBlock = cms.PSet(
  CSCTimingParameters = cms.PSet(
    PruneCut = cms.double(9.),
    CSCStripTimeOffset = cms.double(0.),
    CSCWireTimeOffset = cms.double(0.),
    CSCStripError = cms.double(7.0),
    CSCWireError = cms.double(8.6),
    # One of these next two lines must be true or no time is created
    UseStripTime = cms.bool(True),
    UseWireTime = cms.bool(True),
    debug = cms.bool(False)
  )
)


