import FWCore.ParameterSet.Config as cms

DTTimingExtractorBlock = cms.PSet(
  DTTimingParameters = cms.PSet(
    PruneCut = cms.double(5.),
    DTTimeOffset = cms.double(0.),
    HitError  = cms.double(2.8),
    HitsMin = cms.int32(3),
    UseSegmentT0 = cms.bool(False),
    DoWireCorr = cms.bool(True),
    DropTheta = cms.bool(True),
    RequireBothProjections = cms.bool(False),
    debug = cms.bool(False),
  )
)


