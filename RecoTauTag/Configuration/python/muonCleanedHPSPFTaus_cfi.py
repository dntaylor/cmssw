import FWCore.ParameterSet.Config as cms
import PhysicsTools.PatAlgos.tools.helpers as configtools

def addMuonCleanedTaus(process):
    from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag

    process.load("RecoTauTag.Configuration.muonCleanedHPSPFTaus_cff")
    patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)
    patAlgosToolsTask.add(process.muonCleanedHPSPFTausTask)

    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

    jetSrc = 'ak4PFJetsMuonCleaned'

    process.ptau = cms.Path( process.PFTau )
    process.PATTauSequence = cms.Sequence(process.PFTau+process.makePatTaus+process.selectedPatTaus)
    process.PATTauSequenceMuonCleaned = cloneProcessingSnippet(process,process.PATTauSequence, 'MuonCleaned', addToTask = True)
    process.recoTauAK4PFJets08RegionMuonCleaned.pfCandSrc = cms.InputTag(jetSrc,'particleFlowMuonCleaned')
    massSearchReplaceAnyInputTag(process.PATTauSequenceMuonCleaned,cms.InputTag('ak4PFJets'),cms.InputTag(jetSrc))  
    process.slimmedTausMuonCleaned = process.slimmedTaus.clone(src = cms.InputTag('selectedPatTausMuonCleaned'))
    patAlgosToolsTask.add(process.slimmedTausMuonCleaned)

    return process
