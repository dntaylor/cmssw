#ifndef TopQuarkAnalysis_TopEventProducers_PseudoTopProducer_H
#define TopQuarkAnalysis_TopEventProducers_PseudoTopProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "fastjet/JetDefinition.hh"
#include <set>

class PseudoTopProducer : public edm::stream::EDProducer<>
{
public:
  PseudoTopProducer(const edm::ParameterSet& pset);
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

private:
  bool isHadron(const int pdgId) const;
  bool isFromHadron(const reco::Candidate* p) const;
  bool isBHadron(const reco::Candidate* p) const;
  bool isBHadron(const unsigned int pdgId) const;
  void insertAllDaughters(const reco::Candidate* p, std::set<const reco::Candidate*>& list) const;

  const reco::Candidate* getLast(const reco::Candidate* p);
  reco::GenParticleRef buildGenParticle(const reco::Candidate* p, reco::GenParticleRefProd& refHandle,
                                        std::auto_ptr<reco::GenParticleCollection>& outColl) const;
  typedef reco::Particle::LorentzVector LorentzVector;

private:
  edm::EDGetTokenT<edm::View<reco::Candidate> > finalStateToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > genParticleToken_;
  const double leptonMinPt_, leptonMaxEta_, jetMinPt_, jetMaxEta_;
  const double wMass_, tMass_;

  typedef fastjet::JetDefinition JetDef;
  std::shared_ptr<JetDef> fjLepDef_, fjJetDef_;
  reco::Particle::Point genVertex_;

};

#endif
