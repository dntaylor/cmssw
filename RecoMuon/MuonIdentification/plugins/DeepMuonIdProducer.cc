// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      DeepDeepMuonIdProducer
//
//
// Original Author:  Devin Taylor
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "RecoMuon/MuonIdentification/interface/MuonCaloCompatibility.h"
#include "RecoMuon/MuonIdentification/interface/DeepMuonCache.h"
#include "RecoMuon/MuonIdentification/interface/DeepMuonCaloCompatibility.h"

#include "DataFormats/Common/interface/ValueMap.h"

class DeepMuonIdProducer : public edm::stream::EDProducer<edm::GlobalCache<DeepMuonCache> > {
public:
  explicit DeepMuonIdProducer(const edm::ParameterSet&, const DeepMuonCache*);

  ~DeepMuonIdProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static std::unique_ptr<DeepMuonCache> initializeGlobalCache(const edm::ParameterSet& cfg);
  static void globalEndJob(const DeepMuonCache* cache) {}

private:
  edm::EDGetTokenT<std::vector<reco::Muon>> muonCollectionToken_;

  MuonCaloCompatibility muonCaloCompatibility_;
  std::map<std::string, DeepMuonCaloCompatibility*> deepMuonCaloCompatibilities_;

  std::vector<std::string> deepCaloMuonLabels_;
  const DeepMuonCache* cache_;

};

DeepMuonIdProducer::DeepMuonIdProducer(const edm::ParameterSet& iConfig, const DeepMuonCache* cache)
    : cache_(cache) {
  auto inputLabel = iConfig.getParameter<edm::InputTag>("src");

  // Load DeepMuonCaloCompatibility
  auto deepCfg = iConfig.getParameter<edm::ParameterSet>("DeepCaloMuonConfiguration");
  auto graphDefinitions = deepCfg.getParameter<std::vector<edm::ParameterSet> >("graphDefinitions");
  for (auto graphDefinition : graphDefinitions) {
    std::string graphName = graphDefinition.getParameter<std::string>("name");
    deepMuonCaloCompatibilities_[graphName] = new DeepMuonCaloCompatibility(cache_);
    deepMuonCaloCompatibilities_[graphName]->configure(graphDefinition);
    deepCaloMuonLabels_.push_back(graphName);
  }

  muonCollectionToken_ = consumes<std::vector<reco::Muon>>(inputLabel);

  // output will be association maps for deep id discriminants
  for (auto name : deepCaloMuonLabels_) {
    produces<edm::ValueMap<float> >(name);
  }
}

DeepMuonIdProducer::~DeepMuonIdProducer() {}

void DeepMuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get muon collection
  edm::Handle<std::vector<reco::Muon>> muons;
  iEvent.getByToken(muonCollectionToken_, muons);

  // create output map to store deep id discriminants
  std::map<std::string, std::vector<float> > deepMuonScores;
  for (auto name : deepCaloMuonLabels_) {
    deepMuonScores[name] = std::vector<float>();
  }

  // loop over muons and calculate the deep id discriminants
  for (const auto& muon : *muons) {
    for (auto pair : deepMuonCaloCompatibilities_) {
      deepMuonScores.at(pair.first).push_back(pair.second->evaluate(muon));
    }
  }

  // store them all in the event as value maps
  for (auto name : deepCaloMuonLabels_) {
    std::unique_ptr<edm::ValueMap<float> > output(new edm::ValueMap<float>());
    edm::ValueMap<float>::Filler output_filler(*output);
    output_filler.insert(muons, deepMuonScores.at(name).begin(), deepMuonScores.at(name).end());
    output_filler.fill();
    iEvent.put(std::move(output),name);
  }
}

std::unique_ptr<DeepMuonCache> DeepMuonIdProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  std::map<std::string, std::string> graphNames;
  if (!cfg.exists("DeepCaloMuonConfiguration")) {
    return std::make_unique<DeepMuonCache>(graphNames, false);
  }
  edm::ParameterSet deepCfg = cfg.getParameter<edm::ParameterSet>("DeepCaloMuonConfiguration");
  auto graphDefinitions = deepCfg.getParameter<std::vector<edm::ParameterSet> >("graphDefinitions");
  bool memmapped = deepCfg.getParameter<bool>("memmapped");
  for (auto graphDefinition : graphDefinitions) {
    std::string graphName = graphDefinition.getParameter<std::string>("name");
    edm::FileInPath graphPath = graphDefinition.getParameter<edm::FileInPath>("path");
    std::string graphFullPath = graphPath.fullPath();
    graphNames[graphName] = graphFullPath;
  }
  return std::make_unique<DeepMuonCache>(graphNames, memmapped);
}

void DeepMuonIdProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setAllowAnything();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(DeepMuonIdProducer);
