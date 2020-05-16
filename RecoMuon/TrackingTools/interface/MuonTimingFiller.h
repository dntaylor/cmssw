#ifndef TrackingTools_MuonTimingFiller_h
#define TrackingTools_MuonTimingFiller_h 1

// -*- C++ -*-
//
// Package:    MuonTimingFiller
// Class:      MuonTimingFiller
//
/**\class MuonTimingFiller MuonTimingFiller.h RecoMuon/TrackingTools/interface/MuonTimingFiller.h

 Description: Class filling the DT, CSC and Combined MuonTimeExtra objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Piotr Traczyk, CERN
//         Created:  Mon Mar 16 12:27:22 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "RecoMuon/TrackingTools/interface/DTTimingExtractor.h"
#include "RecoMuon/TrackingTools/interface/CSCTimingExtractor.h"

//
// class decleration
//

class MuonServiceProxy;

class MuonTimingFiller {
public:
  MuonTimingFiller(const edm::ParameterSet&,
                   edm::ConsumesCollector& iC,
                   const MuonServiceProxy* service);
  ~MuonTimingFiller();
  void fillTiming(const reco::Muon& muon,
                  reco::MuonTimeExtra& dtTime,
                  reco::MuonTimeExtra& cscTime,
                  reco::MuonTime& rpcTime,
                  reco::MuonTimeExtra& combinedTime,
                  edm::Event& iEvent);
  void fillTiming(const reco::Track& muon,
                  reco::MuonTimeExtra& dtTime,
                  reco::MuonTimeExtra& cscTime,
                  reco::MuonTime& rpcTime,
                  reco::MuonTimeExtra& combinedTime,
                  edm::Event& iEvent);
  void fillTiming(const Trajectory& muon,
                  reco::MuonTimeExtra& dtTime,
                  reco::MuonTimeExtra& cscTime,
                  reco::MuonTime& rpcTime,
                  reco::MuonTimeExtra& combinedTime,
                  edm::Event& iEvent);

private:
  void fillTimeFromMeasurements(const TimeMeasurementSequence& tmSeq, reco::MuonTimeExtra& muTime);
  template <typename T>
  void fillRPCTime(const T& muon, reco::MuonTime& muTime, edm::Event& iEvent);
  void rawFit(
      double& a, double& da, double& b, double& db, const std::vector<double>& hitsx, const std::vector<double>& hitsy);
  void addEcalTime(const reco::Muon& muon, TimeMeasurementSequence& cmbSeq);
  void combineTMSequences(const TimeMeasurementSequence& dtSeq,
                          const TimeMeasurementSequence& cscSeq,
                          TimeMeasurementSequence& cmbSeq);

  const MuonServiceProxy* theService;
  std::unique_ptr<MuonSegmentMatcher> theMatcher_;
  std::unique_ptr<DTTimingExtractor> theDTTimingExtractor_;
  std::unique_ptr<CSCTimingExtractor> theCSCTimingExtractor_;
  double errorEB_, errorEE_, ecalEcut_;
  bool useDT_, useCSC_, useECAL_;
};

#endif
