#ifndef MuonIdentification_MuonCaloCompatibilityBase_h
#define MuonIdentification_MuonCaloCompatibilityBase_h

// -*- C++ -*-
//
// Package:    MuonIdentification
// Class:      MuonCaloCompatibilityBase
// 
/*

 Description: Base class for testing if track and calorimeter energy
              are consistent with being a muon

*/
//
// Original Author: Devin Taylor
//
//

#include "DataFormats/MuonReco/interface/Muon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MuonCaloCompatibilityBase {
 public:
   MuonCaloCompatibilityBase()
     : isConfigured_(false) 
     {}
   virtual void configure(const edm::ParameterSet&) = 0;
   virtual double evaluate( const reco::Muon& ) = 0;
   virtual ~MuonCaloCompatibilityBase() {}
 protected:
   bool isConfigured_;
};
#endif
