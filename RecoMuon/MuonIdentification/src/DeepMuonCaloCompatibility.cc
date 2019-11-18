/*
 * \class DeepCaloMuonProducer
 *
 * CaloMuon identification using DNN.
 *
 * \author Devin Taylor, UC Davis
 */

#include "RecoMuon/MuonIdentification/interface/DeepMuonCaloCompatibility.h"
#include <fstream>
#include <sstream>

// Define the inputs to the DNN
namespace {

  namespace caloMuonInputs_run3_v1 {
    constexpr int NumberOfOutputs = 2;
    namespace InnerTrackBlockInputs {
      enum vars {
        muon_innerTrack_p = 0,
        muon_innerTrack_eta,
        muon_innerTrack_phi,
        muon_innerTrack_qoverp,
        muon_innerTrack_qoverpError,
        muon_innerTrack_validFraction,
        muon_innerTrack_highPurity,
        muon_innerTrack_hitPattern_trackerLayersWithMeasurement,
        muon_innerTrack_hitPattern_pixelLayersWithMeasurement,
        muon_isolationR03_nTracks,
        muon_isolationR03_sumPt,
        muon_calEnergy_em,
        muon_calEnergy_emMax,
        muon_calEnergy_emS25,
        muon_calEnergy_emS9,
        muon_calEnergy_had,
        muon_calEnergy_hadMax,
        muon_calEnergy_hadS9,
        muon_calEnergy_ho,
        muon_calEnergy_hoS9,
        muon_calEnergy_hcal_ieta,
        muon_calEnergy_hcal_iphi,
        NumberOfInputs
      };
    }

  }  // namespace caloMuonInputs_run3_v1

  namespace caloMuonInputs_run3_v2 {
    constexpr int NumberOfOutputs = 2;
    constexpr int NumberOfHcalDigis = 15;
    namespace InnerTrackBlockInputs {
      enum vars {
        muon_innerTrack_p = 0,
        muon_innerTrack_eta,
        muon_innerTrack_phi,
        muon_innerTrack_qoverp,
        muon_innerTrack_qoverpError,
        muon_innerTrack_validFraction,
        muon_innerTrack_highPurity,
        muon_innerTrack_hitPattern_trackerLayersWithMeasurement,
        muon_innerTrack_hitPattern_pixelLayersWithMeasurement,
        muon_calEnergy_em,
        muon_calEnergy_emMax,
        muon_calEnergy_emS25,
        muon_calEnergy_emS9,
        muon_calEnergy_had,
        muon_calEnergy_hadMax,
        muon_calEnergy_hadS9,
        muon_calEnergy_ho,
        muon_calEnergy_hoS9,
        muon_calEnergy_hcal_ieta,
        muon_calEnergy_hcal_iphi,
        NumberOfInputs
      };
    }

    namespace HcalDigiBlockInputs {
      enum vars {
        muon_calEnergy_crossedHadRecHits_ieta = 0,
        muon_calEnergy_crossedHadRecHits_iphi,
        muon_calEnergy_crossedHadRecHits_depth,
        muon_calEnergy_crossedHadRecHits_energy,
        muon_calEnergy_crossedHadRecHits_time,
        muon_calEnergy_crossedHadRecHits_chi2,
        NumberOfInputs
      };
    }

  }  // namespace caloMuonInputs_run3_v2

}  // anonymous namespace

  
void DeepMuonCaloCompatibility::configure(const edm::ParameterSet& iConfig)
{
    name_ = iConfig.getParameter<std::string>("name");
    path_ = iConfig.getParameter<edm::FileInPath>("path");
    meanPath_ = iConfig.getParameter<edm::FileInPath>("means");

    // load the means and sigmas
    std::string fullMeansPath = meanPath_.fullPath();
    std::ifstream inputMeansFile(fullMeansPath);
    std::string line, word, name;
    float mean, sigma;
    while (std::getline(inputMeansFile, line)) {
      std::istringstream ss(line);
      std::getline(ss,word,',');
      name = word;
      std::getline(ss,word,',');
      mean = std::stof(word);
      std::getline(ss,word,',');
      sigma = std::stof(word);
      names_.push_back(name);
      means_.push_back(mean);
      sigmas_.push_back(sigma);
    }
    inputMeansFile.close();

    // TODO create tensor for given name
    innerTrackBlockTensor_ = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape{1, caloMuonInputs_run3_v1::InnerTrackBlockInputs::NumberOfInputs});
    //hcalDigisTensor_ = std::make_unique<tensorflow::Tensor>(
    //      tensorflow::DT_FLOAT,
    //      tensorflow::TensorShape{1, caloMuonInputs_run3_v1::NumberOfHcalDigis, caloMuonInputs_run3_v1::NumberOfHcalDigis, caloMuonInputs_run3_v1::HcalDigiBlockInputs::NumberOfInputs});

    isConfigured_ = true;
}

double DeepMuonCaloCompatibility::evaluate(const reco::Muon& muon) {
    const tensorflow::Tensor pred = getPrediction(muon);
    double muon_compatibility = (double)pred.matrix<float>()(0,1); // TODO: currently only grab second element (must be muon)
    return muon_compatibility;
}


// get the prediction for the selected DNN
tensorflow::Tensor DeepMuonCaloCompatibility::getPrediction(const reco::Muon& muon) {

  std::vector<tensorflow::Tensor> pred_vector;
  tensorflow::Tensor prediction;

  if (name_=="caloMuonRun3v1") {
    getPrediction_run3_v1(muon, pred_vector);
    prediction = tensorflow::Tensor(tensorflow::DT_FLOAT, {static_cast<int>(1), caloMuonInputs_run3_v1::NumberOfOutputs});
    for (int k = 0; k < caloMuonInputs_run3_v1::NumberOfOutputs; ++k) {
      const float pred = pred_vector[0].flat<float>()(k); // just one prediction vector for now
      if (!(pred >= 0 && pred <= 1)) {
        throw cms::Exception("DeepMuonCaloCompatibility")
            << "invalid prediction = " << pred << " for pred_index = " << k;
      } 
      prediction.matrix<float>()(0, k) = pred;
    }
  } // caloMuonsRun3v1
  else {
    prediction = tensorflow::Tensor(tensorflow::DT_FLOAT, {static_cast<int>(1), 2}); // for now only support [pion, muon]
    prediction.matrix<float>()(0,0) = -1.0;
    prediction.matrix<float>()(0,1) = -1.0;
  }
  
  return prediction;
}

// run3_v1
void DeepMuonCaloCompatibility::getPrediction_run3_v1(const reco::Muon& muon, std::vector<tensorflow::Tensor>& pred_vector) {
  createInnerTrackBlockInputs(muon);

  tensorflow::run(&(cache_->getSession(name_)),
                  {
                    {"input_1", *innerTrackBlockTensor_},
                    //{"input_2", *hcalDigiBlockTensor_},
                  },
                  {"ID_pred/Softmax"},
                  &pred_vector);

}

void DeepMuonCaloCompatibility::createInnerTrackBlockInputs(const reco::Muon& muon) {
    tensorflow::Tensor& inputs = *innerTrackBlockTensor_;
    inputs.flat<float>().setZero();

    // TODO, read means/sigmas from file rather than hard code
    namespace dnn = caloMuonInputs_run3_v1::InnerTrackBlockInputs;
    int v;
    v = dnn::muon_innerTrack_p;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->p() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_eta;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->eta() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_phi;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->phi() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_qoverp;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->qoverp() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_qoverpError;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->qoverpError() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_validFraction;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->validFraction() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_highPurity;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->quality(reco::TrackBase::highPurity) : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_hitPattern_trackerLayersWithMeasurement;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_innerTrack_hitPattern_pixelLayersWithMeasurement;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.innerTrack().isNonnull() ? muon.innerTrack()->hitPattern().pixelLayersWithMeasurement() : 0.0, means_[v], sigmas_[v]);
    v = dnn::muon_isolationR03_nTracks;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.isolationR03().nTracks, means_[v], sigmas_[v]);
    v = dnn::muon_isolationR03_sumPt;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.isolationR03().sumPt/muon.pt(), means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_em;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().em, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_emMax;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().emMax, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_emS25;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().emS25, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_emS9;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().emS9, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_had;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().had, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_hadMax;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().hadMax, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_hadS9;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().hadS9, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_ho;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().ho, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_hoS9;
    inputs.matrix<float>()(0, v) = getValueNorm(muon.calEnergy().hoS9, means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_hcal_ieta;
    inputs.matrix<float>()(0, v) = getValueNorm(((HcalDetId)muon.calEnergy().hcal_id).ieta(), means_[v], sigmas_[v]);
    v = dnn::muon_calEnergy_hcal_iphi;
    inputs.matrix<float>()(0, v) = getValueNorm(((HcalDetId)muon.calEnergy().hcal_id).iphi(), means_[v], sigmas_[v]);

}

