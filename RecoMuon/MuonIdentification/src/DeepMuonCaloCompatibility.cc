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

  namespace empty {
    constexpr int NumberOfOutputs = 1; // Defines the number of elements in the output vector
    constexpr int MuonOutput = 0;      // Defines which element of the output vector corresponds to a muon
  }

  namespace caloMuonInputs_run3_v1 {
    constexpr int NumberOfOutputs = 2;
    constexpr int MuonOutput = 1;
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
    constexpr int MuonOutput = 1;
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

    if (isConfigured_) return;

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

    // define graphs
    if (name_=="caloMuonRun3v1") {
      inputNames_.push_back("input_1");
      inputShapes_.push_back(tensorflow::TensorShape{1, caloMuonInputs_run3_v1::InnerTrackBlockInputs::NumberOfInputs});
      kInnerTrack_ = 0;
      kMuonPosition_ = caloMuonInputs_run3_v1::MuonOutput;
      outputName_ = "ID_pred/Softmax";
    }
    else if (name_=="caloMuonRun3v2") {
      inputNames_.push_back("input_1");
      inputShapes_.push_back(tensorflow::TensorShape{1, caloMuonInputs_run3_v1::InnerTrackBlockInputs::NumberOfInputs});
      kInnerTrack_ = 0;
      inputNames_.push_back("input_2");
      inputShapes_.push_back(tensorflow::TensorShape{1, caloMuonInputs_run3_v2::HcalDigiBlockInputs::NumberOfInputs, caloMuonInputs_run3_v2::NumberOfHcalDigis}); 
      kHcalDigi_ = 1;
      kMuonPosition_ = caloMuonInputs_run3_v2::MuonOutput;
      outputName_ = "ID_pred/Softmax";
    }
    else {
      kMuonPosition_ = empty::MuonOutput;
    }

    inputTensors_.resize(inputShapes_.size());
    for (size_t i=0; i<inputShapes_.size(); i++) {
      inputTensors_[i] = tensorflow::NamedTensor(inputNames_[i], tensorflow::Tensor(tensorflow::DT_FLOAT, inputShapes_.at(i)));
    }

    // now validate the graph
    auto graph = cache_->getGraph(name_);
    for (size_t i=0; i<inputShapes_.size(); i++){
      const auto& name = graph.node(i).name();
      const auto& shape = graph.node(i).attr().at("shape").shape();
      // not necessary to be in same order in the input graph
      auto it = std::find(inputNames_.begin(), inputNames_.end(), name);
      if (it==inputNames_.end()) {
        throw cms::Exception("DeepCaloMuonCompatibility")
          << "Unknown input name " << name;
      }
      int j = std::distance(inputNames_.begin(),it);
      for (int d=1; d<inputShapes_.at(j).dims(); d++) { // skip first dim since it should be -1 and not 1 like we define here for evaluation
        if (shape.dim(d).size() != inputShapes_.at(j).dim_size(d)) {
          throw cms::Exception("DeepMuonCaloCompatibility")
            << "Number of inputs in graph does not match those expected for " << name_ << ".\n"
            << "Expected input " << j << " dim " << d << " = " << inputShapes_.at(j).dim_size(d) << "."
            << " Found " << shape.dim(d).size() << ".";
        }
      }
    }
    const auto& outName = graph.node(graph.node_size() - 1).name();
    if (outName!=outputName_) {
      throw cms::Exception("DeepCaloMuonCompatibility")
        << "Unexpected output name. Expected " << outputName_ << " found " << name << ".";
    }

    isConfigured_ = true;
}

double DeepMuonCaloCompatibility::evaluate(const reco::Muon& muon) {
    const tensorflow::Tensor pred = getPrediction(muon);
    double muon_compatibility = (double)pred.matrix<float>()(0,kMuonPosition_);
    return muon_compatibility;
}


// get the prediction for the selected DNN
tensorflow::Tensor DeepMuonCaloCompatibility::getPrediction(const reco::Muon& muon) {

  std::vector<tensorflow::Tensor> pred_vector;
  tensorflow::Tensor prediction;

  if (name_=="caloMuonRun3v1") {
    getPrediction_run3_v1(muon, pred_vector);
    prediction = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, caloMuonInputs_run3_v1::NumberOfOutputs});
    for (int k = 0; k < caloMuonInputs_run3_v1::NumberOfOutputs; ++k) {
      const float pred = pred_vector[0].flat<float>()(k); // just one prediction vector for now
      if (!(pred >= 0 && pred <= 1)) {
        throw cms::Exception("DeepMuonCaloCompatibility")
            << "invalid prediction = " << pred << " for pred_index = " << k;
      } 
      prediction.matrix<float>()(0, k) = pred;
    }
  } // caloMuonRun3v1
  else if (name_=="caloMuonRun3v2") {
    getPrediction_run3_v2(muon, pred_vector);
    prediction = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, caloMuonInputs_run3_v2::NumberOfOutputs});
    for (int k = 0; k < caloMuonInputs_run3_v2::NumberOfOutputs; ++k) {
      const float pred = pred_vector[0].flat<float>()(k); // just one prediction vector for now
      if (!(pred >= 0 && pred <= 1)) {
        throw cms::Exception("DeepMuonCaloCompatibility")
            << "invalid prediction = " << pred << " for pred_index = " << k;
      } 
      prediction.matrix<float>()(0, k) = pred;
    }
  } // caloMuonRun3v2
  else {
    prediction = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, empty::NumberOfOutputs}); // for now only support [pion, muon]
    prediction.matrix<float>().setZero();
    prediction.matrix<float>()(0, empty::MuonOutput) = -1.0;
  }
  
  return prediction;
}

// run3_v1
void DeepMuonCaloCompatibility::getPrediction_run3_v1(const reco::Muon& muon, std::vector<tensorflow::Tensor>& pred_vector) {
  createInnerTrackBlockInputs(muon);

  tensorflow::run(&(cache_->getSession(name_)),
                  inputTensors_,
                  {outputName_},
                  &pred_vector);

}

// run3_v2
void DeepMuonCaloCompatibility::getPrediction_run3_v2(const reco::Muon& muon, std::vector<tensorflow::Tensor>& pred_vector) {
  createInnerTrackBlockInputs(muon);
  createHcalDigiBlockInputs(muon);

  tensorflow::run(&(cache_->getSession(name_)),
                  inputTensors_,
                  {outputName_},
                  &pred_vector);

}

void DeepMuonCaloCompatibility::createInnerTrackBlockInputs(const reco::Muon& muon) {
    tensorflow::Tensor& inputs = inputTensors_.at(kInnerTrack_).second;
    inputs.flat<float>().setZero();

    // at the moment, v1 and v2 must be the same
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

void DeepMuonCaloCompatibility::createHcalDigiBlockInputs(const reco::Muon& muon) {
    tensorflow::Tensor& inputs = inputTensors_.at(kHcalDigi_).second;
    inputs.flat<float>().setZero();

    namespace dnn = caloMuonInputs_run3_v2::HcalDigiBlockInputs;
    namespace dnnit = caloMuonInputs_run3_v2::InnerTrackBlockInputs;
    // in the means list, the hcal digis are after the inner tracks
    int nit = dnnit::NumberOfInputs;
    int v;
    int idh = 0;
    // ieta and iphi are relative to the first hcal digi
    HcalDetId did = muon.calEnergy().hcal_id;
    // energy is normalized to the sum of all hcal digi energies
    float total_energy = 0;
    for (auto it: muon.calEnergy().crossedHadRecHits) {
      total_energy += it.energy;
    }
    if (total_energy==0) total_energy=1;
    for (auto it: muon.calEnergy().crossedHadRecHits) {
      // limit the number of digis, this is large enough in run3 that we shouldnt actually reach this point even in the endcap
      if (idh>=dnn::NumberOfInputs) break;
      v = dnn::muon_calEnergy_crossedHadRecHits_ieta;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.detId.ieta()-did.ieta(), means_[v+nit], sigmas_[v+nit]);
      v = dnn::muon_calEnergy_crossedHadRecHits_iphi;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.detId.iphi()-did.iphi(), means_[v+nit], sigmas_[v+nit]);
      v = dnn::muon_calEnergy_crossedHadRecHits_depth;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.detId.depth(), means_[v+nit], sigmas_[v+nit]);
      v = dnn::muon_calEnergy_crossedHadRecHits_energy;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.energy/total_energy, means_[v+nit], sigmas_[v+nit]);
      v = dnn::muon_calEnergy_crossedHadRecHits_time;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.time, means_[v+nit], sigmas_[v+nit]);
      v = dnn::muon_calEnergy_crossedHadRecHits_chi2;
      inputs.tensor<float,3>()(0, v, idh) = getValueNorm(it.chi2, means_[v+nit], sigmas_[v+nit]);
      idh++;
    }

}
