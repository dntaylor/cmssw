#ifndef RecoMuon_MuonIdentification_DeepMuonCaloCompatibility_h
#define RecoMuon_MuonIdentification_DeepMuonCaloCompatibility_h

/*
 * \class DeepMuonCaloCompatibility
 *
 * Base class for muon identification using TensorFlow DNN.
 *
 * \author Devin Taylor, UC Davis
 */

#include <Math/VectorUtil.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "MuonCaloCompatibilityBase.h"
#include "DeepMuonCache.h"

class DeepMuonCaloCompatibility : public MuonCaloCompatibilityBase {
  public:
    DeepMuonCaloCompatibility(const DeepMuonCache* cache)
        : MuonCaloCompatibilityBase(),
        cache_(cache)
        {}
    void configure(const edm::ParameterSet&) override;
    double evaluate( const reco::Muon& ) override;
    ~DeepMuonCaloCompatibility() override {}

  private:
    static constexpr float default_value = -999.;
    // Utility to convert value to a normalized float input for the DNN
    template <typename T>
    static float getValue(T value) {
      return std::isnormal(value) ? static_cast<float>(value) : 0.f;
    }

    template <typename T>
    static float getValueLinear(T value, float min_value, float max_value) {
      const float fixed_value = getValue(value);
      const float clamped_value = std::clamp(fixed_value, min_value, max_value);
      float transformed_value = (clamped_value - min_value) / (max_value - min_value);
      //std::cout << "linearized " << value << " " << min_value << " " << max_value << " " << transformed_value << std::endl;
      return transformed_value;
    }

    template <typename T>
    static float getValueNorm(T value, float mean, float sigma, float n_sigmas_max = -1) {
      const float fixed_value = getValue(value);
      const float norm_value = (fixed_value - mean) / sigma;
      float result;
      if (n_sigmas_max>0) {
          result = std::clamp(norm_value, -n_sigmas_max, n_sigmas_max);
      }
      else {
          result = norm_value;
      }
      //std::cout << "normalized " << value << " " << mean << " " << sigma << " " << result << std::endl;
      return result;
    }

    void createInnerTrackBlockInputs(const reco::Muon&);
    void createHcalDigiBlockInputs(const reco::Muon&);
    tensorflow::Tensor getPrediction(const reco::Muon&);
    void getPrediction_run3_v1(const reco::Muon&, std::vector<tensorflow::Tensor>&);
    void getPrediction_run3_v2(const reco::Muon&, std::vector<tensorflow::Tensor>&);

  protected:
    const DeepMuonCache* cache_;
  private:
    std::string name_;
    edm::FileInPath path_;
    edm::FileInPath meanPath_;
    std::vector<std::string> names_;
    std::vector<float> means_;
    std::vector<float> sigmas_;
    //std::unique_ptr<tensorflow::Tensor> innerTrackBlockTensor_;
    //std::unique_ptr<tensorflow::Tensor> hcalDigiBlockTensor_;
    std::vector<tensorflow::TensorShape> inputShapes_;
    std::vector<std::string> inputNames_;
    tensorflow::NamedTensorList inputTensors_;
    size_t kMuonPosition_;
    size_t kInnerTrack_;
    size_t kHcalDigi_;
    std::string outputName_;
};

#endif
