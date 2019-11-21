#ifndef RecoMuon_MuonIdentification_DeepMuonCache_h
#define RecoMuon_MuonIdentification_DeepMuonCache_h

/*
 * \class DeepMuonCache
 *
 * Cache for tensorflow networks
 *
 * \author Devin Taylor, UC Davis
 */

#include <Math/VectorUtil.h>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/util/memmapped_file_system.h"

class DeepMuonCache {
  public:
    DeepMuonCache(const std::map<std::string, std::string>& graph_names, bool mem_mapped);
    ~DeepMuonCache();

    tensorflow::Session& getSession(const std::string& name = "") const { return *sessions_.at(name); }
    const tensorflow::GraphDef& getGraph(const std::string& name = "") const { return *graphs_.at(name); }

  private:
    std::map<std::string, std::shared_ptr<tensorflow::GraphDef> > graphs_;
    std::map<std::string, tensorflow::Session*> sessions_;
    std::map<std::string, std::unique_ptr<tensorflow::MemmappedEnv> > memmappedEnv_;
};
#endif
