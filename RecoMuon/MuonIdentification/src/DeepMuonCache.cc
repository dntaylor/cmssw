/*
 * \class DeepMuonCache
 *
 * Implementation of cache for deep muon tensorflow networks
 *
 * \author Devin Taylor, UC Davis
 */

#include "RecoMuon/MuonIdentification/interface/DeepMuonCache.h"

DeepMuonCache::DeepMuonCache(const std::map<std::string, std::string>& graph_names, bool mem_mapped) {
  for (const auto& graph_entry : graph_names) {
    tensorflow::SessionOptions options;
    tensorflow::setThreading(options, 1, "no_threads");

    const std::string& entry_name = graph_entry.first;
    const std::string& graph_file = graph_entry.second;
    if (mem_mapped) {
      memmappedEnv_[entry_name] = std::make_unique<tensorflow::MemmappedEnv>(tensorflow::Env::Default());
      const tensorflow::Status mmap_status = memmappedEnv_.at(entry_name)->InitializeFromFile(graph_file);
      if (!mmap_status.ok()) {
        throw cms::Exception("DeepMuonCache: unable to initalize memmapped environment for ")
            << graph_file << ". \n"
            << mmap_status.ToString();
      }

      graphs_[entry_name] = std::make_unique<tensorflow::GraphDef>();
      const tensorflow::Status load_graph_status =
          ReadBinaryProto(memmappedEnv_.at(entry_name).get(),
                          tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                          graphs_.at(entry_name).get());
      if (!load_graph_status.ok())
        throw cms::Exception("DeepMuonCache: unable to load graph from ") << graph_file << ". \n"
                                                                         << mmap_status.ToString();
      options.config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(
          ::tensorflow::OptimizerOptions::L0);
      options.env = memmappedEnv_.at(entry_name).get();

      sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);

    } else {
      graphs_[entry_name].reset(tensorflow::loadGraphDef(graph_file));
      sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);
    }
  }
}

DeepMuonCache::~DeepMuonCache() {
  for (auto& session_entry : sessions_) {
    tensorflow::closeSession(session_entry.second);
  }
}

