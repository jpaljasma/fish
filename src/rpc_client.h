// src/rpc_client.h
//
// Plain TCP client for the Iron Fish node's loopback RPC adapter.
// Frames are UTF-8 JSON terminated by a form feed. See docs/protocol.md.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <thread>

#include "json.h"

namespace fishhash {

struct RpcError : public std::exception {
  std::string what_;
  explicit RpcError(std::string w) : what_(std::move(w)) {}
  const char* what() const noexcept override { return what_.c_str(); }
};

class NodeRpc {
 public:
  // Connects to host:port. Throws RpcError on failure.
  static std::unique_ptr<NodeRpc> connect(const std::string& host, int port);
  ~NodeRpc();

  NodeRpc(const NodeRpc&) = delete;
  NodeRpc& operator=(const NodeRpc&) = delete;
  NodeRpc(NodeRpc&&) = delete;
  NodeRpc& operator=(NodeRpc&&) = delete;

  // Called by the mining thread. A null pointer omits the request payload
  // (undefined in the SDK). Server errors and timeouts throw RpcError.
  Json request(const std::string& method, const Json* data = nullptr);

  // A rejected or closed stream surfaces through the template accessors.
  void subscribeBlockTemplateStream();

  // Blocks until the next streamed block template. Throws RpcError on
  // disconnect or protocol error.
  Json nextBlockTemplate();

  // Non-blocking: a newer template has already been streamed.
  bool hasPendingTemplate();

 private:
  NodeRpc() = default;
  void send(const std::string& method, const Json* data, uint64_t mid,
            std::chrono::steady_clock::time_point deadline);
  void failConnection(std::exception_ptr error);
  void checkConnectionLocked() const;
  void readerLoop();
  void handleEnvelope(std::string msg);

  uintptr_t sock_ = ~uintptr_t{0};
  std::thread reader_;
  uint64_t midCounter_ = 0;

  std::mutex mu_;
  std::condition_variable cv_;
  // The mining thread issues one synchronous request alongside the subscription.
  uint64_t requestMid_ = 0;
  std::optional<Json> response_;
  std::string requestError_;
  std::optional<Json> latestTemplate_;
  uint64_t streamMid_ = 0;
  std::exception_ptr connError_;  // first terminal transport / protocol error
};

}  // namespace fishhash
