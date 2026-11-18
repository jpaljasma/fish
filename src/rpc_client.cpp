// src/rpc_client.cpp
//
// NodeRpc implementation (see src/rpc_client.h for the wire protocol).

#include "rpc_client.h"

#include <chrono>
#include <climits>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace fishhash {
namespace {

#ifdef _WIN32
using SockT = SOCKET;  // UINT_PTR on Windows
constexpr SockT kSockInvalid = INVALID_SOCKET;
#else
using SockT = int;
constexpr SockT kSockInvalid = -1;
#endif

using Clock = std::chrono::steady_clock;
constexpr auto kRequestTimeout = std::chrono::seconds(60);
constexpr size_t kMaxFrameBytes = 16 * 1024 * 1024;

void shutdownSocket(SockT s) {
#ifdef _WIN32
  ::shutdown(s, SD_BOTH);
#else
  ::shutdown(s, SHUT_RDWR);
#endif
}

void setSendTimeout(SockT s, Clock::time_point deadline) {
  const auto ms = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
  if (ms <= 0) throw RpcError("send to node timed out");
#ifdef _WIN32
  const DWORD timeout = static_cast<DWORD>(ms);
#else
  const timeval timeout{static_cast<long>(ms / 1000), static_cast<long>(ms % 1000 * 1000)};
#endif
  if (::setsockopt(s, SOL_SOCKET, SO_SNDTIMEO,
                   reinterpret_cast<const char*>(&timeout), sizeof(timeout)) != 0)
    throw RpcError("cannot set node send timeout");
}

void closeSocket(SockT s) {
#ifdef _WIN32
  ::closesocket(s);
#else
  ::close(s);
#endif
}

// All errors relevant to a route arrive in the nested response payload.
std::string errorMessage(const Json& data) {
  const auto it = data.obj.find("message");
  return it != data.obj.end() && it->second.type == Json::Str ? it->second.str : "no error message";
}

}  // namespace

std::unique_ptr<NodeRpc> NodeRpc::connect(const std::string& host, int port) {
#ifdef _WIN32
  static std::once_flag wsOnce;
  std::call_once(wsOnce, [] {
    WSADATA d;
    WSAStartup(MAKEWORD(2, 2), &d);
  });
#endif
  auto c = std::unique_ptr<NodeRpc>(new NodeRpc());

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const std::string portStr = std::to_string(port);
  if (::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res) != 0 || !res) {
    throw RpcError("cannot resolve host " + host);
  }
  SockT s = kSockInvalid;
  for (const addrinfo* ai = res; ai; ai = ai->ai_next) {
    s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (s == kSockInvalid) continue;
    if (::connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) break;
    closeSocket(s);
    s = kSockInvalid;
  }
  freeaddrinfo(res);
  if (s == kSockInvalid)
    throw RpcError("connect to " + host + ":" + portStr + " failed");
  c->sock_ = static_cast<uintptr_t>(s);
  c->reader_ = std::thread(&NodeRpc::readerLoop, c.get());
  return c;
}

NodeRpc::~NodeRpc() {
  if (sock_ == ~uintptr_t{0}) return;
  const auto s = static_cast<SockT>(sock_);
  if (reader_.joinable()) {
    shutdownSocket(s);
    reader_.join();
  }
  closeSocket(s);
}

void NodeRpc::send(const std::string& method, const Json* data, uint64_t mid,
                   Clock::time_point deadline) {
  std::string body = "{\"type\":\"message\",\"data\":{\"mid\":" +
                     std::to_string(mid) + ",\"type\":";
  appendJsonString(body, method);
  if (data) {
    body += ",\"data\":";
    appendJson(body, *data);
  }
  body += "}}\f";
  const auto s = static_cast<SockT>(sock_);
  try {
    for (size_t off = 0; off < body.size();) {
      setSendTimeout(s, deadline);
      const auto remaining = body.size() - off;
      const int count = remaining > INT_MAX ? INT_MAX : static_cast<int>(remaining);
#ifdef MSG_NOSIGNAL
      constexpr int flags = MSG_NOSIGNAL;
#else
      constexpr int flags = 0;
#endif
      const int n = static_cast<int>(::send(s, body.data() + off, count, flags));
      if (n <= 0) throw RpcError("send to node failed or timed out");
      off += static_cast<size_t>(n);
    }
  } catch (...) {
    // A partial frame cannot safely be followed by another request.
    failConnection(std::current_exception());
    throw;
  }
}

void NodeRpc::failConnection(std::exception_ptr error) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (connError_) return;
    connError_ = std::move(error);
  }
  cv_.notify_all();
  shutdownSocket(static_cast<SockT>(sock_));
}

void NodeRpc::checkConnectionLocked() const {
  if (!connError_) return;
  try {
    std::rethrow_exception(connError_);
  } catch (const RpcError&) {
    throw;
  } catch (const std::exception& e) {
    throw RpcError(e.what());
  } catch (...) {
    throw RpcError("node connection failed");
  }
}

void NodeRpc::readerLoop() try {
  char buf[16384];
  std::string msg;
  for (;;) {
    const int n = static_cast<int>(::recv(static_cast<SockT>(sock_), buf, sizeof buf, 0));
    if (n <= 0) throw RpcError("connection closed by node");
    const char* begin = buf;
    const char* end = buf + n;
    while (begin != end) {
      const auto* sep = static_cast<const char*>(std::memchr(begin, '\f', end - begin));
      const size_t count = static_cast<size_t>((sep ? sep : end) - begin);
      if (count > kMaxFrameBytes - msg.size()) throw RpcError("RPC frame too large");
      msg.append(begin, count);
      if (!sep) break;
      handleEnvelope(std::move(msg));
      msg.clear();
      begin = sep + 1;
    }
  }
} catch (...) {
  failConnection(std::current_exception());
}

void NodeRpc::handleEnvelope(std::string msg) {
  Json env = JsonParser(std::move(msg)).parseValue();
  auto type = env.obj.find("type"), frame = env.obj.find("data");
  if (type == env.obj.end() || type->second.type != Json::Str ||
      frame == env.obj.end() || frame->second.type != Json::Obj)
    throw RpcError("malformed RPC envelope");
  const auto& kind = type->second.str;
  if (kind == "error" || kind == "malformedRequest")
    throw RpcError("node error: " + errorMessage(frame->second));
  auto& fields = frame->second.obj;
  auto id = fields.find("id"), payload = fields.find("data");
  if (id == fields.end() || id->second.type != Json::Num || id->second.num < 1 ||
      id->second.num > 9007199254740991.0 || std::trunc(id->second.num) != id->second.num)
    throw RpcError("RPC response has invalid id");
  const auto mid = static_cast<uint64_t>(id->second.num);

  std::lock_guard<std::mutex> lk(mu_);
  if (connError_) return;
  if (kind == "stream") {
    if (mid != streamMid_) return;
    if (payload == fields.end() || payload->second.type != Json::Obj)
      throw RpcError("missing block template object");
    latestTemplate_ = std::move(payload->second);
  } else if (kind == "message") {
    auto status = fields.find("status");
    if (status == fields.end() || status->second.type != Json::Num ||
        status->second.num < 0 || status->second.num > INT_MAX ||
        std::trunc(status->second.num) != status->second.num)
      throw RpcError("RPC response has invalid status");
    const int code = static_cast<int>(status->second.num);
    std::string error;
    if (code < 200 || code >= 300) {
      error = "RPC " + std::to_string(code);
      if (payload != fields.end()) error += ": " + errorMessage(payload->second);
    }
    if (mid == streamMid_)
      throw RpcError(error.empty() ? "block template stream ended" : std::move(error));
    if (mid != requestMid_) return;
    requestError_ = std::move(error);
    response_ = payload == fields.end() ? Json{} : std::move(payload->second);
  } else {
    throw RpcError("unknown RPC envelope type: " + kind);
  }
  cv_.notify_all();
}

Json NodeRpc::request(const std::string& method, const Json* data) {
  const auto deadline = Clock::now() + kRequestTimeout;
  const uint64_t mid = ++midCounter_;
  {
    std::lock_guard<std::mutex> lk(mu_);
    checkConnectionLocked();
    requestMid_ = mid;
    response_.reset();
    requestError_.clear();
  }
  send(method, data, mid, deadline);
  std::unique_lock<std::mutex> lk(mu_);
  const bool received = cv_.wait_until(lk, deadline, [&] {
    return response_.has_value() || connError_;
  });
  requestMid_ = 0;
  if (!received) throw RpcError(method + " timed out");
  if (!requestError_.empty()) throw RpcError(requestError_);
  if (!response_) checkConnectionLocked();
  return std::move(*response_);
}

void NodeRpc::subscribeBlockTemplateStream() {
  const auto deadline = Clock::now() + kRequestTimeout;
  const uint64_t mid = ++midCounter_;
  {
    std::lock_guard<std::mutex> lk(mu_);
    checkConnectionLocked();
    streamMid_ = mid;
  }
  send("miner/blockTemplateStream", nullptr, mid, deadline);
}

Json NodeRpc::nextBlockTemplate() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [&] { return latestTemplate_.has_value() || connError_; });
  checkConnectionLocked();
  Json t = std::move(*latestTemplate_);
  latestTemplate_.reset();
  return t;
}

bool NodeRpc::hasPendingTemplate() {
  std::lock_guard<std::mutex> lk(mu_);
  checkConnectionLocked();
  return latestTemplate_.has_value();
}

}  // namespace fishhash
