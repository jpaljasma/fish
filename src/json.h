// src/json.h
//
// Minimal JSON value, parser, and serializer shared by the tests and the
// miner RPC client (extracted from tests/test_fishhash.cpp).
// Numbers use double; strings are UTF-8.

#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fishhash {

// ── Minimal JSON parser (objects, arrays, strings, numbers) ───────────────
struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  const Json& at(const std::string& k) const {
    return obj.at(k);
  }

  Json& at(const std::string& k) {
    return obj.at(k);
  }
};

class JsonParser {
 public:
  explicit JsonParser(std::string src) : s(std::move(src)) {}

  Json parseValue() {
    Json j = value(0);
    skip();
    if (i != s.size()) fail("unexpected trailing data");
    return j;
  }

 private:
  std::string s;
  size_t i = 0;

  [[noreturn]] void fail(const char* why) const {
    throw std::runtime_error("JSON at byte " + std::to_string(i) + ": " + why);
  }

  char peek() const { return i < s.size() ? s[i] : '\0'; }

  bool take(char c) {
    if (i == s.size() || s[i] != c) return false;
    ++i;
    return true;
  }

  void skip() {
    while (take(' ') || take('\t') || take('\n') || take('\r')) {}
  }

  void literal(const char* text) {
    while (*text) if (!take(*text++)) fail("invalid literal");
  }

  Json value(unsigned depth) {
    skip();
    Json j;
    switch (peek()) {
      case '{':
      case '[': {
        if (depth == 64) fail("nesting limit exceeded");
        const bool object = s[i++] == '{';
        j.type = object ? Json::Obj : Json::Arr;
        const char close = object ? '}' : ']';
        skip();
        if (take(close)) break;
        do {
          if (object) {
            skip();
            std::string key = parseString();
            skip();
            if (!take(':')) fail("expected ':'");
            j.obj.insert_or_assign(std::move(key), value(depth + 1));
          } else {
            j.arr.push_back(value(depth + 1));
          }
          skip();
        } while (take(','));
        if (!take(close)) fail("expected closing delimiter");
        break;
      }
      case '"': j.type = Json::Str; j.str = parseString(); break;
      case 't': literal("true"); j.type = Json::Bool; j.b = true; break;
      case 'f': literal("false"); j.type = Json::Bool; break;
      case 'n': literal("null"); break;
      default: j.type = Json::Num; j.num = parseNumber(); break;
    }
    return j;
  }

  uint32_t hexQuad() {
    uint32_t v = 0;
    for (int n = 0; n < 4; ++n) {
      const char c = peek();
      unsigned digit;
      if (c >= '0' && c <= '9') digit = c - '0';
      else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
      else fail("invalid Unicode escape");
      ++i;
      v = (v << 4) | digit;
    }
    return v;
  }

  void unicodeEscape(std::string& out) {
    uint32_t cp = hexQuad();
    if (cp >= 0xd800 && cp <= 0xdbff) {
      if (!take('\\') || !take('u')) fail("missing low surrogate");
      const uint32_t low = hexQuad();
      if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
      cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
    } else if (cp >= 0xdc00 && cp <= 0xdfff) {
      fail("unexpected low surrogate");
    }
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else {
      const int tails = cp < 0x800 ? 1 : cp < 0x10000 ? 2 : 3;
      const unsigned lead = tails == 1 ? 0xc0 : tails == 2 ? 0xe0 : 0xf0;
      out += static_cast<char>(lead | (cp >> (6 * tails)));
      for (int shift = 6 * (tails - 1); shift >= 0; shift -= 6)
        out += static_cast<char>(0x80 | ((cp >> shift) & 0x3f));
    }
  }

  void rawUtf8(std::string& out, unsigned char lead) {
    const size_t start = i - 1;
    if (lead < 0xc2 || lead > 0xf4) fail("invalid UTF-8 lead byte");
    const int tails = lead < 0xe0 ? 1 : lead < 0xf0 ? 2 : 3;
    const auto first = static_cast<unsigned char>(peek());
    if ((lead == 0xe0 && first < 0xa0) || (lead == 0xed && first >= 0xa0) ||
        (lead == 0xf0 && first < 0x90) || (lead == 0xf4 && first >= 0x90))
      fail("invalid UTF-8 code point");
    for (int n = 0; n < tails; ++n) {
      if ((static_cast<unsigned char>(peek()) & 0xc0) != 0x80)
        fail("invalid UTF-8 continuation");
      ++i;
    }
    out.append(s, start, 1 + tails);
  }

  std::string parseString() {
    if (!take('"')) fail("expected string");
    std::string out;
    while (i < s.size()) {
      const auto c = static_cast<unsigned char>(s[i++]);
      if (c == '"') return out;
      if (c < 0x20) fail("control character in string");
      if (c >= 0x80) {
        rawUtf8(out, c);
      } else if (c != '\\') {
        out += static_cast<char>(c);
      } else {
        if (i == s.size()) fail("incomplete string escape");
        switch (s[i++]) {
          case '"': case '\\': case '/': out += s[i - 1]; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': unicodeEscape(out); break;
          default: fail("invalid string escape");
        }
      }
    }
    fail("unterminated string");
  }

  void digits() {
    if (peek() < '0' || peek() > '9') fail("expected digit");
    do { ++i; } while (peek() >= '0' && peek() <= '9');
  }

  double parseNumber() {
    const auto start = i;
    take('-');
    if (!take('0')) digits();
    if (take('.')) digits();
    if (take('e') || take('E')) {
      if (!take('+')) take('-');
      digits();
    }
    double n;
    const auto result = std::from_chars(s.data() + start, s.data() + i, n);
    if (result.ec != std::errc{} || result.ptr != s.data() + i)
      fail("number outside supported range");
    return n;
  }
};

// ── Serializer ────────────────────────────────────────────────────────────
inline std::string escapeJsonString(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x",
                        static_cast<unsigned int>(static_cast<unsigned char>(c)));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

inline void appendJsonString(std::string& out, const std::string& s) {
  out += '"';
  out += escapeJsonString(s);
  out += '"';
}

inline void appendJson(std::string& out, const Json& j) {
  switch (j.type) {
    case Json::Null:
      out += "null";
      break;
    case Json::Bool:
      out += j.b ? "true" : "false";
      break;
    case Json::Num: {
      // Integers up to 2^53 (timestamps, sequence numbers) emit exactly.
      if (std::fabs(j.num) < 9007199254740992.0 && std::trunc(j.num) == j.num) {
        const int64_t asInt = static_cast<int64_t>(j.num);
        char buf[24];
        std::snprintf(buf, sizeof buf, "%lld", static_cast<long long>(asInt));
        out += buf;
      } else {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.17g", j.num);
        out += buf;
      }
      break;
    }
    case Json::Str:
      appendJsonString(out, j.str);
      break;
    case Json::Arr: {
      out += '[';
      for (size_t i = 0; i < j.arr.size(); ++i) {
        if (i) out += ',';
        appendJson(out, j.arr[i]);
      }
      out += ']';
      break;
    }
    case Json::Obj: {
      out += '{';
      bool first = true;
      for (const auto& kv : j.obj) {
        if (!first) out += ',';
        first = false;
        appendJsonString(out, kv.first);
        out += ':';
        appendJson(out, kv.second);
      }
      out += '}';
      break;
    }
  }
}

inline std::string jsonString(const Json& j) {
  std::string out;
  appendJson(out, j);
  return out;
}

}  // namespace fishhash
