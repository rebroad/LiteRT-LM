// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ODML pipeline to execute or benchmark LLM graph on device.
//
// The pipeline does the following
// 1) Read the corresponding parameters, weight and model file paths.
// 2) Construct a graph model with the setting.
// 3) Execute model inference and generate the output.

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu",
          "Executor backend to use for LLM execution (cpu, gpu, etc.)");
ABSL_FLAG(std::string, model_path, "", "Model path to use for LLM execution.");
ABSL_FLAG(std::string, input_prompt, "",
          "Input prompt to use for testing LLM execution.");
ABSL_FLAG(std::string, input_prompt_file, "", "File path to the input prompt.");
ABSL_FLAG(int, max_output_tokens, -1,
          "Maximum number of output tokens to generate. If negative, use the "
          "model default.");
ABSL_FLAG(int, num_output_candidates, 1,
          "Number of output candidates to generate.");
ABSL_FLAG(std::string, sampler_type, "",
          "Sampler type override. Supported values: auto, top_p, top_k, "
          "greedy.");
ABSL_FLAG(float, temperature, -1.0f,
          "Sampler temperature override. If negative, use the model default.");
ABSL_FLAG(float, top_p, -1.0f,
          "Top-p override. If negative, use the model default.");
ABSL_FLAG(int, top_k, -1,
          "Top-k override. If negative, use the model default.");
ABSL_FLAG(int, seed, -1,
          "Sampler seed override. If negative, use the model default.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::InputData;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::litert::lm::proto::SamplerParameters;
using ::nlohmann::json;

bool HasSamplerOverride() {
  const std::string sampler_type = absl::GetFlag(FLAGS_sampler_type);
  return (!sampler_type.empty() && !absl::EqualsIgnoreCase(sampler_type, "auto")) ||
         absl::GetFlag(FLAGS_temperature) >= 0.0f ||
         absl::GetFlag(FLAGS_top_p) >= 0.0f || absl::GetFlag(FLAGS_top_k) >= 0 ||
         absl::GetFlag(FLAGS_seed) >= 0;
}

absl::StatusOr<SamplerParameters::Type> ParseSamplerType(
    absl::string_view sampler_type) {
  if (sampler_type.empty() || absl::EqualsIgnoreCase(sampler_type, "auto")) {
    return SamplerParameters::TYPE_UNSPECIFIED;
  }
  if (absl::EqualsIgnoreCase(sampler_type, "top_p")) {
    return SamplerParameters::TOP_P;
  }
  if (absl::EqualsIgnoreCase(sampler_type, "top_k")) {
    return SamplerParameters::TOP_K;
  }
  if (absl::EqualsIgnoreCase(sampler_type, "greedy")) {
    return SamplerParameters::GREEDY;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported sampler_type: ", sampler_type,
                   " (expected auto, top_p, top_k, or greedy)"));
}

absl::Status ApplySamplerOverrides(litert::lm::SessionConfig& session_config) {
  if (!HasSamplerOverride()) {
    return absl::OkStatus();
  }

  SamplerParameters& sampler_params = session_config.GetMutableSamplerParams();
  const std::string sampler_type = absl::GetFlag(FLAGS_sampler_type);
  SamplerParameters::Type effective_type = SamplerParameters::TYPE_UNSPECIFIED;
  if (!sampler_type.empty() && !absl::EqualsIgnoreCase(sampler_type, "auto")) {
    auto parsed_type = ParseSamplerType(sampler_type);
    if (!parsed_type.ok()) {
      return parsed_type.status();
    }
    effective_type = *parsed_type;
  } else if (absl::GetFlag(FLAGS_top_k) >= 0 &&
             absl::GetFlag(FLAGS_top_p) < 0.0f &&
             absl::GetFlag(FLAGS_temperature) < 0.0f) {
    effective_type = SamplerParameters::TOP_K;
  } else {
    effective_type = SamplerParameters::TOP_P;
  }

  sampler_params.set_type(effective_type);
  sampler_params.set_k(
      absl::GetFlag(FLAGS_top_k) >= 0 ? absl::GetFlag(FLAGS_top_k) : 1);
  sampler_params.set_p(
      absl::GetFlag(FLAGS_top_p) >= 0.0f ? absl::GetFlag(FLAGS_top_p) : 0.95f);
  sampler_params.set_temperature(absl::GetFlag(FLAGS_temperature) >= 0.0f
                                     ? absl::GetFlag(FLAGS_temperature)
                                     : (effective_type == SamplerParameters::GREEDY
                                            ? 0.0f
                                            : 1.0f));
  if (absl::GetFlag(FLAGS_seed) >= 0) {
    sampler_params.set_seed(absl::GetFlag(FLAGS_seed));
  }
  return absl::OkStatus();
}

}  // namespace

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateMessageCallback() {
  return [](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      std::cout << "Error: " << message.status() << std::endl;
      return;
    }
    if (message->is_null()) {
      std::cout << std::endl << std::flush;
      return;
    }
    for (const auto& content : (*message)["content"]) {
      std::cout << content["text"].get<std::string>();
    }
    std::cout << std::flush;
  };
}

// Gets the input prompt from the command line flag or file.
std::string GetInputPrompt() {
  const std::string input_prompt = absl::GetFlag(FLAGS_input_prompt);
  const std::string input_prompt_file = absl::GetFlag(FLAGS_input_prompt_file);
  if (!input_prompt.empty() && !input_prompt_file.empty()) {
    ABSL_LOG(FATAL) << "Only one of --input_prompt and --input_prompt_file can "
                       "be specified.";
  }
  if (!input_prompt.empty()) {
    return input_prompt;
  }
  if (!input_prompt_file.empty()) {
    std::ifstream file(input_prompt_file);
    if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << input_prompt_file
                << std::endl;
      return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }
  // If no input prompt is provided, use the default prompt.
  return "What is the tallest building in the world?";
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  // Overrides the default for FLAGS_minloglevel to error.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                   ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(backend_str));
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));
  // Enable benchmark by default.
  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();

  // Create the engine.
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                    std::move(engine_settings)));

  // Create the conversation.
  std::unique_ptr<Conversation> conversation;
  auto session_config = litert::lm::SessionConfig::CreateDefault();
  session_config.SetNumOutputCandidates(absl::GetFlag(FLAGS_num_output_candidates));
  if (absl::GetFlag(FLAGS_max_output_tokens) > 0) {
    session_config.SetMaxOutputTokens(absl::GetFlag(FLAGS_max_output_tokens));
  }
  RETURN_IF_ERROR(ApplySamplerOverrides(session_config));
  ASSIGN_OR_RETURN(auto conversation_config,
                   ConversationConfig::Builder()
                       .SetSessionConfig(session_config)
                       .Build(*engine));
  ASSIGN_OR_RETURN(conversation,
                   Conversation::Create(*engine, conversation_config));

  // Prepare the message to send.
  json content_list = json::array();
  const std::string input_prompt = GetInputPrompt();
  std::cout << "input_prompt: " << input_prompt << std::endl;
  content_list.push_back({{"type", "text"}, {"text", input_prompt}});

  // Send the message and wait for the response, asynchronously log the
  // response.
  RETURN_IF_ERROR(conversation->SendMessageAsync(
      json::object({{"role", "user"}, {"content", content_list}}),
      CreateMessageCallback()));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));

  // Print the benchmark info.
  auto benchmark_info = conversation->GetBenchmarkInfo();
  std::cout << std::endl << *benchmark_info << std::endl;
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
