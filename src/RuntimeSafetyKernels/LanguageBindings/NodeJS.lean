/--
Node.js bindings for Runtime Safety Kernels using N-API.

This module provides Node.js bindings for all RSK components,
enabling seamless integration with Node.js AI frameworks.
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape
import Lean.Data.Json

/-- Node.js bindings module -/
module RuntimeSafetyKernels.LanguageBindings.NodeJS

/-- Node.js-compatible sampling configuration -/
structure NodeJSSamplingConfig where
  method : String  -- "topk", "topp", "mirostat"
  k : Option Nat   -- for topK
  p : Option Float -- for topP
  temperature : Float
  targetEntropy : Option Float  -- for mirostat
  learningRate : Option Float   -- for mirostat
  maxIterations : Option Nat    -- for mirostat
  deriving Repr

/-- Node.js-compatible sampling result -/
structure NodeJSSamplingResult where
  probabilities : List Float
  selectedToken : Nat
  entropy : Float
  iterations : Nat
  deriving Repr

/-- Convert Node.js config to Lean config -/
def toLeanSamplingConfig (config : NodeJSSamplingConfig) : Option SamplingConfig :=
  match config.method with
  | "topk" =>
    match config.k with
    | none => none
    | some k => some (SamplingConfig.topK ⟨k, config.temperature⟩)
  | "topp" =>
    match config.p with
    | none => none
    | some p => some (SamplingConfig.topP ⟨p, config.temperature⟩)
  | "mirostat" =>
    match (config.targetEntropy, config.learningRate, config.maxIterations) with
    | (some target, some lr, some maxIter) =>
      some (SamplingConfig.mirostat ⟨target, lr, maxIter, 0.01⟩)
    | _ => none
  | _ => none

/-- Convert Lean result to Node.js result -/
def toNodeJSSamplingResult {n : Nat} (result : SamplingResult n) : NodeJSSamplingResult :=
  match result with
  | SamplingResult.topK r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, 1⟩
  | SamplingResult.topP r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, 1⟩
  | SamplingResult.mirostat r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, r.iterations⟩

/-- Node.js sampling function -/
def nodeJSSample (logits : List Float) (config : NodeJSSamplingConfig) : IO (Option NodeJSSamplingResult) := do
  let leanConfig := toLeanSamplingConfig config
  match leanConfig with
  | none => return none
  | some cfg =>
    let logitsVec := Vector.ofList logits
    let result := sample logitsVec cfg
    return some (toNodeJSSamplingResult result)

/-- Node.js-compatible policy configuration -/
structure NodeJSPolicyConfig where
  allowAllTokens : Bool
  blockedTokens : List Nat
  rateLimitTokensPerSecond : Nat
  maxContextLength : Nat
  maxTokensPerRequest : Nat
  deriving Repr

/-- Node.js-compatible policy result -/
structure NodeJSPolicyResult where
  allowed : Bool
  blockedToken : Option Nat
  rateLimited : Bool
  contextTooLong : Bool
  errorCode : Nat
  deriving Repr

/-- Convert Node.js policy config to Lean config -/
def toLeanPolicyConfig (config : NodeJSPolicyConfig) : PolicyConfig :=
  ⟨config.allowAllTokens, config.blockedTokens, config.rateLimitTokensPerSecond,
   config.maxContextLength, config.maxTokensPerRequest⟩

/-- Convert Lean policy result to Node.js result -/
def toNodeJSPolicyResult (result : PolicyGuardResult) : NodeJSPolicyResult :=
  ⟨result.allowed, result.blockedToken, result.rateLimited, result.contextTooLong, result.errorCode⟩

/-- Node.js policy guard function -/
def nodeJSPolicyGuard (config : NodeJSPolicyConfig) (token : Nat) (currentTime : Nat) : IO NodeJSPolicyResult := do
  let leanConfig := toLeanPolicyConfig config
  let decoderState := DecoderState.mk 0 0 0 0

  let (result, _) := policyGuard leanConfig decoderState token currentTime
  return toNodeJSPolicyResult result

/-- Node.js-compatible tensor shape -/
structure NodeJSTensorShape where
  dimensions : List Nat
  size : Nat
  deriving Repr

/-- Node.js-compatible tensor data -/
structure NodeJSTensorData where
  data : List Float
  shape : NodeJSTensorShape
  deriving Repr

/-- Convert Node.js tensor shape to Lean shape -/
def toLeanTensorShape (shape : NodeJSTensorShape) : TensorShape :=
  ⟨Vector.ofList shape.dimensions, shape.size⟩

/-- Convert Node.js tensor data to Lean data -/
def toLeanTensorData (tensor : NodeJSTensorData) : TensorData :=
  ⟨Vector.ofList tensor.data, toLeanTensorShape tensor.shape⟩

/-- Convert Lean tensor data to Node.js data -/
def toNodeJSTensorData (tensor : TensorData) : NodeJSTensorData :=
  ⟨tensor.data.toList, ⟨tensor.shape.dimensions.toList, tensor.shape.size⟩⟩

/-- Node.js tensor creation -/
def nodeJSCreateTensor (data : List Float) (shape : NodeJSTensorShape) : IO (Option NodeJSTensorData) := do
  let leanData := Vector.ofList data
  let leanShape := toLeanTensorShape shape

  let result := createTensor leanData leanShape
  match result with
  | TensorResult.success tensor => return some (toNodeJSTensorData tensor)
  | TensorResult.failure _ _ => return none

/-- Node.js tensor multiplication -/
def nodeJSMatrixMultiply (a : NodeJSTensorData) (b : NodeJSTensorData) : IO (Option NodeJSTensorData) := do
  let leanA := toLeanTensorData a
  let leanB := toLeanTensorData b

  let result := matrixMultiply leanA leanB
  match result with
  | TensorResult.success tensor => return some (toNodeJSTensorData tensor)
  | TensorResult.failure _ _ => return none

/-- Generate Node.js N-API module -/
def generateNodeJSModule : String :=
"#include <node_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// C function declarations
extern \"C\" {
    typedef struct {
        uint32_t method;
        float temperature;
        uint32_t k;
        float p;
        float target_entropy;
        float learning_rate;
        uint32_t max_iterations;
    } rsk_sampling_config_t;

    typedef struct {
        bool success;
        float* probs;
        uint32_t probs_len;
        uint32_t selected_token;
        float entropy;
        uint32_t iterations;
    } rsk_sampling_result_t;

    typedef struct {
        bool allow_all_tokens;
        uint32_t* blocked_tokens;
        uint32_t blocked_tokens_count;
        uint32_t rate_limit_tokens_per_second;
        uint32_t max_context_length;
        uint32_t max_tokens_per_request;
    } rsk_policy_config_t;

    typedef struct {
        bool allowed;
        bool blocked_token_present;
        uint32_t blocked_token;
        bool rate_limited;
        bool context_too_long;
        uint32_t error_code;
    } rsk_policy_guard_result_t;

    typedef struct {
        uint32_t* dimensions;
        uint32_t dimensions_count;
        uint32_t size;
    } rsk_tensor_shape_t;

    typedef struct {
        float* data;
        uint32_t data_count;
        rsk_tensor_shape_t shape;
    } rsk_tensor_data_t;

    typedef struct {
        bool success;
        rsk_tensor_data_t data;
        uint32_t error_code;
        const char* error_message;
    } rsk_tensor_result_t;

    rsk_sampling_result_t rsk_sample(const float* logits, uint32_t n, rsk_sampling_config_t config);
    rsk_policy_guard_result_t rsk_policy_guard(rsk_policy_config_t config, uint32_t token, uint64_t current_time);
    rsk_tensor_result_t rsk_create_tensor(const float* data, uint32_t data_count, rsk_tensor_shape_t shape);
    rsk_tensor_result_t rsk_matrix_multiply(rsk_tensor_data_t a, rsk_tensor_data_t b);
}

// Helper functions
napi_value create_sampling_config(napi_env env, napi_value obj, rsk_sampling_config_t* config) {
    napi_status status;
    napi_value method_val, temp_val, k_val, p_val, target_entropy_val, learning_rate_val, max_iterations_val;

    // Get method
    status = napi_get_named_property(env, obj, \"method\", &method_val);
    if (status != napi_ok) return nullptr;

    char method_str[32];
    size_t method_len;
    status = napi_get_value_string_utf8(env, method_val, method_str, sizeof(method_str), &method_len);
    if (status != napi_ok) return nullptr;

    if (strcmp(method_str, \"topk\") == 0) config->method = 0;
    else if (strcmp(method_str, \"topp\") == 0) config->method = 1;
    else if (strcmp(method_str, \"mirostat\") == 0) config->method = 2;
    else config->method = 0;

    // Get temperature
    status = napi_get_named_property(env, obj, \"temperature\", &temp_val);
    if (status == napi_ok) {
        status = napi_get_value_double(env, temp_val, &config->temperature);
        if (status != napi_ok) return nullptr;
    }

    // Get k (optional)
    status = napi_get_named_property(env, obj, \"k\", &k_val);
    if (status == napi_ok) {
        bool has_k;
        status = napi_has_property(env, obj, \"k\", &has_k);
        if (status == napi_ok && has_k) {
            uint32_t k;
            status = napi_get_value_uint32(env, k_val, &k);
            if (status == napi_ok) config->k = k;
        }
    }

    // Get p (optional)
    status = napi_get_named_property(env, obj, \"p\", &p_val);
    if (status == napi_ok) {
        bool has_p;
        status = napi_has_property(env, obj, \"p\", &has_p);
        if (status == napi_ok && has_p) {
            double p;
            status = napi_get_value_double(env, p_val, &p);
            if (status == napi_ok) config->p = (float)p;
        }
    }

    // Get target_entropy (optional)
    status = napi_get_named_property(env, obj, \"targetEntropy\", &target_entropy_val);
    if (status == napi_ok) {
        bool has_target;
        status = napi_has_property(env, obj, \"targetEntropy\", &has_target);
        if (status == napi_ok && has_target) {
            double target;
            status = napi_get_value_double(env, target_entropy_val, &target);
            if (status == napi_ok) config->target_entropy = (float)target;
        }
    }

    // Get learning_rate (optional)
    status = napi_get_named_property(env, obj, \"learningRate\", &learning_rate_val);
    if (status == napi_ok) {
        bool has_lr;
        status = napi_has_property(env, obj, \"learningRate\", &has_lr);
        if (status == napi_ok && has_lr) {
            double lr;
            status = napi_get_value_double(env, learning_rate_val, &lr);
            if (status == napi_ok) config->learning_rate = (float)lr;
        }
    }

    // Get max_iterations (optional)
    status = napi_get_named_property(env, obj, \"maxIterations\", &max_iterations_val);
    if (status == napi_ok) {
        bool has_max_iter;
        status = napi_has_property(env, obj, \"maxIterations\", &has_max_iter);
        if (status == napi_ok && has_max_iter) {
            uint32_t max_iter;
            status = napi_get_value_uint32(env, max_iterations_val, &max_iter);
            if (status == napi_ok) config->max_iterations = max_iter;
        }
    }

    return nullptr;
}

napi_value create_policy_config(napi_env env, napi_value obj, rsk_policy_config_t* config) {
    napi_status status;
    napi_value allow_all_val, blocked_tokens_val, rate_limit_val, max_context_val, max_tokens_val;

    // Get allowAllTokens
    status = napi_get_named_property(env, obj, \"allowAllTokens\", &allow_all_val);
    if (status == napi_ok) {
        bool allow_all;
        status = napi_get_value_bool(env, allow_all_val, &allow_all);
        if (status == napi_ok) config->allow_all_tokens = allow_all;
    }

    // Get blockedTokens array
    status = napi_get_named_property(env, obj, \"blockedTokens\", &blocked_tokens_val);
    if (status == napi_ok) {
        bool is_array;
        status = napi_is_array(env, blocked_tokens_val, &is_array);
        if (status == napi_ok && is_array) {
            uint32_t array_length;
            status = napi_get_array_length(env, blocked_tokens_val, &array_length);
            if (status == napi_ok && array_length > 0) {
                config->blocked_tokens = (uint32_t*)malloc(array_length * sizeof(uint32_t));
                config->blocked_tokens_count = array_length;

                for (uint32_t i = 0; i < array_length; i++) {
                    napi_value element;
                    status = napi_get_element(env, blocked_tokens_val, i, &element);
                    if (status == napi_ok) {
                        uint32_t token;
                        status = napi_get_value_uint32(env, element, &token);
                        if (status == napi_ok) {
                            config->blocked_tokens[i] = token;
                        }
                    }
                }
            }
        }
    }

    // Get rateLimitTokensPerSecond
    status = napi_get_named_property(env, obj, \"rateLimitTokensPerSecond\", &rate_limit_val);
    if (status == napi_ok) {
        uint32_t rate_limit;
        status = napi_get_value_uint32(env, rate_limit_val, &rate_limit);
        if (status == napi_ok) config->rate_limit_tokens_per_second = rate_limit;
    }

    // Get maxContextLength
    status = napi_get_named_property(env, obj, \"maxContextLength\", &max_context_val);
    if (status == napi_ok) {
        uint32_t max_context;
        status = napi_get_value_uint32(env, max_context_val, &max_context);
        if (status == napi_ok) config->max_context_length = max_context;
    }

    // Get maxTokensPerRequest
    status = napi_get_named_property(env, obj, \"maxTokensPerRequest\", &max_tokens_val);
    if (status == napi_ok) {
        uint32_t max_tokens;
        status = napi_get_value_uint32(env, max_tokens_val, &max_tokens);
        if (status == napi_ok) config->max_tokens_per_request = max_tokens;
    }

    return nullptr;
}

napi_value create_tensor_shape(napi_env env, napi_value obj, rsk_tensor_shape_t* shape) {
    napi_status status;
    napi_value dimensions_val, size_val;

    // Get dimensions array
    status = napi_get_named_property(env, obj, \"dimensions\", &dimensions_val);
    if (status == napi_ok) {
        bool is_array;
        status = napi_is_array(env, dimensions_val, &is_array);
        if (status == napi_ok && is_array) {
            uint32_t array_length;
            status = napi_get_array_length(env, dimensions_val, &array_length);
            if (status == napi_ok && array_length > 0) {
                shape->dimensions = (uint32_t*)malloc(array_length * sizeof(uint32_t));
                shape->dimensions_count = array_length;

                for (uint32_t i = 0; i < array_length; i++) {
                    napi_value element;
                    status = napi_get_element(env, dimensions_val, i, &element);
                    if (status == napi_ok) {
                        uint32_t dim;
                        status = napi_get_value_uint32(env, element, &dim);
                        if (status == napi_ok) {
                            shape->dimensions[i] = dim;
                        }
                    }
                }
            }
        }
    }

    // Get size
    status = napi_get_named_property(env, obj, \"size\", &size_val);
    if (status == napi_ok) {
        uint32_t size;
        status = napi_get_value_uint32(env, size_val, &size);
        if (status == napi_ok) shape->size = size;
    }

    return nullptr;
}

napi_value create_sampling_result(napi_env env, const rsk_sampling_result_t* result) {
    napi_status status;
    napi_value result_obj, probs_array, selected_token_val, entropy_val, iterations_val;

    status = napi_create_object(env, &result_obj);
    if (status != napi_ok) return nullptr;

    // Create probabilities array
    status = napi_create_array_with_length(env, result->probs_len, &probs_array);
    if (status == napi_ok) {
        for (uint32_t i = 0; i < result->probs_len; i++) {
            napi_value prob_val;
            status = napi_create_double(env, result->probs[i], &prob_val);
            if (status == napi_ok) {
                status = napi_set_element(env, probs_array, i, prob_val);
            }
        }
        status = napi_set_named_property(env, result_obj, \"probabilities\", probs_array);
    }

    // Set selected token
    status = napi_create_uint32(env, result->selected_token, &selected_token_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"selectedToken\", selected_token_val);
    }

    // Set entropy
    status = napi_create_double(env, result->entropy, &entropy_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"entropy\", entropy_val);
    }

    // Set iterations
    status = napi_create_uint32(env, result->iterations, &iterations_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"iterations\", iterations_val);
    }

    return result_obj;
}

napi_value create_policy_result(napi_env env, const rsk_policy_guard_result_t* result) {
    napi_status status;
    napi_value result_obj, allowed_val, blocked_token_val, rate_limited_val, context_too_long_val, error_code_val;

    status = napi_create_object(env, &result_obj);
    if (status != napi_ok) return nullptr;

    // Set allowed
    status = napi_create_boolean(env, result->allowed, &allowed_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"allowed\", allowed_val);
    }

    // Set blocked token (optional)
    if (result->blocked_token_present) {
        status = napi_create_uint32(env, result->blocked_token, &blocked_token_val);
        if (status == napi_ok) {
            status = napi_set_named_property(env, result_obj, \"blockedToken\", blocked_token_val);
        }
    }

    // Set rate limited
    status = napi_create_boolean(env, result->rate_limited, &rate_limited_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"rateLimited\", rate_limited_val);
    }

    // Set context too long
    status = napi_create_boolean(env, result->context_too_long, &context_too_long_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"contextTooLong\", context_too_long_val);
    }

    // Set error code
    status = napi_create_uint32(env, result->error_code, &error_code_val);
    if (status == napi_ok) {
        status = napi_set_named_property(env, result_obj, \"errorCode\", error_code_val);
    }

    return result_obj;
}

// N-API function implementations
napi_value sample(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 2;
    napi_value args[2];
    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok) return nullptr;

    if (argc < 2) {
        napi_throw_error(env, nullptr, \"Expected 2 arguments: logits array and config object\");
        return nullptr;
    }

    // Get logits array
    bool is_array;
    status = napi_is_array(env, args[0], &is_array);
    if (status != napi_ok || !is_array) {
        napi_throw_error(env, nullptr, \"First argument must be an array\");
        return nullptr;
    }

    uint32_t array_length;
    status = napi_get_array_length(env, args[0], &array_length);
    if (status != napi_ok) return nullptr;

    float* logits = (float*)malloc(array_length * sizeof(float));
    for (uint32_t i = 0; i < array_length; i++) {
        napi_value element;
        status = napi_get_element(env, args[0], i, &element);
        if (status == napi_ok) {
            double val;
            status = napi_get_value_double(env, element, &val);
            if (status == napi_ok) {
                logits[i] = (float)val;
            }
        }
    }

    // Get config object
    rsk_sampling_config_t config = {0};
    create_sampling_config(env, args[1], &config);

    // Call C function
    rsk_sampling_result_t result = rsk_sample(logits, array_length, config);

    // Create result object
    napi_value result_obj = create_sampling_result(env, &result);

    // Cleanup
    free(logits);
    if (config.blocked_tokens) free(config.blocked_tokens);

    return result_obj;
}

napi_value policyGuard(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 3;
    napi_value args[3];
    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok) return nullptr;

    if (argc < 3) {
        napi_throw_error(env, nullptr, \"Expected 3 arguments: config, token, currentTime\");
        return nullptr;
    }

    // Get config object
    rsk_policy_config_t config = {0};
    create_policy_config(env, args[0], &config);

    // Get token
    uint32_t token;
    status = napi_get_value_uint32(env, args[1], &token);
    if (status != napi_ok) return nullptr;

    // Get current time
    uint64_t current_time;
    status = napi_get_value_uint64(env, args[2], &current_time);
    if (status != napi_ok) return nullptr;

    // Call C function
    rsk_policy_guard_result_t result = rsk_policy_guard(config, token, current_time);

    // Create result object
    napi_value result_obj = create_policy_result(env, &result);

    // Cleanup
    if (config.blocked_tokens) free(config.blocked_tokens);

    return result_obj;
}

napi_value createTensor(napi_env env, napi_callback_info info) {
    napi_status status;
    size_t argc = 2;
    napi_value args[2];
    status = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (status != napi_ok) return nullptr;

    if (argc < 2) {
        napi_throw_error(env, nullptr, \"Expected 2 arguments: data array and shape object\");
        return nullptr;
    }

    // Get data array
    bool is_array;
    status = napi_is_array(env, args[0], &is_array);
    if (status != napi_ok || !is_array) {
        napi_throw_error(env, nullptr, \"First argument must be an array\");
        return nullptr;
    }

    uint32_t array_length;
    status = napi_get_array_length(env, args[0], &array_length);
    if (status != napi_ok) return nullptr;

    float* data = (float*)malloc(array_length * sizeof(float));
    for (uint32_t i = 0; i < array_length; i++) {
        napi_value element;
        status = napi_get_element(env, args[0], i, &element);
        if (status == napi_ok) {
            double val;
            status = napi_get_value_double(env, element, &val);
            if (status == napi_ok) {
                data[i] = (float)val;
            }
        }
    }

    // Get shape object
    rsk_tensor_shape_t shape = {0};
    create_tensor_shape(env, args[1], &shape);

    // Call C function
    rsk_tensor_result_t result = rsk_create_tensor(data, array_length, shape);

    // Create result object
    napi_value result_obj;
    if (result.success) {
        status = napi_create_object(env, &result_obj);
        if (status == napi_ok) {
            // Create data array
            napi_value data_array;
            status = napi_create_array_with_length(env, result.data.data_count, &data_array);
            if (status == napi_ok) {
                for (uint32_t i = 0; i < result.data.data_count; i++) {
                    napi_value val;
                    status = napi_create_double(env, result.data.data[i], &val);
                    if (status == napi_ok) {
                        status = napi_set_element(env, data_array, i, val);
                    }
                }
                status = napi_set_named_property(env, result_obj, \"data\", data_array);
            }

            // Create shape object
            napi_value shape_obj;
            status = napi_create_object(env, &shape_obj);
            if (status == napi_ok) {
                napi_value dims_array;
                status = napi_create_array_with_length(env, result.data.shape.dimensions_count, &dims_array);
                if (status == napi_ok) {
                    for (uint32_t i = 0; i < result.data.shape.dimensions_count; i++) {
                        napi_value dim;
                        status = napi_create_uint32(env, result.data.shape.dimensions[i], &dim);
                        if (status == napi_ok) {
                            status = napi_set_element(env, dims_array, i, dim);
                        }
                    }
                    status = napi_set_named_property(env, shape_obj, \"dimensions\", dims_array);
                }

                napi_value size_val;
                status = napi_create_uint32(env, result.data.shape.size, &size_val);
                if (status == napi_ok) {
                    status = napi_set_named_property(env, shape_obj, \"size\", size_val);
                }

                status = napi_set_named_property(env, result_obj, \"shape\", shape_obj);
            }
        }
    } else {
        // Return null on failure
        status = napi_get_null(env, &result_obj);
    }

    // Cleanup
    free(data);
    if (shape.dimensions) free(shape.dimensions);

    return result_obj;
}

// Module initialization
napi_value init(napi_env env, napi_value exports) {
    napi_status status;
    napi_value sample_fn, policy_guard_fn, create_tensor_fn;

    // Create function objects
    status = napi_create_function(env, nullptr, 0, sample, nullptr, &sample_fn);
    if (status != napi_ok) return nullptr;

    status = napi_create_function(env, nullptr, 0, policyGuard, nullptr, &policy_guard_fn);
    if (status != napi_ok) return nullptr;

    status = napi_create_function(env, nullptr, 0, createTensor, nullptr, &create_tensor_fn);
    if (status != napi_ok) return nullptr;

    // Add functions to exports
    status = napi_set_named_property(env, exports, \"sample\", sample_fn);
    if (status != napi_ok) return nullptr;

    status = napi_set_named_property(env, exports, \"policyGuard\", policy_guard_fn);
    if (status != napi_ok) return nullptr;

    status = napi_set_named_property(env, exports, \"createTensor\", create_tensor_fn);
    if (status != napi_ok) return nullptr;

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, init)"

/-- Generate package.json for Node.js package -/
def generatePackageJson : String :=
"{
  \"name\": \"runtime-safety-kernels\",
  \"version\": \"0.1.0\",
  \"description\": \"State-of-the-art runtime safety components for AI model inference\",
  \"main\": \"index.js\",
  \"scripts\": {
    \"install\": \"node-gyp rebuild\",
    \"test\": \"node test/test.js\",
    \"benchmark\": \"node benchmark/benchmark.js\"
  },
  \"keywords\": [
    \"ai\",
    \"machine-learning\",
    \"sampling\",
    \"safety\",
    \"runtime\",
    \"kernels\"
  ],
  \"author\": \"Runtime Safety Kernels Team\",
  \"license\": \"MIT\",
  \"engines\": {
    \"node\": \">=16.0.0\"
  },
  \"dependencies\": {
  },
  \"devDependencies\": {
    \"node-gyp\": \"^9.0.0\"
  },
  \"gypfile\": true,
  \"binary\": {
    \"module_name\": \"runtime_safety_kernels\",
    \"module_path\": \"./lib/binding\",
    \"host\": \"https://github.com/runtime-safety-kernels/node/releases/download/\",
    \"remote_path\": \"{version}\",
    \"package_name\": \"{node_abi}-{platform}-{arch}.tar.gz\"
  }
}"

/-- Generate binding.gyp for native compilation -/
def generateBindingGyp : String :=
"{
  \"targets\": [
    {
      \"target_name\": \"runtime_safety_kernels\",
      \"sources\": [
        \"src/binding.cpp\",
        \"src/rsk_sampler.c\",
        \"src/rsk_policy.c\",
        \"src/rsk_shape.c\"
      ],
      \"include_dirs\": [
        \"<!@(node -p \\\"require('node-addon-api').include\\\")\",
        \"include\"
      ],
      \"dependencies\": [
        \"<!(node -p \\\"require('node-addon-api').gyp\\\")\"
      ],
      \"cflags!\": [ \"-fno-exceptions\" ],
      \"cflags_cc!\": [ \"-fno-exceptions\" ],
      \"xcode_settings\": {
        \"GCC_ENABLE_CPP_EXCEPTIONS\": \"YES\",
        \"CLANG_CXX_LIBRARY\": \"libc++\",
        \"MACOSX_DEPLOYMENT_TARGET\": \"10.15\"
      },
      \"msvs_settings\": {
        \"VCCLCompilerTool\": {
          \"ExceptionHandling\": 1
        }
      },
      \"conditions\": [
        [\"OS==\\\"win\\\"\", {
          \"msvs_settings\": {
            \"VCCLCompilerTool\": {
              \"ExceptionHandling\": 1
            }
          }
        }]
      ]
    }
  ]
}"

/-- Generate index.js for Node.js module -/
def generateIndexJS : String :=
"const binding = require('./build/Release/runtime_safety_kernels.node');

class SamplingConfig {
  constructor(options = {}) {
    this.method = options.method || 'topk';
    this.k = options.k;
    this.p = options.p;
    this.temperature = options.temperature || 1.0;
    this.targetEntropy = options.targetEntropy;
    this.learningRate = options.learningRate;
    this.maxIterations = options.maxIterations;
  }
}

class PolicyConfig {
  constructor(options = {}) {
    this.allowAllTokens = options.allowAllTokens || false;
    this.blockedTokens = options.blockedTokens || [];
    this.rateLimitTokensPerSecond = options.rateLimitTokensPerSecond || 1000;
    this.maxContextLength = options.maxContextLength || 8192;
    this.maxTokensPerRequest = options.maxTokensPerRequest || 1000;
  }
}

class TensorShape {
  constructor(dimensions = [], size = 0) {
    this.dimensions = dimensions;
    this.size = size;
  }
}

// Export the native functions
module.exports = {
  // Sampling
  sample: (logits, config) => {
    if (!Array.isArray(logits)) {
      throw new Error('Logits must be an array');
    }
    if (!(config instanceof SamplingConfig)) {
      config = new SamplingConfig(config);
    }
    return binding.sample(logits, config);
  },

  // Policy guarding
  policyGuard: (config, token, currentTime) => {
    if (!(config instanceof PolicyConfig)) {
      config = new PolicyConfig(config);
    }
    if (typeof token !== 'number' || token < 0) {
      throw new Error('Token must be a non-negative number');
    }
    if (typeof currentTime !== 'number' || currentTime < 0) {
      throw new Error('Current time must be a non-negative number');
    }
    return binding.policyGuard(config, token, currentTime);
  },

  // Tensor operations
  createTensor: (data, shape) => {
    if (!Array.isArray(data)) {
      throw new Error('Data must be an array');
    }
    if (!(shape instanceof TensorShape)) {
      shape = new TensorShape(shape.dimensions, shape.size);
    }
    return binding.createTensor(data, shape);
  },

  // Classes
  SamplingConfig,
  PolicyConfig,
  TensorShape
};"

/-- Generate test file for Node.js -/
def generateNodeJSTest : String :=
"const rsk = require('../index.js');

// Test sampling
function testSampling() {
  console.log('Testing sampling...');

  const logits = [1.0, 2.0, 3.0, 4.0, 5.0];
  const config = new rsk.SamplingConfig({
    method: 'topk',
    k: 3,
    temperature: 1.0
  });

  const result = rsk.sample(logits, config);
  console.log('Sampling result:', result);

  // Verify probabilities sum to 1
  const sum = result.probabilities.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1.0) > 1e-6) {
    throw new Error(`Probabilities should sum to 1, got ${sum}`);
  }

  console.log('✓ Sampling test passed');
}

// Test policy guarding
function testPolicyGuard() {
  console.log('Testing policy guard...');

  const config = new rsk.PolicyConfig({
    allowAllTokens: false,
    blockedTokens: [1, 2, 3],
    rateLimitTokensPerSecond: 100,
    maxContextLength: 1000,
    maxTokensPerRequest: 100
  });

  // Test allowed token
  const allowedResult = rsk.policyGuard(config, 5, Date.now());
  if (!allowedResult.allowed) {
    throw new Error('Token 5 should be allowed');
  }

  // Test blocked token
  const blockedResult = rsk.policyGuard(config, 1, Date.now());
  if (blockedResult.allowed) {
    throw new Error('Token 1 should be blocked');
  }

  if (!blockedResult.blockedToken || blockedResult.blockedToken !== 1) {
    throw new Error('Blocked token should be 1');
  }

  console.log('✓ Policy guard test passed');
}

// Test tensor operations
function testTensorOps() {
  console.log('Testing tensor operations...');

  const data = [1.0, 2.0, 3.0, 4.0];
  const shape = new rsk.TensorShape([2, 2], 4);

  const tensor = rsk.createTensor(data, shape);
  if (!tensor) {
    throw new Error('Tensor creation failed');
  }

  if (tensor.data.length !== data.length) {
    throw new Error(`Expected ${data.length} data elements, got ${tensor.data.length}`);
  }

  if (tensor.shape.dimensions.length !== shape.dimensions.length) {
    throw new Error(`Expected ${shape.dimensions.length} dimensions, got ${tensor.shape.dimensions.length}`);
  }

  console.log('✓ Tensor operations test passed');
}

// Run all tests
function runTests() {
  console.log('Running Node.js bindings tests...\\n');

  try {
    testSampling();
    testPolicyGuard();
    testTensorOps();

    console.log('\\n✓ All tests passed!');
  } catch (error) {
    console.error('\\n✗ Test failed:', error.message);
    process.exit(1);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests();
}

module.exports = {
  testSampling,
  testPolicyGuard,
  testTensorOps,
  runTests
};"

/-- Main extraction entry point -/
def main : IO Unit := do
  -- Generate Node.js N-API module
  IO.FS.writeFile "src/extracted/binding.cpp" generateNodeJSModule

  -- Generate package.json
  IO.FS.writeFile "src/extracted/package.json" generatePackageJson

  -- Generate binding.gyp
  IO.FS.writeFile "src/extracted/binding.gyp" generateBindingGyp

  -- Generate index.js
  IO.FS.writeFile "src/extracted/index.js" generateIndexJS

  -- Generate test file
  IO.FS.writeFile "src/extracted/test/test.js" generateNodeJSTest

  -- Run extraction tests
  let testLogits := [1.0, 2.0, 3.0, 4.0, 5.0]
  let config := NodeJSSamplingConfig.mk "topk" (some 3) none 1.0 none none none

  let result ← nodeJSSample testLogits config
  match result with
  | none => IO.println "Node.js sampling test failed"
  | some res => IO.println s!"Node.js sampling test result: {res}"

  IO.println "Node.js bindings extraction completed successfully"

/-- Export for Lake build -/
#eval main
