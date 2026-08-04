package com.fluidllm.fluidllm_backend.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.reactive.function.client.WebClient;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import com.fluidllm.fluidllm_backend.service.MicrofluidicChipValidationService;
import com.fluidllm.fluidllm_backend.service.MicrofluidicChipValidationService.ValidationResponse;
import com.fluidllm.fluidllm_backend.service.MicrofluidicChipSynthesisService;
import com.fluidllm.fluidllm_backend.service.MicrofluidicChipSynthesisService.ComsolPackageResponse;
import com.fluidllm.fluidllm_backend.service.MicrofluidicChipSynthesisService.ComsolResponse;
import com.fluidllm.fluidllm_backend.service.MicrofluidicChipSynthesisService.SynthesisResponse;

@RestController
@CrossOrigin(origins = "${cors.allowed.origin}")
@RequestMapping("/api")
public class FluidllmBackendController {

    private final WebClient webClient;
    private final MicrofluidicChipValidationService validationService;
    private final MicrofluidicChipSynthesisService synthesisService;
    private final Map<String, Object> microfluidicJsonSchema;
    private final String microfluidcReasoningGrammar;
    private final String ollamaBaselineModelname;
    private final String ollamaReasoningModelname;

    public FluidllmBackendController(WebClient.Builder webClientBuilder,
            @Value("${ollama.api.url}") String ollamaApiUrl,
            Map<String, Object> microfluidicJsonSchema,
            String microfluidcReasoningGrammar,
            MicrofluidicChipValidationService validationService,
            MicrofluidicChipSynthesisService synthesisService,
            @Value("${ollama.modelname.baseline}") String ollamaBaselineModelname,
            @Value("${ollama.modelname.reasoning}") String ollamaReasoningModelname) {
        this.webClient = webClientBuilder.baseUrl(ollamaApiUrl).build();
        this.microfluidicJsonSchema = microfluidicJsonSchema;
        this.microfluidcReasoningGrammar = microfluidcReasoningGrammar;
        this.validationService = validationService;
        this.synthesisService = synthesisService;
        this.ollamaBaselineModelname = ollamaBaselineModelname;
        this.ollamaReasoningModelname = ollamaReasoningModelname;
    }

    /**
     * Handles chat requests for generating and modifying microfluidic chip designs.
     * This endpoint proxies requests from the frontend to the locally hosted Ollama
     * API. It dynamically selects the appropriate LLM (baseline or reasoning) based
     * on the user's choice. To ensure structured and valid output, it constrains
     * the LLM's generation using either a JSON schema for baseline requests or a
     * custom grammar for reasoning requests. The response is streamed back to the
     * client as newline-delimited JSON (NDJSON), allowing the frontend to display
     * the token-by-token generation in real-time.
     */
    @PostMapping(value = "/chat", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_NDJSON_VALUE)
    public Flux<JsonNode> chat(@RequestBody Map<String, Object> request) {

        boolean reasoning = (boolean) request.get("reasoning");

        // Filter messages to include only user and assistant roles
        ObjectMapper mapper = new ObjectMapper();
        List<Map<String, String>> messages = mapper.convertValue(
                request.get("messages"),
                new TypeReference<List<Map<String, String>>>() {
                });
        messages.removeIf(message -> !("user".equals(message.get("role")) || "assistant".equals(message.get("role"))));

        Map<String, Object> payload;
        if (reasoning) {

            // Prepare ollama payload for reasoning model
            payload = Map.of(
                    "model", ollamaReasoningModelname,
                    "stream", true,
                    "messages", messages,
                    "think", true,
                    "options", Map.of(
                            "num_ctx", 32768));

        } else {

            // Prepare ollama payload for non-reasoning model
            payload = Map.of(
                    "model", ollamaBaselineModelname,
                    "stream", true,
                    "messages", messages,
                    "think", false);

        }

        // Make non-blocking WebClient request
        return webClient.post()
                .uri("/api/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(payload)
                .retrieve()
                .bodyToFlux(JsonNode.class)
                .onErrorResume(e -> Flux.error(new RuntimeException("Failed to connect to Ollama API", e)));
    }

    /**
     * Handles requests for the validation and correction of an LLM-generated chip
     * design. This endpoint exposes the MicrofluidicChipValidationService to the
     * frontend. It takes a JSON chip design and a boolean flag indicating whether
     * to perform algorithmic correction on the junction network. The service
     * identifies and reports problems for LLM-based re-prompting or applies direct
     * fixes.
     */
    @PostMapping(value = "/validate", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<?> validateChipDesign(@RequestBody ChipDesignValidationRequest request) {
        try {
            ValidationResponse response = validationService.validateChipDesign(request.getChipDesign(),
                    request.isFixJunctions());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity
                    .badRequest()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(new RuntimeException("Failed to validate chip", e));
        }
    }

    /**
     * This endpoint synthesizes a microfluidic chip's physical layout. It uses the
     * integrated 3M-DeSyn tool to translate a validated JSON design into a layout
     * by solving a constraint optimization problem. The endpoint returns a response
     * containing the chip visualization as an SVG string and an STL file, which is
     * omitted if its generation times out.
     */
    @PostMapping(value = "/synthesize", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<ResponseEntity<SynthesisResponse>> synthesizeChipDesign(
            @RequestBody ChipDesignSettingsRequest request) {
        return synthesisService.synthesizeChipDesign(request.getChipDesign(), request.getSettings())
                .map(response -> ResponseEntity.ok(response))
                .onErrorResume(e -> Mono.error(new RuntimeException("Failed to synthesize chip", e)));
    }

    /**
     * Generates standalone COMSOL Java source for a synthesized chip. The physical
     * layout is solved from the same conceptual chip design and settings used by
     * /synthesize, but the source is generated only when explicitly requested.
     */
    @PostMapping(value = "/comsol", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<ResponseEntity<ComsolResponse>> generateComsolJava(
            @RequestBody ComsolGenerationRequest request) {
        return synthesisService.generateComsolJava(
                        request.getChipDesign(),
                        request.getSettings(),
                        request.getDimension(),
                        request.getClassName())
                .map(response -> ResponseEntity.ok(response))
                .onErrorResume(e -> Mono.error(new RuntimeException("Failed to generate COMSOL Java", e)));
    }

    /**
     * 20260624 Generates a downloadable COMSOL package containing the Java source and a
     * README with basic usage instructions.
     */
    @PostMapping(value = "/comsol/package", consumes = MediaType.APPLICATION_JSON_VALUE, produces = "application/zip")
    public Mono<ResponseEntity<byte[]>> generateComsolPackage(
            @RequestBody ComsolGenerationRequest request) {
        return synthesisService.generateComsolPackage(
                        request.getChipDesign(),
                        request.getSettings(),
                        request.getDimension(),
                        request.getClassName())
                .map(this::comsolPackageResponse)
                .onErrorResume(e -> Mono.error(new RuntimeException("Failed to generate COMSOL package", e)));
    }

    private ResponseEntity<byte[]> comsolPackageResponse(ComsolPackageResponse response) {
        return ResponseEntity.ok()
                .contentType(MediaType.parseMediaType("application/zip"))
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + response.getZipFileName() + "\"")
                .body(response.getZipBytes());
    }

}

class ChipDesignValidationRequest {
    private JsonNode chipDesign;
    private boolean fixJunctions;

    public JsonNode getChipDesign() {
        return chipDesign;
    }

    public void setChipDesign(JsonNode chipDesign) {
        this.chipDesign = chipDesign;
    }

    public boolean isFixJunctions() {
        return fixJunctions;
    }

    public void setFixJunctions(boolean fixJunctions) {
        this.fixJunctions = fixJunctions;
    }

}

class ChipDesignSettingsRequest {
    private JsonNode chipDesign;
    private JsonNode settings;

    public JsonNode getChipDesign() {
        return chipDesign;
    }

    public void setChipDesign(JsonNode chipDesign) {
        this.chipDesign = chipDesign;
    }

    public JsonNode getSettings() {
        return settings;
    }

    public void setSettings(JsonNode settings) {
        this.settings = settings;
    }

}

class ComsolGenerationRequest {
    private JsonNode chipDesign;
    private JsonNode settings;
    private String dimension;
    private String className;

    public JsonNode getChipDesign() {
        return chipDesign;
    }

    public void setChipDesign(JsonNode chipDesign) {
        this.chipDesign = chipDesign;
    }

    public JsonNode getSettings() {
        return settings;
    }

    public void setSettings(JsonNode settings) {
        this.settings = settings;
    }

    public String getDimension() {
        return dimension;
    }

    public void setDimension(String dimension) {
        this.dimension = dimension;
    }

    public String getClassName() {
        return className;
    }

    public void setClassName(String className) {
        this.className = className;
    }

}
