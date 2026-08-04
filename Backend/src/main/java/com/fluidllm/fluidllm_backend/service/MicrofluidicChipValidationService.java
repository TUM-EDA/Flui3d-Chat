package com.fluidllm.fluidllm_backend.service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.AllDirectedPaths;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DirectedMultigraph;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import javafx.util.Pair;

@Service
public class MicrofluidicChipValidationService {
    private final ObjectMapper objectMapper;

    public MicrofluidicChipValidationService(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    /**
     * Contains the comprehensive validation and correction of a microfluidic chip
     * design. This method serves as the main entry point for the validation
     * service, processing a given chip design through a multi-stage validation and
     * correction pipeline. The process includes fixing simple omissions
     * algorithmically (like unconnected ports and missing default parameters) and
     * handling complex junction network errors. For junction errors, it can either
     * collect problems for LLM-based re-prompting or perform direct algorithmic
     * correction, depending on the fixJunctions flag.
     */
    public ValidationResponse validateChipDesign(JsonNode chipDesign, boolean fixJunctions) {
        List<String> problems = new ArrayList<>();
        List<String> addedJunctions = new ArrayList<>();
        ObjectNode correctedChipDesign = null;
        boolean hasChanges = false;

        // Create a working copy of the chip design
        ObjectNode workingCopy = chipDesign.deepCopy();

        // Step 1: Check for unused ports and add necessary connections
        hasChanges |= validateAndFixPortConnections(workingCopy);

        // Step 2: Validate and add missing component parameters
        hasChanges |= validateAndFixComponentParams(workingCopy);

        // Step 3: Normalize junction definitions
        hasChanges |= normalizeJunctionDefinitions(workingCopy);

        // Steps 4-6: Validate junctions (if fixJunctions=false these steps only collect
        // problems)
        do {
            problems.clear();
            boolean localChanges = false;

            // Step 4: Validate/fix junction references
            localChanges |= validateJunctionReferences(workingCopy, problems, fixJunctions);

            // Step 5: Validate/fix junction connectivity
            localChanges |= validateJunctionConnectivity(workingCopy, problems, fixJunctions);

            hasChanges |= localChanges;
        } while (fixJunctions && !problems.isEmpty());

        if (problems.isEmpty()) {
            // Step 6: Validate/fix junction flows
            hasChanges |= validateJunctionFlows(workingCopy, problems, addedJunctions, fixJunctions);
        }

        // Only include correctedChipDesign if changes were made
        if (hasChanges) {
            updateJunctionNames(workingCopy, addedJunctions);
            correctedChipDesign = workingCopy;
        }

        return new ValidationResponse(correctedChipDesign, problems, addedJunctions);
    }

    /**
     * Validates and corrects port connectivity issues in the chip design (Error
     * Type 1). This method ensures that all component ports are properly connected.
     * It automatically generates and connects new, dedicated inlet or outlet
     * components for any orphaned ports found in the design's 'connections' array,
     * ensuring a topologically complete design.
     */
    private boolean validateAndFixPortConnections(ObjectNode chipDesign) {
        Set<String> sources = new HashSet<>();
        Set<String> targets = new HashSet<>();
        Map<String, Set<String>> dropletPorts = new HashMap<>();
        Map<String, Set<String>> filterPorts = new HashMap<>();
        boolean hasChanges = false;

        // Build a set of DLD filter IDs to distinguish from pillar_matrix filters
        Set<String> dldFilterIds = new HashSet<>();
        JsonNode filterParams = chipDesign.get("component_params").get("filters");
        if (filterParams != null) {
            for (JsonNode fp : filterParams) {
                if (fp.has("type") && "dld".equals(fp.get("type").asText())) {
                    dldFilterIds.add(fp.get("id").asText());
                }
            }
        }

        // Collect all sources and targets
        for (JsonNode connection : chipDesign.get("connections")) {
            String source = connection.get("source").asText();
            String target = connection.get("target").asText();

            // v2: droplet generators have a single port (droplet_N), no sub-ports
            if (source.startsWith("droplet_")) {
                String baseId = extractBaseId(source, "droplet_");
                dropletPorts.computeIfAbsent(baseId, k -> new HashSet<>()).add(source);
                sources.add(baseId);
            } else if (source.startsWith("filter_") && source.matches(".*_(smaller|larger)$")) {
                String baseId = source.substring(0, source.lastIndexOf('_'));
                filterPorts.computeIfAbsent(baseId, k -> new HashSet<>()).add(source);
                sources.add(baseId);
            } else {
                sources.add(source);
            }

            if (target.startsWith("droplet_")) {
                String baseId = extractBaseId(target, "droplet_");
                dropletPorts.computeIfAbsent(baseId, k -> new HashSet<>()).add(target);
                targets.add(baseId);
            } else if (target.startsWith("filter_")) {
                filterPorts.computeIfAbsent(target, k -> new HashSet<>()).add(target);
                targets.add(target);
            } else {
                targets.add(target);
            }

        }

        // Check and fix unused ports
        ArrayNode connectionsArray = (ArrayNode) chipDesign.get("connections");
        int nextInletId = getNextComponentId(sources, "inlet_");
        int nextOutletId = getNextComponentId(targets, "outlet_");

        // Fix regular components (including tesla_valve)
        for (String source : new HashSet<>(sources)) {
            if (!source.startsWith("inlet_") && !source.contains("_smaller") && !source.contains("_larger") &&
                    !targets.contains(source)) {
                connectionsArray.add(objectMapper.createObjectNode()
                        .put("source", "inlet_" + nextInletId++)
                        .put("target", source));
                hasChanges = true;
            }
        }

        for (String target : new HashSet<>(targets)) {
            if (!target.startsWith("outlet_") && !target.contains("_smaller") && !target.contains("_larger") &&
                    !sources.contains(target)) {
                connectionsArray.add(objectMapper.createObjectNode()
                        .put("source", target)
                        .put("target", "outlet_" + nextOutletId++));
                hasChanges = true;
            }
        }

        // v2: Fix droplet generators — single input port only
        for (Map.Entry<String, Set<String>> entry : dropletPorts.entrySet()) {
            String baseId = entry.getKey();
            Set<String> ports = entry.getValue();

            // Ensure there is an inlet connection to the droplet
            if (!ports.contains(baseId) && !targets.contains(baseId)) {
                connectionsArray.add(objectMapper.createObjectNode()
                        .put("source", "inlet_" + nextInletId++)
                        .put("target", baseId));
                hasChanges = true;
            }
            // Ensure there is an outlet connection from the droplet
            if (!sources.contains(baseId)) {
                connectionsArray.add(objectMapper.createObjectNode()
                        .put("source", baseId)
                        .put("target", "outlet_" + nextOutletId++));
                hasChanges = true;
            }
        }

        // Fix DLD filters (2 outputs: smaller/larger)
        for (Map.Entry<String, Set<String>> entry : filterPorts.entrySet()) {
            String baseId = entry.getKey();
            Set<String> ports = entry.getValue();

            // Only add smaller/larger outputs for DLD-type filters
            if (dldFilterIds.contains(baseId)) {
                if (!ports.contains(baseId + "_smaller")) {
                    connectionsArray.add(objectMapper.createObjectNode()
                            .put("source", baseId + "_smaller")
                            .put("target", "outlet_" + nextOutletId++));
                    hasChanges = true;
                }
                if (!ports.contains(baseId + "_larger")) {
                    connectionsArray.add(objectMapper.createObjectNode()
                            .put("source", baseId + "_larger")
                            .put("target", "outlet_" + nextOutletId++));
                    hasChanges = true;
                }
            }
            if (!ports.contains(baseId)) {
                connectionsArray.add(objectMapper.createObjectNode()
                        .put("source", "inlet_" + nextInletId++)
                        .put("target", baseId));
                hasChanges = true;
            }
        }

        return hasChanges;
    }

    /**
     * Extracts the base ID (prefix + number) from a component ID string.
     * E.g., "droplet_1" from "droplet_1" or "droplet_1_something".
     */
    private String extractBaseId(String id, String prefix) {
        String rest = id.substring(prefix.length());
        String[] parts = rest.split("_", 2);
        return prefix + parts[0];
    }

    /**
     * Validates and corrects missing component parameters in the chip design (Error
     * Type 2). This method ensures that all components requiring specific
     * parameters (mixers, delays, chambers, filters) have them defined. If a
     * component is used in the design but its parameters are missing from the
     * 'component_params' section, this method adds them using the default values.
     */
    private boolean validateAndFixComponentParams(ObjectNode chipDesign) {
        Set<String> usedComponents = new HashSet<>();
        boolean hasChanges = false;

        // Collect all used components from connections
        for (JsonNode connection : chipDesign.get("connections")) {
            String source = connection.get("source").asText();
            String target = connection.get("target").asText();

            if (source.startsWith("mixer_") || source.startsWith("delay_") ||
                    source.startsWith("chamber_") || source.startsWith("filter_") ||
                    source.startsWith("droplet_") || source.startsWith("tesla_valve_")) {
                // v2: tesla_valve has 3-part ID (tesla_valve_N)
                if (source.startsWith("tesla_valve_")) {
                    String[] parts = source.split("_");
                    if (parts.length >= 3) usedComponents.add(parts[0] + "_" + parts[1] + "_" + parts[2]);
                } else {
                    usedComponents.add(source.split("_")[0] + "_" + source.split("_")[1]);
                }
            }
            if (target.startsWith("mixer_") || target.startsWith("delay_") ||
                    target.startsWith("chamber_") || target.startsWith("filter_") ||
                    target.startsWith("droplet_") || target.startsWith("tesla_valve_")) {
                if (target.startsWith("tesla_valve_")) {
                    String[] parts = target.split("_");
                    if (parts.length >= 3) usedComponents.add(parts[0] + "_" + parts[1] + "_" + parts[2]);
                } else {
                    usedComponents.add(target.split("_")[0] + "_" + target.split("_")[1]);
                }
            }
        }

        ObjectNode componentParams = (ObjectNode) chipDesign.get("component_params");

        // Ensure all v2 param buckets exist
        if (!componentParams.has("droplets")) componentParams.putArray("droplets");
        if (!componentParams.has("tesla_valves")) componentParams.putArray("tesla_valves");

        // Check and update parameters for each component type
        hasChanges |= updateComponentParams(componentParams, "mixers", usedComponents, "mixer_");
        hasChanges |= updateComponentParams(componentParams, "delays", usedComponents, "delay_");
        hasChanges |= updateComponentParams(componentParams, "chambers", usedComponents, "chamber_");
        hasChanges |= updateComponentParams(componentParams, "filters", usedComponents, "filter_");
        hasChanges |= updateComponentParams(componentParams, "droplets", usedComponents, "droplet_");
        hasChanges |= updateComponentParams(componentParams, "tesla_valves", usedComponents, "tesla_valve_");

        return hasChanges;
    }

    /**
     * Normalizes junction definitions before the stricter junction validation
     * steps run. This fills missing junction IDs, accepts common singular field
     * names from model output, such as source/target, normalizes common junction
     * ID aliases like merge_N, split_N, and juction_N to junction_N, and
     * rewrites sources[]/targets[] references to match any renamed junction IDs.
     * Unknown malformed junction IDs are assigned the next available junction_N
     * name so later validation steps can rely on the standard junction ID format.
     */
    private boolean normalizeJunctionDefinitions(ObjectNode chipDesign) {
        JsonNode junctionsNode = chipDesign.get("junctions");
        if (!(junctionsNode instanceof ArrayNode junctions)) {
            return false;
        }

        boolean hasChanges = false;
        Set<String> usedIds = new HashSet<>();
        Map<String, String> renameMap = new HashMap<>();

        for (JsonNode junction : junctions) {
            String id = junction.path("id").asText("").trim();
            if (isValidJunctionId(id)) {
                usedIds.add(id);
            }
        }

        int nextJunctionId = 1;
        for (JsonNode junction : junctions) {
            if (!(junction instanceof ObjectNode junctionObject)) {
                continue;
            }

            hasChanges |= normalizeJunctionArrayField(junctionObject, "source", "sources");
            hasChanges |= normalizeJunctionArrayField(junctionObject, "target", "targets");

            String id = junctionObject.path("id").asText("").trim();
            if (id.isBlank()) {
                String replacementId = nextAvailableJunctionId(usedIds, nextJunctionId);
                nextJunctionId = Integer.parseInt(replacementId.replace("junction_", "")) + 1;
                junctionObject.put("id", replacementId);
                usedIds.add(replacementId);
                hasChanges = true;
                continue;
            }

            String normalizedId = normalizeJunctionId(id);
            if (!isValidJunctionId(id)) {
                String replacementId = normalizedId;
                if (!isValidJunctionId(replacementId) || usedIds.contains(replacementId)) {
                    replacementId = nextAvailableJunctionId(usedIds, nextJunctionId);
                    nextJunctionId = Integer.parseInt(replacementId.replace("junction_", "")) + 1;
                }

                junctionObject.put("id", replacementId);
                renameMap.put(id, replacementId);
                usedIds.add(replacementId);
                hasChanges = true;
            }
        }

        if (!renameMap.isEmpty()) {
            for (JsonNode junction : junctions) {
                if (junction instanceof ObjectNode junctionObject) {
                    hasChanges |= renameJunctionArrayReferences(junctionObject, renameMap, "sources");
                    hasChanges |= renameJunctionArrayReferences(junctionObject, renameMap, "targets");
                }
            }
        }

        return hasChanges;
    }

    private boolean isValidJunctionId(String id) {
        return id.matches("^junction_[0-9]+$");
    }

    private String normalizeJunctionId(String id) {
        String trimmedId = id.trim();
        if (trimmedId.matches("(?i)^(junction|juction|merge|split)_[0-9]+$")) {
            return "junction_" + trimmedId.substring(trimmedId.lastIndexOf('_') + 1);
        }
        return trimmedId;
    }

    private String nextAvailableJunctionId(Set<String> usedIds, int startAt) {
        int nextId = Math.max(1, startAt);
        String replacementId;
        do {
            replacementId = "junction_" + nextId++;
        } while (usedIds.contains(replacementId));
        return replacementId;
    }

    private boolean renameJunctionArrayReferences(ObjectNode junction, Map<String, String> renameMap, String arrayField) {
        JsonNode values = junction.get(arrayField);
        if (!(values instanceof ArrayNode arrayValues)) {
            return false;
        }

        boolean hasChanges = false;
        ArrayNode updatedValues = objectMapper.createArrayNode();
        for (JsonNode value : arrayValues) {
            String oldValue = value.asText();
            String newValue = renameMap.getOrDefault(oldValue, oldValue);
            updatedValues.add(newValue);
            hasChanges |= !oldValue.equals(newValue);
        }
        junction.set(arrayField, updatedValues);
        return hasChanges;
    }

    private boolean normalizeJunctionArrayField(ObjectNode junction, String singularField, String arrayField) {
        if (junction.has(arrayField) || !junction.has(singularField)) {
            return false;
        }

        JsonNode singularValue = junction.get(singularField);
        if (singularValue.isArray()) {
            junction.set(arrayField, singularValue);
        } else if (!singularValue.isNull()) {
            ArrayNode values = objectMapper.createArrayNode();
            values.add(singularValue.asText());
            junction.set(arrayField, values);
        } else {
            junction.putArray(arrayField);
        }
        junction.remove(singularField);
        return true;
    }

    /**
     * Validates that all junctions reference existing and valid components or other
     * junctions (Error Type 3). It checks if the source and target ports defined
     * within each junction correspond to components present in the 'connections'
     * list or to other defined junctions. If fixJunctions is true, it removes any
     * junctions with invalid references; otherwise, it adds a description of the
     * problem to the problems list for LLM re-prompting.
     */
    private boolean validateJunctionReferences(ObjectNode chipDesign, List<String> problems, boolean fixJunctions) {
        boolean hasChanges = false;

        Set<String> sourcesInConnections = new HashSet<>();
        Set<String> targetsInConnections = new HashSet<>();
        Set<String> junctionIds = new HashSet<>();

        // Collect components from connections
        for (JsonNode connection : chipDesign.get("connections")) {
            sourcesInConnections.add(connection.get("source").asText());
            targetsInConnections.add(connection.get("target").asText());
        }

        // Collect junction IDs
        for (JsonNode junction : chipDesign.get("junctions")) {
            junctionIds.add(junction.get("id").asText());
        }

        // Validate junction references
        Iterator<JsonNode> iterator = chipDesign.get("junctions").elements();
        Set<String> junctionsToRemove = new HashSet<>();

        while (iterator.hasNext()) {
            JsonNode junction = iterator.next();
            if (validateJunctionPorts(junction, sourcesInConnections, targetsInConnections, junctionIds, problems)) {
                if (fixJunctions) {
                    junctionsToRemove.add(junction.get("id").asText());
                    hasChanges = true;
                }
            }
        }

        // Remove problematic junctions
        if (fixJunctions && !junctionsToRemove.isEmpty()) {
            ArrayNode junctionsArray = (ArrayNode) chipDesign.get("junctions");
            ArrayNode updatedJunctions = chipDesign.putArray("junctions");

            for (JsonNode junction : junctionsArray) {
                if (!junctionsToRemove.contains(junction.get("id").asText())) {
                    updatedJunctions.add(junction);
                }
            }
        }

        return hasChanges;
    }

    /**
     * Validates the integrity and consistency of the junction network (Error Types
     * 4 & 5). This method performs two key checks:
     * 1. (Error Type 4) It ensures that junction-to-junction connections are
     * consistent (i.e., if junction_A lists junction_B as a target, junction_B must
     * list junction_A as a source).
     * 2. (Error Type 5) It verifies that component ports are not connected to
     * multiple junctions, which would violate the physical constraints of the
     * components.
     * If fixJunctions is true, problematic junctions are removed. Otherwise, issues
     * are added to the problems list.
     */
    private boolean validateJunctionConnectivity(ObjectNode chipDesign, List<String> problems, boolean fixJunctions) {
        boolean hasChanges = false;
        Map<String, Set<String>> junctionSources = new HashMap<>();
        Map<String, Set<String>> junctionTargets = new HashMap<>();
        Set<String> junctionsToRemove = new HashSet<>();
        Map<String, List<String>> sourceUsage = new HashMap<>();
        Map<String, List<String>> targetUsage = new HashMap<>();

        // Build junction connectivity map (v2: sources[] and targets[] arrays)
        for (JsonNode junction : chipDesign.get("junctions")) {
            String junctionId = junction.get("id").asText();
            Set<String> sources = new HashSet<>();
            Set<String> targets = new HashSet<>();

            for (JsonNode s : junction.get("sources")) {
                sources.add(s.asText());
            }
            for (JsonNode t : junction.get("targets")) {
                targets.add(t.asText());
            }

            int sourceCount = sources.size();
            int targetCount = targets.size();
            int totalPorts = sourceCount + targetCount;
            boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
            boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;

            if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
                problems.add("Junction " + junctionId +
                        " must be either a split (1 source, 2-5 targets) or a merge (2-5 sources, 1 target), with total ports between 3 and 6");
                if (fixJunctions) {
                    junctionsToRemove.add(junctionId);
                    hasChanges = true;
                }
                continue;
            }

            junctionSources.put(junctionId, sources);
            junctionTargets.put(junctionId, targets);
        }

        // Validate bidirectional connectivity
        for (Map.Entry<String, Set<String>> entry : junctionSources.entrySet()) {
            String junctionId = entry.getKey();
            Set<String> connections = entry.getValue();

            for (String connectedComponent : connections) {
                if (isJunction(connectedComponent)) {
                    Set<String> reverseConnections = junctionTargets.get(connectedComponent);
                    if (reverseConnections == null) {
                        continue;
                    }
                    if (!reverseConnections.contains(junctionId)) {
                        problems.add("Junction " + junctionId + " references " + connectedComponent +
                                " as source but " + connectedComponent + " does not reference " + junctionId
                                + " as target");
                        if (fixJunctions) {
                            junctionsToRemove.add(junctionId);
                            hasChanges = true;
                        }
                    }
                }
            }
        }
        for (Map.Entry<String, Set<String>> entry : junctionTargets.entrySet()) {
            String junctionId = entry.getKey();
            Set<String> connections = entry.getValue();

            for (String connectedComponent : connections) {
                if (isJunction(connectedComponent)) {
                    Set<String> reverseConnections = junctionSources.get(connectedComponent);
                    if (reverseConnections == null) {
                        continue;
                    }
                    if (!reverseConnections.contains(junctionId)) {
                        problems.add("Junction " + junctionId + " references " + connectedComponent +
                                " as target but " + connectedComponent + " does not reference " + junctionId
                                + " as source");
                        if (fixJunctions) {
                            junctionsToRemove.add(junctionId);
                            hasChanges = true;
                        }
                    }
                }
            }
        }

        // Check for multiple junctions using the same non-junction source/target
        for (Map.Entry<String, Set<String>> entry : junctionSources.entrySet()) {
            String junctionId = entry.getKey();
            for (String source : entry.getValue()) {
                if (!isJunction(source)) {
                    sourceUsage.putIfAbsent(source, new ArrayList<>());
                    sourceUsage.get(source).add(junctionId);
                }
            }
        }

        for (Map.Entry<String, Set<String>> entry : junctionTargets.entrySet()) {
            String junctionId = entry.getKey();
            for (String target : entry.getValue()) {
                if (!isJunction(target)) {
                    targetUsage.putIfAbsent(target, new ArrayList<>());
                    targetUsage.get(target).add(junctionId);
                }
            }
        }

        for (Map.Entry<String, List<String>> entry : sourceUsage.entrySet()) {
            if (entry.getValue().size() > 1) {
                problems.add("Junctions " + entry.getValue() + " use the same source " + entry.getKey()
                        + ". Additional junctions need to be added.");
                if (fixJunctions) {
                    junctionsToRemove.addAll(entry.getValue().subList(1, entry.getValue().size()));
                    hasChanges = true;
                }
            }
        }

        for (Map.Entry<String, List<String>> entry : targetUsage.entrySet()) {
            if (entry.getValue().size() > 1) {
                problems.add("Junctions " + entry.getValue() + " use the same target " + entry.getKey()
                        + ". Additional junctions need to be added.");
                if (fixJunctions) {
                    junctionsToRemove.addAll(entry.getValue().subList(1, entry.getValue().size()));
                    hasChanges = true;
                }
            }
        }

        // Remove problematic junctions
        if (fixJunctions && !junctionsToRemove.isEmpty()) {
            ArrayNode junctionsArray = (ArrayNode) chipDesign.get("junctions");
            ArrayNode updatedJunctions = chipDesign.putArray("junctions");

            for (JsonNode junction : junctionsArray) {
                if (!junctionsToRemove.contains(junction.get("id").asText())) {
                    updatedJunctions.add(junction);
                }
            }
        }

        return hasChanges;
    }

    /**
     * Validates the logical flow paths created by the junction network against the
     * conceptual connections (Error Types 6 & 7). This method builds a directed
     * graph from the entire design and checks for two types of errors:
     * 1. (Error Type 6) Extraneous paths: It identifies flow paths enabled by the
     * junction network that were not specified in the conceptual 'connections'
     * list.
     * 2. (Error Type 7) Missing paths: It identifies required conceptual
     * connections that the junction network fails to implement.
     * If fixJunctions is true, it attempts to algorithmically correct these issues;
     * otherwise, it reports them as problems.
     */
    private boolean validateJunctionFlows(ObjectNode chipDesign, List<String> problems, List<String> addedJunctions,
            boolean fixJunctions) {
        boolean hasChanges = false;

        // Build a directed graph representing the flow network
        Graph<String, DefaultEdge> flowGraph = new DirectedMultigraph<>(DefaultEdge.class);

        // Add vertices and edges from connections
        for (JsonNode connection : chipDesign.get("connections")) {
            String source = connection.get("source").asText();
            String target = connection.get("target").asText();

            flowGraph.addVertex(source);
            flowGraph.addVertex(target);
            flowGraph.addEdge(source, target);
        }

        // Remove edges that are replaced by junctions
        Set<String> verticesToRemoveOutgoing = new HashSet<>();
        Set<String> verticesToRemoveIncoming = new HashSet<>();

        // Phase 1: Determine which vertices exceed edge limits
        for (String vertex : flowGraph.vertexSet()) {
            if (flowGraph.outgoingEdgesOf(vertex).size() > 1) {
                verticesToRemoveOutgoing.add(vertex);
            }
            if (flowGraph.incomingEdgesOf(vertex).size() > 1) {
                verticesToRemoveIncoming.add(vertex);
            }
        }

        // Phase 2: Remove the edges
        for (String vertex : verticesToRemoveOutgoing) {
            Set<DefaultEdge> outgoingEdges = new HashSet<>(flowGraph.outgoingEdgesOf(vertex));
            for (DefaultEdge edge : outgoingEdges) {
                flowGraph.removeEdge(edge);
            }
        }

        for (String vertex : verticesToRemoveIncoming) {
            Set<DefaultEdge> incomingEdges = new HashSet<>(flowGraph.incomingEdgesOf(vertex));
            for (DefaultEdge edge : incomingEdges) {
                flowGraph.removeEdge(edge);
            }
        }

        // Add vertices and edges from junctions
        for (JsonNode junction : chipDesign.get("junctions")) {
            String junctionId = junction.get("id").asText();
            JsonNode sources = junction.get("sources");
            JsonNode targets = junction.get("targets");
            if (sources == null || !sources.isArray() || targets == null || !targets.isArray()) {
                continue;
            }
            int sourceCount = sources.size();
            int targetCount = targets.size();
            int totalPorts = sourceCount + targetCount;
            boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
            boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;
            if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
                continue;
            }
            flowGraph.addVertex(junctionId);
        }
        for (JsonNode junction : chipDesign.get("junctions")) {
            String junctionId = junction.get("id").asText();
            JsonNode sources = junction.get("sources");
            JsonNode targets = junction.get("targets");
            if (sources == null || !sources.isArray() || targets == null || !targets.isArray()) {
                continue;
            }
            int sourceCount = sources.size();
            int targetCount = targets.size();
            int totalPorts = sourceCount + targetCount;
            boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
            boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;
            if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
                continue;
            }

            // v2: iterate sources[] and targets[] arrays
            for (JsonNode s : sources) {
                String src = s.asText();
                if (!flowGraph.containsEdge(src, junctionId)) {
                    flowGraph.addEdge(src, junctionId);
                }
            }
            for (JsonNode t : targets) {
                String tgt = t.asText();
                if (!flowGraph.containsEdge(junctionId, tgt)) {
                    flowGraph.addEdge(junctionId, tgt);
                }
            }
        }

        AllDirectedPaths<String, DefaultEdge> allPaths = new AllDirectedPaths<>(flowGraph);
        Set<String> connections = new HashSet<>();
        Set<Pair<String, String>> missingPaths = new HashSet<>();
        Set<GraphPath<String, DefaultEdge>> invalidPaths = new HashSet<>();

        // Validate that all connections are possible through the junction network
        for (JsonNode connection : chipDesign.get("connections")) {
            String source = connection.get("source").asText();
            String target = connection.get("target").asText();

            if (!hasPath(allPaths, source, target)) {
                problems.add("No valid flow path found from " + source + " to " + target +
                        " through the junction network");
                if (fixJunctions) {
                    missingPaths.add(new Pair<>(source, target));
                }
            }

            connections.add(source + "->" + target);
        }

        // Validate that there are no extra paths that are not part of the connections

        Set<String> reportedPairs = new HashSet<>(); // Track reported source/target pairs

        for (String source : flowGraph.vertexSet()) {
            if (isJunction(source))
                continue; // Skip junction nodes

            for (String target : flowGraph.vertexSet()) {
                if (isJunction(target))
                    continue; // Skip junction nodes

                // Check if the path contains at least one junction node
                List<GraphPath<String, DefaultEdge>> paths = allPaths.getAllPaths(source, target, true, null);
                for (GraphPath<String, DefaultEdge> path : paths) {

                    // Convert the path to a list of vertexes and check if all vertexes inbetween
                    // are junctions
                    List<String> pathVertices = path.getVertexList();

                    // Check if there are intermediate Nodes
                    if (pathVertices.size() <= 2) {
                        break;
                    }

                    boolean allIntermediateAreJunctions = true;
                    for (int i = 1; i < pathVertices.size() - 1; i++) {
                        if (!isJunction(pathVertices.get(i))) {
                            allIntermediateAreJunctions = false;
                            break;
                        }
                    }

                    if (allIntermediateAreJunctions) {
                        // Check whether the path is already part of the specified connections
                        String connectionKey = source + "->" + target;
                        if (!connections.contains(connectionKey)) {
                            if (!reportedPairs.contains(connectionKey)) {
                                problems.add("A path was detected from " + source + " to " + target +
                                        " through junction(s) that is not specified in the connections list");
                            }
                            reportedPairs.add(connectionKey); // Mark as reported
                            if (fixJunctions) {
                                invalidPaths.add(path);
                            }
                        }
                    }
                }
            }
        }

        // If fixJunctions is true, apply the fixes
        if (fixJunctions) {
            hasChanges |= removeInvalidPaths(chipDesign, flowGraph, invalidPaths);
            hasChanges |= enableMissingPaths(chipDesign, flowGraph, missingPaths, addedJunctions);
            if (hasChanges) {
                problems.clear();
                validateJunctionFlows(chipDesign, problems, addedJunctions, true);
            }
        }

        return hasChanges;

    }

    /**
     * Helper function to determine if a given component ID string represents a
     * junction.
     */
    private boolean isJunction(String node) {
        return node.startsWith("junction_");
    }

    /**
     * Checks if at least one valid path exists between a source and a target node
     * in the flow graph.
     */
    public boolean hasPath(AllDirectedPaths<String, DefaultEdge> allPaths, String source, String target) {

        // Use AllDirectedPaths to find all paths between source and target
        List<GraphPath<String, DefaultEdge>> paths = allPaths.getAllPaths(Set.of(source), Set.of(target), true, null);

        // Return true if at least one path exists, otherwise false
        return !paths.isEmpty();
    }

    /**
     * A helper method to validate the source and target ports of a single junction.
     * It checks if the components referenced by a junction's ports exist either in
     * the main 'connections' list or as another defined junction.
     */
    private boolean validateJunctionPorts(JsonNode junction, Set<String> validSources, Set<String> validTargets,
            Set<String> junctionIds, List<String> problems) {
        boolean foundProblems = false;

        List<String> sources = new ArrayList<>();
        List<String> targets = new ArrayList<>();

        if (!junction.has("sources") || !junction.get("sources").isArray()
                || !junction.has("targets") || !junction.get("targets").isArray()) {
            problems.add("Junction " + junction.get("id").asText() +
                    " must contain sources[] and targets[] arrays");
            return true;
        }

        // v2: read from sources[] and targets[] arrays
        for (JsonNode s : junction.get("sources")) {
            sources.add(s.asText());
        }
        for (JsonNode t : junction.get("targets")) {
            targets.add(t.asText());
        }

        int sourceCount = sources.size();
        int targetCount = targets.size();
        int totalPorts = sourceCount + targetCount;
        boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
        boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;

        if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
            problems.add("Junction " + junction.get("id").asText() +
                    " must be either a split (1 source, 2-5 targets) or a merge (2-5 sources, 1 target), with total ports between 3 and 6");
            foundProblems = true;
        }

        for (String port : sources) {
            if (!validSources.contains(port) && !junctionIds.contains(port)) {
                problems.add("Junction " + junction.get("id").asText() +
                        " references undefined source: " + port);
                foundProblems = true;
            }
        }

        for (String port : targets) {
            if (!validTargets.contains(port) && !junctionIds.contains(port)) {
                problems.add("Junction " + junction.get("id").asText() +
                        " references undefined target: " + port);
                foundProblems = true;
            }
        }

        return foundProblems;
    }

    /**
     * A helper method to manage the parameters for a specific category of
     * components (e.g., "mixers"). It ensures that the 'component_params' section
     * only contains entries for components that are actually used in the
     * 'connections' list. It also adds entries with default parameter values for
     * any used components that are missing from the parameters section.
     */
    private boolean updateComponentParams(ObjectNode componentParams, String componentType,
            Set<String> usedComponents, String prefix) {
        boolean hasChanges = false;
        Set<String> existingComponents = new HashSet<>();
        ArrayNode componentsArray = (ArrayNode) componentParams.get(componentType);

        // Collect existing components
        Iterator<JsonNode> iterator = componentsArray.iterator();
        while (iterator.hasNext()) {
            JsonNode component = iterator.next();
            String componentId = component.get("id").asText();
            existingComponents.add(componentId);

            // Remove components not in usedComponents
            if (!usedComponents.contains(componentId)) {
                iterator.remove();
                hasChanges = true;
            }
        }

        // Add missing components with default parameters
        for (String component : usedComponents) {
            if (component.startsWith(prefix) && !existingComponents.contains(component)) {
                ObjectNode newComponent = createDefaultComponentParams(component, componentType);
                if (newComponent != null) {
                    componentsArray.add(newComponent);
                    hasChanges = true;
                }
            }
        }

        return hasChanges;
    }

    /**
     * Creates a JSON object for a component with its default parameter values.
     */
    private ObjectNode createDefaultComponentParams(String componentId, String componentType) {
        ObjectNode newComponent = objectMapper.createObjectNode();
        newComponent.put("id", componentId);

        switch (componentType) {
            case "mixers":
                newComponent.put("type", "serpentine");
                newComponent.put("num_turnings", 4);
                newComponent.put("amplitude_um", 2000);
                newComponent.put("distance_between_turnings_um", 200);
                break;
            case "delays":
                newComponent.put("type", "serpentine");
                newComponent.put("num_turnings", 4);
                newComponent.put("amplitude_um", 2000);
                newComponent.put("distance_between_turnings_um", 200);
                break;
            case "chambers":
                newComponent.put("length_um", 4000);
                newComponent.put("width_um", 3200);
                break;
            case "filters":
                newComponent.put("type", "pillar_matrix");
                newComponent.put("length_um", 4000);
                newComponent.put("width_um", 3200);
                newComponent.put("post_shape", "circle");
                newComponent.put("post_diameter_um", 200);
                newComponent.put("columns", 6);
                newComponent.put("rows", 7);
                break;
            case "droplets":
                newComponent.put("type", "t_junction");
                newComponent.put("nozzle_width_um", 100);
                break;
            case "tesla_valves":
                newComponent.put("num_segment_pairs", 2);
                newComponent.put("segment_length_um", 1000);
                newComponent.put("segment_width_um", 600);
                break;
            default:
                return null;
        }

        return newComponent;
    }

    /**
     * Calculates the next available sequential ID for a given component prefix. It
     * scans a set of existing component IDs to find the highest number used for a
     * given prefix (e.g., "inlet_") and returns the next integer.
     */
    private int getNextComponentId(Set<String> existingComponents, String prefix) {
        int maxId = 0;
        for (String component : existingComponents) {
            if (component.startsWith(prefix)) {
                try {
                    int id = Integer.parseInt(component.substring(prefix.length()));
                    maxId = Math.max(maxId, id);
                } catch (NumberFormatException e) {
                    // Skip invalid component IDs
                }
            }
        }
        return maxId + 1;
    }

    /**
     * Implements the algorithmic correction for extraneous paths (Error Type 6).
     * This method traverses identified invalid paths within the junction network
     * and removes the specific edge that creates the unwanted connection, aiming to
     * do so with minimal disruption to other valid paths. It also prunes any
     * junctions that become redundant.
     */
    private boolean removeInvalidPaths(ObjectNode chipDesign, Graph<String, DefaultEdge> flowGraph,
            Set<GraphPath<String, DefaultEdge>> invalidPaths) {
        boolean hasChanges = false;

        // Step 1: Traverse each invalid path
        for (GraphPath<String, DefaultEdge> path : invalidPaths) {
            List<String> pathNodes = path.getVertexList();
            String target = pathNodes.get(pathNodes.size() - 1);

            // Traverse through the path and find the removable edge
            for (int i = 1; i < pathNodes.size(); i++) {
                String current = pathNodes.get(i);
                String previous = pathNodes.get(i - 1);

                // Find all reachable nodes from 'current' in the whole graph
                Set<String> reachable = getReachableNodes(flowGraph, current);

                // Break if path is already broken
                if (!reachable.contains(target))
                    break;

                // If only the target is reachable or we reached the target, remove the edge
                if (reachable.size() == 1 && reachable.contains(target)) {
                    if (flowGraph.containsEdge(previous, current)) {
                        flowGraph.removeEdge(previous, current);
                        hasChanges = true;
                    }
                    break; // Stop further traversal in this path
                }
            }
        }

        // Step 2: Remove unnecessary junctions (those with only or less than one input
        // and one output)
        Set<String> removableJunctions = new HashSet<>();

        for (String node : new HashSet<>(flowGraph.vertexSet())) {
            if (isJunction(node) && flowGraph.inDegreeOf(node) <= 1 && flowGraph.outDegreeOf(node) <= 1) {
                removableJunctions.add(node);
            }
        }

        for (String junction : removableJunctions) {
            // Get the single input and output nodes
            String inputNode = flowGraph.getEdgeSource(flowGraph.incomingEdgesOf(junction).iterator().next());
            String outputNode = flowGraph.getEdgeTarget(flowGraph.outgoingEdgesOf(junction).iterator().next());

            // Remove the junction and reconnect input to output
            flowGraph.removeVertex(junction);
            flowGraph.addEdge(inputNode, outputNode);

            hasChanges = true;
        }

        // Step 3: Update the JSON structure
        if (hasChanges) {
            updateJsonWithGraph(chipDesign, flowGraph);
        }

        return hasChanges;
    }

    /**
     * Implements the algorithmic correction for missing paths (Error Type 7). For
     * each required conceptual connection that is not implemented by the junction
     * network, this method attempts to minimally augment the existing network to
     * establish the connection. It uses heuristics to find optimal divergence and
     * convergence points to add new connections and, if necessary, new binary
     * Y-junctions.
     */
    private boolean enableMissingPaths(ObjectNode chipDesign, Graph<String, DefaultEdge> flowGraph,
            Set<Pair<String, String>> missingPaths, List<String> addedJunctions) {
        boolean hasChanges = false;

        List<Pair<String, String>> shuffledPaths = new ArrayList<>(missingPaths);
        Collections.shuffle(shuffledPaths);

        // Step 1: Traverse each missing path
        for (Pair<String, String> missingPath : shuffledPaths) {

            String source = missingPath.getKey();
            String target = missingPath.getValue();

            // Skip if target is already reachable from source
            Set<String> reachableFromSource = new HashSet<>();

            for (DefaultEdge edge : flowGraph.outgoingEdgesOf(source)) {
                String outgoingNode = flowGraph.getEdgeTarget(edge);
                reachableFromSource.addAll(getReachableNodes(flowGraph, outgoingNode));
            }

            if (reachableFromSource.contains(target)) {
                continue; // Path is already enabled
            }

            // Step 2: Find all sources that need to reach this target
            Set<String> sourcesWithSameTarget = new HashSet<>();
            for (Pair<String, String> otherPath : missingPaths) {
                if (otherPath.getValue().equals(target)) {
                    sourcesWithSameTarget.add(otherPath.getKey());
                }
            }

            // Step 3: Find the best junction for the source
            String bestSourceNode = source;
            Queue<String> queue = new LinkedList<>();
            Set<String> visited = new HashSet<>();
            Set<String> coveredSources = new HashSet<>(Collections.singleton(source));
            queue.add(source);
            visited.add(source);

            while (!queue.isEmpty()) {
                String current = queue.poll();

                for (DefaultEdge edge : flowGraph.outgoingEdgesOf(current)) {
                    String nextNode = flowGraph.getEdgeTarget(edge);

                    // Ensure that the next node is a junction
                    if (isJunction(nextNode) && !visited.contains(nextNode)) {
                        visited.add(nextNode);

                        Set<String> reachableSources = getReachableSources(flowGraph, nextNode);
                        if (sourcesWithSameTarget.containsAll(reachableSources)) {
                            if (reachableSources.size() >= coveredSources.size()) {
                                bestSourceNode = nextNode;
                                coveredSources = new HashSet<>(reachableSources);
                            }
                            queue.add(nextNode);
                        }
                    }
                }
            }

            // Step 4: Find all targets in missingPaths for which a pair exists for all
            // sources in coveredSources
            Set<String> targetsWithAllSources = new HashSet<>();
            for (Pair<String, String> otherPath : missingPaths) {
                String potentialTarget = otherPath.getValue();
                boolean allSourcesCovered = true;

                for (String coveredSource : coveredSources) {
                    if (!missingPaths.contains(new Pair<>(coveredSource, potentialTarget))) {
                        allSourcesCovered = false;
                        break;
                    }
                }

                if (allSourcesCovered) {
                    targetsWithAllSources.add(potentialTarget);
                }
            }

            // Step 5: Find the best junction for the target
            String bestTargetNode = target;
            queue = new LinkedList<>();
            visited = new HashSet<>();
            Set<String> coveredTargets = new HashSet<>(Collections.singleton(target));
            queue.add(target);
            visited.add(target);

            while (!queue.isEmpty()) {
                String current = queue.poll();

                for (DefaultEdge edge : flowGraph.incomingEdgesOf(current)) {
                    String prevNode = flowGraph.getEdgeSource(edge);

                    // Ensure that the prev node is a junction
                    if (isJunction(prevNode) && !visited.contains(prevNode)) {
                        visited.add(prevNode);

                        Set<String> reachableTargets = getReachableNodes(flowGraph, prevNode);
                        if (targetsWithAllSources.containsAll(reachableTargets)) {
                            if (reachableTargets.size() >= coveredTargets.size()) {
                                bestTargetNode = prevNode;
                                coveredTargets = new HashSet<>(reachableTargets);
                            }
                            queue.add(prevNode);
                        }
                    }
                }
            }

            // Step 6: Add an edge between the best source and target junctions if found
            if (bestSourceNode != null && bestTargetNode != null) {
                // Function to create a new junction node
                Function<String, String> createNewJunction = (prefix) -> {
                    int maxJunctionId = flowGraph.vertexSet().stream()
                            .filter(node -> node.startsWith(prefix))
                            .mapToInt(node -> Integer.parseInt(node.replace(prefix, "")))
                            .max()
                            .orElse(0) + 1;
                    return prefix + maxJunctionId;
                };

                // Check and modify bestSourceNode if necessary
                if ((flowGraph.incomingEdgesOf(bestSourceNode).size() == 2
                        && flowGraph.outgoingEdgesOf(bestSourceNode).size() >= 1) ||
                        (flowGraph.incomingEdgesOf(bestSourceNode).size() == 1
                                && flowGraph.outgoingEdgesOf(bestSourceNode).size() == 2)
                        ||
                        (!isJunction(bestSourceNode) && flowGraph.outgoingEdgesOf(bestSourceNode).size() >= 1)) {

                    String newSourceJunction = createNewJunction.apply("junction_");
                    flowGraph.addVertex(newSourceJunction);
                    addedJunctions.add(newSourceJunction);

                    Set<DefaultEdge> outgoingEdges = flowGraph.outgoingEdgesOf(bestSourceNode);
                    String outgoingEdgeTarget = flowGraph.getEdgeTarget(new ArrayList<>(outgoingEdges).get(0));
                    flowGraph.removeEdge(bestSourceNode, outgoingEdgeTarget);
                    flowGraph.addEdge(newSourceJunction, outgoingEdgeTarget);
                    flowGraph.addEdge(bestSourceNode, newSourceJunction);

                    bestSourceNode = newSourceJunction;
                }

                // Check and modify bestTargetNode if necessary
                if ((flowGraph.outgoingEdgesOf(bestTargetNode).size() == 2
                        && flowGraph.incomingEdgesOf(bestTargetNode).size() >= 1) ||
                        (flowGraph.outgoingEdgesOf(bestTargetNode).size() == 1
                                && flowGraph.incomingEdgesOf(bestTargetNode).size() == 2)
                        ||
                        (!isJunction(bestTargetNode) && flowGraph.incomingEdgesOf(bestTargetNode).size() >= 1)) {

                    String newTargetJunction = createNewJunction.apply("junction_");
                    flowGraph.addVertex(newTargetJunction);
                    addedJunctions.add(newTargetJunction);

                    Set<DefaultEdge> incomingEdges = flowGraph.incomingEdgesOf(bestTargetNode);
                    String incomingEdgeSource = flowGraph.getEdgeSource(new ArrayList<>(incomingEdges).get(0));
                    flowGraph.removeEdge(incomingEdgeSource, bestTargetNode);
                    flowGraph.addEdge(incomingEdgeSource, newTargetJunction);
                    flowGraph.addEdge(newTargetJunction, bestTargetNode);

                    bestTargetNode = newTargetJunction;
                }

                // Add the new edge
                if (isJunction(bestSourceNode)) {
                    int sourceIncoming = flowGraph.incomingEdgesOf(bestSourceNode).size();
                    int sourceOutgoing = flowGraph.outgoingEdgesOf(bestSourceNode).size() + 1;
                    int sourceTotalPorts = sourceIncoming + sourceOutgoing;
                    boolean validSourceSplit = sourceIncoming == 1 && sourceOutgoing >= 2 && sourceOutgoing <= 5;
                    boolean validSourceMerge = sourceIncoming >= 2 && sourceIncoming <= 5 && sourceOutgoing == 1;
                    if (sourceTotalPorts < 3 || sourceTotalPorts > 6 || (!validSourceSplit && !validSourceMerge)) {
                        continue;
                    }
                }

                if (isJunction(bestTargetNode)) {
                    int targetIncoming = flowGraph.incomingEdgesOf(bestTargetNode).size() + 1;
                    int targetOutgoing = flowGraph.outgoingEdgesOf(bestTargetNode).size();
                    int targetTotalPorts = targetIncoming + targetOutgoing;
                    boolean validTargetSplit = targetIncoming == 1 && targetOutgoing >= 2 && targetOutgoing <= 5;
                    boolean validTargetMerge = targetIncoming >= 2 && targetIncoming <= 5 && targetOutgoing == 1;
                    if (targetTotalPorts < 3 || targetTotalPorts > 6 || (!validTargetSplit && !validTargetMerge)) {
                        continue;
                    }
                }

                flowGraph.addEdge(bestSourceNode, bestTargetNode);
                hasChanges = true;
            }
        }

        // Step 7: Update the JSON structure if changes were made
        if (hasChanges) {
            updateJsonWithGraph(chipDesign, flowGraph);
        }

        return hasChanges;
    }

    /**
     * Finds all reachable nodes from a given node in the graph,
     * considering only paths that pass exclusively through junctions.
     */
    private Set<String> getReachableNodes(Graph<String, DefaultEdge> graph, String startNode) {
        Set<String> reachable = new HashSet<>();
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        if (isJunction(startNode)) {
            queue.add(startNode);
        } else {
            reachable.add(startNode);
        }

        while (!queue.isEmpty()) {
            String current = queue.poll();

            for (DefaultEdge edge : graph.outgoingEdgesOf(current)) {
                String nextNode = graph.getEdgeTarget(edge);

                // Ensure that the next node is a junction
                if (isJunction(nextNode) && !visited.contains(nextNode)) {
                    visited.add(nextNode);
                    queue.add(nextNode);
                }
                // Stop at a non-junction (valid endpoint)
                else if (!isJunction(nextNode)) {
                    visited.add(nextNode);
                    reachable.add(nextNode);
                }
            }
        }

        return reachable;
    }

    /**
     * Finds all nodes from which the given node can be reached in the graph,
     * considering only paths that pass exclusively through junctions.
     */
    private Set<String> getReachableSources(Graph<String, DefaultEdge> graph, String targetNode) {
        Set<String> reachable = new HashSet<>();
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        if (isJunction(targetNode)) {
            queue.add(targetNode);
        } else {
            reachable.add(targetNode);
        }

        while (!queue.isEmpty()) {
            String current = queue.poll();

            for (DefaultEdge edge : graph.incomingEdgesOf(current)) {
                String previousNode = graph.getEdgeSource(edge);

                // Ensure that the previous node is a junction
                if (isJunction(previousNode) && !visited.contains(previousNode)) {
                    visited.add(previousNode);
                    queue.add(previousNode);
                }
                // Stop at a non-junction (valid source)
                else if (!isJunction(previousNode)) {
                    visited.add(previousNode);
                    reachable.add(previousNode);
                }
            }
        }

        return reachable;
    }

    /**
     * Updates the junctions section in the JSON based on the current state of the
     * graph.
     * - Removes junctions that no longer exist in the graph.
     * - Updates only the changed edges while keeping the original order of
     * sources/targets.
     * - Adds new junctions if they exist in the graph but were missing in the JSON.
     */
    private void updateJsonWithGraph(ObjectNode chipDesign, Graph<String, DefaultEdge> graph) {
        ArrayNode junctions = (ArrayNode) chipDesign.get("junctions");
        ArrayNode updatedJunctions = chipDesign.putArray("junctions");

        for (JsonNode junction : junctions) {
            String junctionId = junction.get("id").asText();

            // Skip junctions that no longer exist in the graph
            if (!graph.containsVertex(junctionId)) {
                continue;
            }

            ObjectNode updatedJunction = (ObjectNode) junction.deepCopy();

            // v2: Get existing sources and targets from JSON arrays
            List<String> sources = new ArrayList<>();
            List<String> targets = new ArrayList<>();
            JsonNode existingSources = junction.get("sources");
            JsonNode existingTargets = junction.get("targets");
            if (existingSources == null || !existingSources.isArray()
                    || existingTargets == null || !existingTargets.isArray()) {
                continue;
            }
            for (JsonNode s : existingSources) sources.add(s.asText());
            for (JsonNode t : existingTargets) targets.add(t.asText());

            // Get actual edges from the graph
            List<String> newSources = graph.incomingEdgesOf(junctionId).stream()
                    .map(graph::getEdgeSource)
                    .collect(Collectors.toList());

            List<String> newTargets = graph.outgoingEdgesOf(junctionId).stream()
                    .map(graph::getEdgeTarget)
                    .collect(Collectors.toList());

            // v2: Rebuild sources[] and targets[] arrays from graph state
            ArrayNode sourcesArr = objectMapper.createArrayNode();
            ArrayNode targetsArr = objectMapper.createArrayNode();

            // Preserve original order where possible, substitute removed entries
            List<String> updatedSources = new ArrayList<>(sources);
            for (int i = 0; i < updatedSources.size(); i++) {
                if (!newSources.contains(updatedSources.get(i))) {
                    for (String candidate : newSources) {
                        if (!updatedSources.contains(candidate)) {
                            updatedSources.set(i, candidate);
                            break;
                        }
                    }
                }
            }
            // Add any brand-new sources not in original list
            for (String ns : newSources) {
                if (!updatedSources.contains(ns)) updatedSources.add(ns);
            }
            updatedSources.removeIf(s -> !newSources.contains(s));

            List<String> updatedTargets = new ArrayList<>(targets);
            for (int i = 0; i < updatedTargets.size(); i++) {
                if (!newTargets.contains(updatedTargets.get(i))) {
                    for (String candidate : newTargets) {
                        if (!updatedTargets.contains(candidate)) {
                            updatedTargets.set(i, candidate);
                            break;
                        }
                    }
                }
            }
            for (String nt : newTargets) {
                if (!updatedTargets.contains(nt)) updatedTargets.add(nt);
            }
            updatedTargets.removeIf(t -> !newTargets.contains(t));

            int sourceCount = updatedSources.size();
            int targetCount = updatedTargets.size();
            int totalPorts = sourceCount + targetCount;
            boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
            boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;
            if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
                continue;
            }

            updatedSources.forEach(sourcesArr::add);
            updatedTargets.forEach(targetsArr::add);

            updatedJunction.set("sources", sourcesArr);
            updatedJunction.set("targets", targetsArr);

            updatedJunctions.add(updatedJunction);
        }

        // Add missing junctions from the graph
        for (String vertex : graph.vertexSet()) {
            if (!isJunction(vertex)) {
                continue;
            }

            boolean existsInJson = StreamSupport.stream(junctions.spliterator(), false)
                    .anyMatch(j -> j.get("id").asText().equals(vertex));

            if (!existsInJson) {
                ObjectNode newJunction = chipDesign.objectNode();
                newJunction.put("id", vertex);
                newJunction.put("type", "Y-junction");

                List<String> newSources = graph.incomingEdgesOf(vertex).stream()
                        .map(graph::getEdgeSource)
                        .collect(Collectors.toList());
                List<String> newTargets = graph.outgoingEdgesOf(vertex).stream()
                        .map(graph::getEdgeTarget)
                        .collect(Collectors.toList());
                int sourceCount = newSources.size();
                int targetCount = newTargets.size();
                int totalPorts = sourceCount + targetCount;
                boolean validSplit = sourceCount == 1 && targetCount >= 2 && targetCount <= 5;
                boolean validMerge = sourceCount >= 2 && sourceCount <= 5 && targetCount == 1;
                if (totalPorts < 3 || totalPorts > 6 || (!validSplit && !validMerge)) {
                    continue;
                }

                // v2: build sources[] and targets[] arrays
                ArrayNode sourcesArr = objectMapper.createArrayNode();
                ArrayNode targetsArr = objectMapper.createArrayNode();
                newSources.forEach(sourcesArr::add);
                newTargets.forEach(targetsArr::add);
                newJunction.set("sources", sourcesArr);
                newJunction.set("targets", targetsArr);

                updatedJunctions.add(newJunction);
            }
        }
    }

    /**
     * Renumbers all junction IDs to be sequential, resolving any inconsistencies
     * that may have arisen during the validation and correction process. This
     * ensures that junction IDs in the final design are clean and consecutively
     * numbered (e.g., junction_1, junction_2, ...). It also updates all references
     * to these junctions throughout the design.
     */
    private void updateJunctionNames(ObjectNode chipDesign, List<String> addedJunctions) {
        ArrayNode junctions = (ArrayNode) chipDesign.get("junctions");
        if (junctions == null) {
            return;
        }

        // Collect and sort existing junction IDs
        List<String> junctionIds = new ArrayList<>();
        for (JsonNode junction : junctions) {
            junctionIds.add(junction.get("id").asText());
        }
        Collections.sort(junctionIds, Comparator.comparingInt(j -> Integer.parseInt(j.replace("junction_", ""))));

        // Map old names to new sequential names
        Map<String, String> renameMap = new HashMap<>();
        for (int i = 0; i < junctionIds.size(); i++) {
            String oldName = junctionIds.get(i);
            String newName = "junction_" + (i + 1);
            renameMap.put(oldName, newName);
        }

        // Update addedJunctions list
        for (int i = 0; i < addedJunctions.size(); i++) {
            String oldName = addedJunctions.get(i);
            if (renameMap.containsKey(oldName)) {
                addedJunctions.set(i, renameMap.get(oldName));
            }
        }

        // Apply renaming
        ArrayNode updatedJunctions = chipDesign.putArray("junctions");
        for (JsonNode junction : junctions) {
            ObjectNode updatedJunction = (ObjectNode) junction.deepCopy();
            String oldId = updatedJunction.get("id").asText();
            updatedJunction.put("id", renameMap.get(oldId));

            // v2: Update references inside sources[] and targets[] arrays
            for (String key : Arrays.asList("sources", "targets")) {
                if (updatedJunction.has(key)) {
                    ArrayNode arr = (ArrayNode) updatedJunction.get(key);
                    ArrayNode updatedArr = objectMapper.createArrayNode();
                    for (JsonNode item : arr) {
                        String ref = item.asText();
                        updatedArr.add(renameMap.getOrDefault(ref, ref));
                    }
                    updatedJunction.set(key, updatedArr);
                }
            }
            updatedJunctions.add(updatedJunction);
        }
    }

    // Helper class for validation response
    public static class ValidationResponse {
        private final JsonNode correctedChipDesign;
        private final List<String> problems;
        private final List<String> addedJunctions;

        public ValidationResponse(JsonNode correctedChipDesign, List<String> problems, List<String> addedJunctions) {
            this.correctedChipDesign = correctedChipDesign;
            this.problems = problems;
            this.addedJunctions = addedJunctions;
        }

        public JsonNode getCorrectedChipDesign() {
            return correctedChipDesign;
        }

        public List<String> getProblems() {
            return problems;
        }

        public List<String> getAddedJunctions() {
            return addedJunctions;
        }
    }
}
