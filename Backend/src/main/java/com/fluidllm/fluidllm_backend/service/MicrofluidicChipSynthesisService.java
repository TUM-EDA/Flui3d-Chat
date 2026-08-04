package com.fluidllm.fluidllm_backend.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Value;

import reactor.core.publisher.Mono;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import com.threedmf.designsynthesis.chip.Chip;
import com.threedmf.designsynthesis.comsol.ComsolJavaGenerator3D;
import com.threedmf.designsynthesis.connection.Connection;
import com.threedmf.designsynthesis.module.*;
import com.threedmf.designsynthesis.util.Orientation;

import org.jfree.svg.ViewBox;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.time.Duration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

@Service
public class MicrofluidicChipSynthesisService {
    private static final int SOLVED_CHIP_CACHE_MAX_ENTRIES = 8;

    private final ObjectMapper objectMapper = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);
    private final Map<String, Chip> solvedChipCache = new LinkedHashMap<>(16, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, Chip> eldest) {
            return size() > SOLVED_CHIP_CACHE_MAX_ENTRIES;
        }
    };

    @Value("${desyn.stl.timeout}")
    private int stlTimeout;

    @Value("${desyn.gurobi.focus}")
    private int mipFocus;

    @Value("${desyn.gurobi.gap}")
    private double mipGap;

    /**
     * Handles the end-to-end synthesis of a microfluidic chip design. This method
     * contains the entire synthesis process by taking the validated JSON
     * representation of the chip and user-defined settings. It uses the 3M-DeSyn
     * library to instantiate components, solve the physical layout and routing as a
     * constraint optimization problem with Gurobi, and generate a visual
     * representation. The method returns both an SVG visualization of the chip
     * layout and, in a separate asynchronous step, a Base64-encoded STL file for 3D
     * printing, with a configurable timeout.
     */
    public Mono<SynthesisResponse> synthesizeChipDesign(JsonNode chipDesign, JsonNode settings) {
        return Mono.fromCallable(() -> {
            Chip chip = buildAndCacheSolvedChip(chipDesign, settings);
            String chipSvg = chip.drawSVG(1).getSVGElement(null, false,
                    new ViewBox(0, 0, chip.getWidth(), chip.getLength()), null, null);
            String chipJson = objectMapper.writeValueAsString(chip.generatePhysicalChip());
            return new SolvedChipView(chip, chipSvg, chipJson);
        })
                /*
                .flatMap(view -> {
                    // Async STL generation with timeout
                    CompletableFuture<String> stlFuture = CompletableFuture.supplyAsync(() -> {
                        try {
                            byte[] stlBytes = view.chip().generateSTL().stlToBytes();
                            return Base64.getEncoder().encodeToString(stlBytes);
                        } catch (Exception e) {
                            return null;
                        }
                    });

                    return Mono.fromFuture(stlFuture)
                            .timeout(Duration.ofSeconds(stlTimeout))
                            .onErrorResume(e -> Mono.empty())
                            .map(stlBase64 -> new SynthesisResponse(view.svg(), stlBase64, view.chipJson()))
                            .defaultIfEmpty(new SynthesisResponse(view.svg(), null, view.chipJson()));
                })
                */
                // STL is generated in the frontend from chipJson using OCCT.
                .map(view -> new SynthesisResponse(view.svg(), null, view.chipJson()));
    }

    public Mono<ComsolResponse> generateComsolJava(
            JsonNode chipDesign,
            JsonNode settings,
            String dimension,
            String requestedClassName) {
        return Mono.fromCallable(() -> {
            String normalizedDimension = normalizeComsolDimension(dimension);
            String defaultClassName = "2D".equals(normalizedDimension)
                    ? "MicrofluidicChipModel2D"
                    : "MicrofluidicChipModel3D";
            String className = ComsolJavaGenerator3D.sanitizeClassName(
                    requestedClassName == null || requestedClassName.isBlank()
                            ? defaultClassName
                            : requestedClassName);

            Chip chip = getOrBuildSolvedChip(chipDesign, settings);
            String javaSource = "2D".equals(normalizedDimension)
                    ? chip.generateComsolJava2D(className)
                    : chip.generateComsolJava3D(className);

            return new ComsolResponse(className, normalizedDimension, javaSource);
        });
    }

    //20260624 comsol_zip
    public Mono<ComsolPackageResponse> generateComsolPackage(
            JsonNode chipDesign,
            JsonNode settings,
            String dimension,
            String requestedClassName) {
        return generateComsolJava(chipDesign, settings, dimension, requestedClassName)
                .map(response -> new ComsolPackageResponse(
                        response.getClassName(),
                        response.getDimension(),
                        createComsolPackageZip(response)));
    }

    private byte[] createComsolPackageZip(ComsolResponse response) {
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                ZipOutputStream zip = new ZipOutputStream(bytes, StandardCharsets.UTF_8)) {
            writeZipEntry(zip, response.getClassName() + ".java", response.getJavaSource());
            writeZipEntry(zip, "README.md", buildComsolReadme(response));
            zip.finish();
            return bytes.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to create COMSOL package ZIP", e);
        }
    }

    private void writeZipEntry(ZipOutputStream zip, String fileName, String content) throws IOException {
        zip.putNextEntry(new ZipEntry(fileName));
        zip.write(content.getBytes(StandardCharsets.UTF_8));
        zip.closeEntry();
    }

    private String buildComsolReadme(ComsolResponse response) {
        String className = response.getClassName();
        String javaFileName = className + ".java";
        String classFileName = className + ".class";
        String mphFileName = className + ".mph";

        return """
                # COMSOL Model Package

                This package contains a standalone COMSOL Java API model generated from the synthesized microfluidic chip design.

                ## Files

                - %s: COMSOL Java source file.
                - README.md: Usage notes and default simulation settings.

                ## Generate an MPH file

                1. Open a terminal in this folder.
                2. Make sure COMSOL Multiphysics is installed and `comsolcompile` / `comsolbatch` are available in PATH.
                3. Run:

                   ```bash
                   comsolcompile %s
                   comsolbatch -inputfile %s -outputfile %s
                   ```

                On Windows, if COMSOL commands are not in PATH, use the full executable path, for example:

                ```powershell
                & "C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics\\bin\\win64\\comsolcompile.exe" "%s"
                & "C:\\Program Files\\COMSOL\\COMSOL63\\Multiphysics\\bin\\win64\\comsolbatch.exe" -inputfile "%s" -outputfile "%s"
                ```

                After the command finishes, open `%s` in COMSOL Desktop. The generated Java builds the model, runs the configured studies, updates numerical results and image exports, and saves the computed MPH file.

                ## Model contents

                - Dimension: %s.
                - Geometry unit: micrometer (`um`).
                - Physics: laminar flow, diluted species transport, and droplet/phase-field setup when the design contains a droplet generator.
                - Mesh: default generated mesh sequence with automatic sizing.
                - Studies and results: stationary flow/species study, numerical result, plot, and image export nodes are configured and run by the generated Java during `comsolbatch`. Droplet designs also include an uncomputed transient phase-field study for optional manual simulation in COMSOL Desktop.

                ## Default parameters

                - `rho = 1000[kg/m^3]`: fluid density.
                - `mu = 1e-3[Pa*s]`: dynamic viscosity.
                - `uin_oil = 0.06[m/s]`: mean oil inlet velocity for `inlet_cp`.
                - `uin_water = 0.05[m/s]`: mean water inlet velocity for other inlets.
                - `D = 1e-9[m^2/s]`: species diffusion coefficient.
                - `c_high = 5[mol/m^3]`: tracer concentration at inlet 1.
                - `c_low = 3[mol/m^3]`: tracer concentration at other inlets.
                - `rho_water = 1000[kg/m^3]`: water-phase density.
                - `mu_water = 1e-3[Pa*s]`: water-phase dynamic viscosity.
                - `rho_oil = 900[kg/m^3]`: oil-phase density.
                - `mu_oil = 5e-3[Pa*s]`: oil-phase dynamic viscosity.
                - `sigma = 0.02[N/m]`: water-oil interfacial tension.
                - Phase-field interface thickness uses COMSOL's default expression `pf.ep_default`.
                - `pf_mobility = 7[m*s/kg]`: phase-field mobility tuning parameter.
                - `droplet_dt = 0.1[s]`: droplet stability output time step.
                - `droplet_t_end = 1[s]`: droplet stability final time.
                """.formatted(
                javaFileName,
                javaFileName,
                classFileName,
                mphFileName,
                javaFileName,
                classFileName,
                mphFileName,
                mphFileName,
                response.getDimension());
    }

    private Chip buildAndCacheSolvedChip(JsonNode chipDesign, JsonNode settings) throws Exception {
        String cacheKey = solvedChipCacheKey(chipDesign, settings);
        Chip solvedChip = buildSolvedChip(chipDesign, settings);
        synchronized (solvedChipCache) {
            solvedChipCache.put(cacheKey, solvedChip);
        }
        return solvedChip;
    }

    private Chip getOrBuildSolvedChip(JsonNode chipDesign, JsonNode settings) throws Exception {
        String cacheKey = solvedChipCacheKey(chipDesign, settings);
        synchronized (solvedChipCache) {
            Chip cachedChip = solvedChipCache.get(cacheKey);
            if (cachedChip != null) {
                return cachedChip;
            }

            Chip solvedChip = buildSolvedChip(chipDesign, settings);
            solvedChipCache.put(cacheKey, solvedChip);
            return solvedChip;
        }
    }

    private String solvedChipCacheKey(JsonNode chipDesign, JsonNode settings) {
        try {
            return objectMapper.writeValueAsString(chipDesign)
                    + "\n---settings---\n"
                    + objectMapper.writeValueAsString(settings)
                    + "\n---solver---\n"
                    + mipGap
                    + ":"
                    + mipFocus;
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to build solved chip cache key", e);
        }
    }

    private Chip buildSolvedChip(JsonNode chipDesign, JsonNode settings) throws Exception {
        Chip chip = new Chip();
        Map<String, Object> components = new HashMap<>();
        Map<String, JsonNode> junctions = new HashMap<>();

        // Apply chip settings
        chip.setMargin(settings.get("chipMargin").asInt());
        if (settings.hasNonNull("height")) {
            chip.setHeight(settings.get("height").asInt());
        }
        chip.setDefaultModuleMargin(settings.get("moduleMargin").asInt());
        chip.setDefaultConnectionWidth(settings.get("channelWidth").asInt());
        chip.setDefaultConnectionMargin(settings.get("channelMargin").asInt());
        chip.setLayerHeight(settings.get("layerHeight").asInt());
        double objectiveWidthWeight = doubleValue(settings, "objectiveWidthWeight", 0.3);
        objectiveWidthWeight = Math.max(0.1, Math.min(0.9, objectiveWidthWeight));
        chip.setObjectiveWidthWeight(objectiveWidthWeight);

        createComponents(chipDesign, settings, chip, components);
        createJunctions(chipDesign, settings, components, junctions, chip);
        createConnections(chipDesign, components, junctions, chip);

        chip.initialize();
        chip.solve(120, mipGap, mipFocus, 0.6);
        return chip;
    }

    private String normalizeComsolDimension(String dimension) {
        if (dimension == null || dimension.isBlank()) {
            return "3D";
        }
        String normalized = dimension.trim().toUpperCase(Locale.ROOT);
        if (!normalized.equals("2D") && !normalized.equals("3D")) {
            throw new IllegalArgumentException("COMSOL dimension must be 2D or 3D");
        }
        return normalized;
    }

    private void createComponents(JsonNode chipDefinition, JsonNode settings, Chip chip,
            Map<String, Object> components) {

        Set<String> ports = new HashSet<>();
        Set<String> dropletIds = new HashSet<>();
        Set<String> teslaValveIds = new HashSet<>();

        for (JsonNode connection : arrayField(chipDefinition, "connections")) {
            collectEndpoint(connection.path("source").asText(), ports, dropletIds, teslaValveIds);
            collectEndpoint(connection.path("target").asText(), ports, dropletIds, teslaValveIds);
        }
        for (JsonNode junction : arrayField(chipDefinition, "junctions")) {
            junction.path("sources")
                    .forEach(endpoint -> collectEndpoint(endpoint.asText(), ports, dropletIds, teslaValveIds));
            junction.path("targets")
                    .forEach(endpoint -> collectEndpoint(endpoint.asText(), ports, dropletIds, teslaValveIds));
        }

        ports.forEach(portName -> {
            Port port = new Port(portName, settingInt(settings, "portDiameter", 400));
            components.put(portName, port);
            chip.addModule(port);
        });

        Map<String, JsonNode> dropletParams = byId(componentParamsArray(chipDefinition, "droplets"));
        dropletIds.addAll(dropletParams.keySet());
        dropletIds.forEach(dropletId -> {
            JsonNode params = dropletParams.get(dropletId);
            DropletGenerator dg = new DropletGenerator();
            dg.setName(dropletId);
            dg.setChannelWidth(settingInt(settings, "channelWidth", 200));
            dg.setChannelMargin(settingInt(settings, "channelMargin", 300));
            String dropletType = textValue(params, "type", "t_junction");
            dg.setDropletType(dropletType);
            if ("flow_focusing".equals(dropletType)) {
                dg.setOrientation(Orientation.D180);
            }
            int nozzleWidth = intValue(params, "nozzle_width_um", 100);
            dg.setNozzleWidth(nozzleWidth);
            if (settings != null && settings.hasNonNull("NOZZLE_LENGTH")) {
                dg.setNozzleLength(settings.get("NOZZLE_LENGTH").asInt());
            }
            components.put(dropletId, dg);
            chip.addModule(dg);
        });

        Map<String, JsonNode> teslaValveParams = byId(componentParamsArray(chipDefinition, "tesla_valves"));
        teslaValveIds.addAll(teslaValveParams.keySet());
        teslaValveIds.forEach(teslaValveId -> {
            JsonNode params = teslaValveParams.get(teslaValveId);
            TeslaValve teslaValve = new TeslaValve(
                    teslaValveId,
                    intValue(params, "num_segment_pairs", 2),
                    intValue(params, "segment_length_um", 1000),
                    intValue(params, "segment_width_um", 600));
            teslaValve.setChannelWidth(settingInt(settings, "channelWidth", 200));
            components.put(teslaValveId, teslaValve);
            chip.addModule(teslaValve);
        });

        for (JsonNode chamber : componentParamsArray(chipDefinition, "chambers")) {
            String id = chamber.path("id").asText();
            int length = intValue(chamber, "length_um",
                    chamber.path("dimensions").path("length").asInt(4000));
            int width = intValue(chamber, "width_um",
                    chamber.path("dimensions").path("width").asInt(3200));
            Chamber chamberModule = new Chamber(id, length, width);
            components.put(id, chamberModule);
            chip.addModule(chamberModule);
        }

        for (JsonNode mixer : componentParamsArray(chipDefinition, "mixers")) {
            String mixerId = mixer.path("id").asText();
            String mixerType = textValue(mixer, "type", "serpentine");
            if ("ring".equals(mixerType)) {
                int diameter = intValue(mixer, "diameter_um", 1000);
                int numCircles = intValue(mixer, "num_circles", 3);
                int circleDistance = intValue(mixer, "distance_between_circles_um", 200);
                int channelWidth = settingInt(settings, "channelWidth", 200);
                Mixer ringModule = new Mixer();
                ringModule.setName(mixerId);
                ringModule.setMixerType("ring");
                ringModule.setDiameter(diameter);
                ringModule.setNumCircles(numCircles);
                ringModule.setCircleDistance(circleDistance);
                ringModule.setChannelWidth(channelWidth);
                ringModule.setWidth(diameter + channelWidth * 2);
                ringModule.setLength(numCircles * (diameter + circleDistance + channelWidth));
                components.put(mixerId, ringModule);
                chip.addModule(ringModule);
            } else {
                int numTurnings = intValue(mixer, "num_turnings", 4);
                int amplitude = intValue(mixer, "amplitude_um", 2000);
                int turnDistance = intValue(mixer, "distance_between_turnings_um", 200);
                Mixer mixerModule = new Mixer(mixerId, 2 * amplitude, numTurnings);
                mixerModule.setChannelDistance(turnDistance);
                components.put(mixerId, mixerModule);
                chip.addModule(mixerModule);
            }
        }

        for (JsonNode filter : componentParamsArray(chipDefinition, "filters")) {
            String id = filter.path("id").asText();
            String filterType = textValue(filter, "type", "dld");
            String postShape = textValue(filter, "post_shape", "circle");
            double criticalDiameter = "dld".equals(filterType)
                    ? doubleValue(filter, "critical_particle_diameter_um",
                            doubleValue(filter, "critical_particle_diameter", 10.0))
                    : doubleValue(filter, "post_diameter_um", 400);
            int width = intValue(filter, "width_um", 3200);
            int length = intValue(filter, "length_um", 4000);
            int postDiameter = intValue(filter, "post_diameter_um", 400);

            double rowShiftFraction = doubleValue(filter, "row_shift_fraction", 0.20);

            Filter filterModule = new Filter(id, width, length, criticalDiameter, Math.max(1, postDiameter / 2),
                    postShape, rowShiftFraction);
            filterModule.setCornerRadius(settingInt(settings, "channelWidth", 200));
            if ("pillar_matrix".equals(filterType)) {
                filterModule.setPillarMatrix(true);
                filterModule.setPillarCols(intValue(filter, "columns", 3));
                filterModule.setPillarRows(intValue(filter, "rows", 4));
                filterModule.setEpsilon(0.0);
                filterModule.setDeltaLambda(0.0);
            }
            components.put(id, filterModule);
            chip.addModule(filterModule);
        }

        for (JsonNode delay : componentParamsArray(chipDefinition, "delays")) {
            int numTurnings = intValue(delay, "num_turnings", 4);
            int amplitude = intValue(delay, "amplitude_um", 2000);
            int turnDistance = intValue(delay, "distance_between_turnings_um", 200);
            Mixer delayModule = new Mixer(delay.path("id").asText(), 2 * amplitude, numTurnings);
            delayModule.setChannelDistance(turnDistance);
            components.put(delay.path("id").asText(), delayModule);
            chip.addModule(delayModule);
        }
    }

    private void collectEndpoint(String name, Set<String> ports, Set<String> dropletIds, Set<String> teslaValveIds) {
        addPort(name, ports);
        extractDropletId(name, dropletIds);
        extractTeslaValveId(name, teslaValveIds);
    }

    private void addPort(String name, Set<String> ports) {
        if (name.startsWith("inlet_") || name.startsWith("outlet_")) {
            ports.add(name);
        }
    }

    private void extractDropletId(String str, Set<String> dropletIds) {
        if (str.startsWith("droplet_")) {
            String[] parts = str.split("_");
            if (parts.length >= 2) {
                dropletIds.add(parts[0] + "_" + parts[1]);
            }
        }
    }

    private void extractTeslaValveId(String str, Set<String> teslaValveIds) {
        if (str.startsWith("tesla_valve_")) {
            String[] parts = str.split("_");
            if (parts.length >= 3) {
                teslaValveIds.add(parts[0] + "_" + parts[1] + "_" + parts[2]);
            }
        }
    }

    private void createJunctions(JsonNode chipDefinition, JsonNode settings, Map<String, Object> components,
            Map<String, JsonNode> junctions, Chip chip) {

        for (JsonNode junction : arrayField(chipDefinition, "junctions")) {
            JsonNode sources = junction.path("sources");
            JsonNode targets = junction.path("targets");
            boolean isMultiPort = sources.size() + targets.size() > 3;
            int angle = isMultiPort || "T-junction".equalsIgnoreCase(junction.path("type").asText()) ? 90
                    : intValue(junction, "angle", 45);
            int channelWidth = settingInt(settings, "channelWidth", 200);
            int channelMargin = settingInt(settings, "channelMargin", 300);

            boolean isSplitting = sources.size() == 1 && targets.size() >= 2;
            int inputCount = Math.max(2, isSplitting ? targets.size() : sources.size());

            addJointModule(junction.path("id").asText(), angle, channelWidth, channelMargin,
                    inputCount, settings, components, chip);
            junctions.put(junction.path("id").asText(), junction);
        }
    }

    private void addJointModule(String name, int angle, int channelWidth, int channelMargin, int inputCount,
            JsonNode settings, Map<String, Object> components, Chip chip) {
        int portDiameter = channelWidth * 2;
        int portGap = angle == 90 && inputCount > 2
                ? Math.max(channelMargin, settingInt(settings, "moduleMargin", 400))
                : channelMargin;
        int jointLength = channelMargin * 2 + channelWidth * 2;
        int jointWidth = angle == 90 ? jointLength : (int) (jointLength * Math.cos(Math.toRadians(angle)));
        if (angle == 90 && inputCount > 2) {
            jointWidth = 2 * channelMargin + inputCount * portDiameter + (inputCount - 1) * portGap;
            jointLength = 2 * channelMargin + portDiameter + channelWidth * 8;
        }

        Joint joint = Joint.builder()
                .name(name)
                .angle(angle)
                .orientation(angle == 90 && inputCount > 2 ? Orientation.D0 : null)
                .channelWidth(channelWidth)
                .channelMargin(channelMargin)
                .inputCount(inputCount)
                .portDiameter(portDiameter)
                .portGap(portGap)
                .width(jointWidth)
                .length(jointLength)
                .build();

        components.put(name, joint);
        chip.addModule(joint);
    }

    private void createConnections(JsonNode chipDefinition, Map<String, Object> components,
            Map<String, JsonNode> junctions, Chip chip) {

        Set<Connection> connections = new HashSet<>();
        Set<String> sourceComponents = new HashSet<>();
        Set<String> targetComponents = new HashSet<>();
        Set<String> processedJunctionEdges = new HashSet<>();

        for (JsonNode junction : arrayField(chipDefinition, "junctions")) {
            createV2JunctionConnections(junction, components, junctions, connections, sourceComponents,
                    targetComponents, processedJunctionEdges);
        }

        for (JsonNode connection : arrayField(chipDefinition, "connections")) {
            String source = connection.path("source").asText();
            String target = connection.path("target").asText();
            Pin sourcePin = resolvePin(source, components, true);
            Pin targetPin = resolvePin(target, components, false);

            if (!sourceComponents.contains(source) && !targetComponents.contains(target)) {
                connections.add(new Connection(sourcePin, targetPin));
            }
        }

        addAutomaticDropletContinuousInputs(chipDefinition, components, chip, connections);

        chip.addConnection(connections.toArray(new Connection[0]));
    }

    private void addAutomaticDropletContinuousInputs(JsonNode chipDefinition, Map<String, Object> components,
            Chip chip, Set<Connection> connections) {

        Map<String, Set<String>> dropletInputPins = collectDropletInputPins(chipDefinition);

        new ArrayList<>(components.entrySet()).forEach(entry -> {
            String componentId = entry.getKey();
            Object component = entry.getValue();
            if (!(component instanceof DropletGenerator droplet)) {
                return;
            }

            Set<String> inputPins = dropletInputPins.getOrDefault(componentId, Collections.emptySet());
            if (!inputPins.contains("dispersed") || inputPins.contains("continuous")) {
                return;
            }

            String autoPortId = uniqueAutoPortId(components, "inlet_cp");
            Port autoPort = new Port(autoPortId, chip.getDefaultConnectionWidth() * 2);
            components.put(autoPortId, autoPort);
            chip.addModule(autoPort);
            connections.add(new Connection(autoPort.getPortPin(), droplet.getContinuousInputPin()));

            if (!"flow_focusing".equals(droplet.getDropletType())) {
                return;
            }

            String secondAutoPortId = uniqueAutoPortId(components, "inlet_cp");
            Port secondAutoPort = new Port(secondAutoPortId, chip.getDefaultConnectionWidth() * 2);
            components.put(secondAutoPortId, secondAutoPort);
            chip.addModule(secondAutoPort);
            connections.add(new Connection(secondAutoPort.getPortPin(), droplet.getSecondContinuousInputPin()));
        });
    }

    private String uniqueAutoPortId(Map<String, Object> components, String baseId) {
        if (!components.containsKey(baseId)) {
            return baseId;
        }
        int index = 2;
        while (components.containsKey(baseId + "_" + index)) {
            index++;
        }
        return baseId + "_" + index;
    }

    private Map<String, Set<String>> collectDropletInputPins(JsonNode chipDefinition) {
        Map<String, Set<String>> dropletInputPins = new HashMap<>();

        for (JsonNode connection : arrayField(chipDefinition, "connections")) {
            recordDropletInputPin(connection.path("target").asText(), dropletInputPins);
        }

        for (JsonNode junction : arrayField(chipDefinition, "junctions")) {
            junction.path("targets")
                    .forEach(endpoint -> recordDropletInputPin(endpoint.asText(), dropletInputPins));
        }

        return dropletInputPins;
    }

    private void recordDropletInputPin(String endpoint, Map<String, Set<String>> dropletInputPins) {
        DropletEndpoint dropletEndpoint = parseDropletEndpoint(endpoint);
        if (dropletEndpoint == null) {
            return;
        }
        dropletInputPins
                .computeIfAbsent(dropletEndpoint.id(), id -> new HashSet<>())
                .add(dropletEndpoint.inputPin());
    }

    private DropletEndpoint parseDropletEndpoint(String endpoint) {
        if (endpoint == null || !endpoint.startsWith("droplet_")) {
            return null;
        }

        String[] parts = endpoint.split("_");
        if (parts.length < 2) {
            return null;
        }

        String dropletId = parts[0] + "_" + parts[1];
        if (parts.length == 2) {
            return new DropletEndpoint(dropletId, "dispersed");
        }
        if ("continuous".equals(parts[2]) || "dispersed".equals(parts[2])) {
            return new DropletEndpoint(dropletId, parts[2]);
        }
        return null;
    }

    private record DropletEndpoint(String id, String inputPin) {
    }

    private void createV2JunctionConnections(
            JsonNode junction,
            Map<String, Object> components,
            Map<String, JsonNode> junctions,
            Set<Connection> connections,
            Set<String> sourceComponents,
            Set<String> targetComponents,
            Set<String> processedJunctionEdges) {

        String junctionId = junction.path("id").asText();
        Joint joint = (Joint) components.get(junctionId);
        JsonNode sources = junction.path("sources");
        JsonNode targets = junction.path("targets");
        boolean isSplitting = sources.size() == 1 && targets.size() >= 2;

        if (isSplitting) {
            String sourceId = sources.get(0).asText();
            Pin sourcePin = resolvePin(resolveJunctionEndpoint(junctions, junctionId, sourceId, true), components,
                    true);
            sourceComponents.add(sourceId);
            connections.add(new Connection(sourcePin, joint.getOutputPin()));

            for (int i = 0; i < targets.size(); i++) {
                String targetId = targets.get(i).asText();
                if (targetId.startsWith("junction") && !processedJunctionEdges.add(junctionId + "->" + targetId)) {
                    continue;
                }
                String resolvedTarget = resolveJunctionEndpoint(junctions, junctionId, targetId, false);
                Pin targetPin = resolvePin(resolvedTarget, components, false);
                targetComponents.add(targetId);
                connections.add(new Connection(getJointInputPin(joint, i), targetPin));
            }
        } else {
            String targetId = targets.get(0).asText();
            Pin targetPin = resolvePin(resolveJunctionEndpoint(junctions, junctionId, targetId, false), components,
                    false);
            targetComponents.add(targetId);

            for (int i = 0; i < sources.size(); i++) {
                String sourceId = sources.get(i).asText();
                if (sourceId.startsWith("junction") && !processedJunctionEdges.add(sourceId + "->" + junctionId)) {
                    continue;
                }
                Pin sourcePin = resolvePin(resolveJunctionEndpoint(junctions, junctionId, sourceId, true), components,
                        true);
                sourceComponents.add(sourceId);
                connections.add(new Connection(sourcePin, getJointInputPin(joint, i)));
            }
            connections.add(new Connection(joint.getOutputPin(), targetPin));
        }
    }

    private String resolveJunctionEndpoint(Map<String, JsonNode> junctions, String currentJunctionId,
            String endpointId, boolean isEndpointSource) {
        return endpointId.startsWith("junction")
                ? getJunctionPinName(junctions, currentJunctionId, endpointId, isEndpointSource)
                : endpointId;
    }

    private Pin getJointInputPin(Joint joint, int index) {
        return switch (index) {
            case 0 -> joint.getInput1Pin();
            case 1 -> joint.getInput2Pin();
            default -> joint.getInputPin(index);
        };
    }

    private String getJunctionPinName(Map<String, JsonNode> junctions, String currentJunctionId,
            String desiredJunctionId, boolean isDesiredSource) {

        JsonNode desiredJunction = junctions.get(desiredJunctionId);
        if (desiredJunction == null) {
            throw new IllegalArgumentException("Unknown junction reference: " + desiredJunctionId);
        }

        JsonNode sources = desiredJunction.path("sources");
        JsonNode targets = desiredJunction.path("targets");
        boolean isDesiredSplitting = sources.size() == 1 && targets.size() >= 2;

        if (isDesiredSource) {
            if (!isDesiredSplitting) {
                return desiredJunctionId;
            }
            for (int i = 0; i < targets.size(); i++) {
                if (targets.get(i).asText().equalsIgnoreCase(currentJunctionId)) {
                    return desiredJunctionId + "_pin" + (i + 1);
                }
            }
        } else {
            if (isDesiredSplitting) {
                return desiredJunctionId;
            }
            for (int i = 0; i < sources.size(); i++) {
                if (sources.get(i).asText().equalsIgnoreCase(currentJunctionId)) {
                    return desiredJunctionId + "_pin" + (i + 1);
                }
            }
        }
        throw new IllegalArgumentException(
                "Invalid connection between junctions: " + currentJunctionId + " and " + desiredJunctionId);
    }

    private Pin resolvePin(String str, Map<String, Object> components, boolean isSource) {
        Object component = components.get(str);
        if (component != null) {
            return getDefaultPin(component, isSource);
        }

        String[] parts = str.split("_(?=[^_]+$)");
        String base = parts[0];
        String pinName = parts.length > 1 ? parts[1] : null;

        component = components.get(base);
        if (component == null) {
            throw new IllegalArgumentException("Component not found: " + base);
        }

        return getPin(component, pinName, isSource);
    }

    private Pin getDefaultPin(Object component, boolean isSource) {
        if (component instanceof Port) {
            return ((Port) component).getPortPin();
        } else if (component instanceof TeslaValve) {
            return isSource ? ((TeslaValve) component).getOutputPin() : ((TeslaValve) component).getInputPin();
        } else if (component instanceof Mixer) {
            return isSource ? ((Mixer) component).getOutputPin() : ((Mixer) component).getInputPin();
        } else if (component instanceof Filter) {
            return ((Filter) component).getPin(isSource ? Filter.Pins.SmallerOut : Filter.Pins.In);
        } else if (component instanceof DropletGenerator) {
            return isSource ? ((DropletGenerator) component).getOutputPin()
                    : ((DropletGenerator) component).getDispersedInputPin();
        } else if (component instanceof Chamber) {
            return isSource ? ((Chamber) component).getPin(Chamber.Pins.East)
                    : ((Chamber) component).getPin(Chamber.Pins.West);
        } else if (component instanceof Joint) {
            return ((Joint) component).getOutputPin();
        }

        throw new IllegalArgumentException("Unknown component type: " + component.getClass());
    }

    private Pin getPin(Object component, String pinName, boolean isSource) {
        if (pinName == null) {
            return getDefaultPin(component, isSource);
        }

        if (component instanceof Filter) {
            return switch (pinName) {
                case "smaller" -> ((Filter) component).getPin(Filter.Pins.SmallerOut);
                case "larger" -> ((Filter) component).getPin(Filter.Pins.LargerOut);
                default -> throw new IllegalArgumentException("Invalid pin for Filter: " + pinName);
            };
        } else if (component instanceof DropletGenerator) {
            return switch (pinName) {
                case "dispersed" -> ((DropletGenerator) component).getDispersedInputPin();
                case "continuous" -> ((DropletGenerator) component).getContinuousInputPin();
                default -> throw new IllegalArgumentException("Invalid pin for DropletGenerator: " + pinName);
            };
        } else if (component instanceof Joint) {
            if (pinName.matches("pin\\d+")) {
                int inputIndex = Integer.parseInt(pinName.substring(3)) - 1;
                return ((Joint) component).getInputPin(inputIndex);
            }
            throw new IllegalArgumentException("Invalid pin for Junction: " + pinName);
        }
        throw new IllegalArgumentException("Unsupported component for pin resolution: " + component.getClass());
    }

    private JsonNode arrayField(JsonNode node, String fieldName) {
        JsonNode value = node.path(fieldName);
        return value.isArray() ? value : objectMapper.createArrayNode();
    }

    private JsonNode componentParamsArray(JsonNode chipDefinition, String fieldName) {
        JsonNode value = chipDefinition.path("component_params").path(fieldName);
        return value.isArray() ? value : objectMapper.createArrayNode();
    }

    private Map<String, JsonNode> byId(JsonNode array) {
        Map<String, JsonNode> result = new HashMap<>();
        array.forEach(item -> {
            if (item.hasNonNull("id")) {
                result.put(item.get("id").asText(), item);
            }
        });
        return result;
    }

    private int settingInt(JsonNode settings, String fieldName, int defaultValue) {
        return intValue(settings, fieldName, defaultValue);
    }

    private int intValue(JsonNode node, String fieldName, int defaultValue) {
        return node != null && node.hasNonNull(fieldName) ? node.get(fieldName).asInt() : defaultValue;
    }

    private double doubleValue(JsonNode node, String fieldName, double defaultValue) {
        return node != null && node.hasNonNull(fieldName) ? node.get(fieldName).asDouble() : defaultValue;
    }

    private boolean booleanValue(JsonNode node, String fieldName, boolean defaultValue) {
        return node != null && node.hasNonNull(fieldName) ? node.get(fieldName).asBoolean() : defaultValue;
    }

    private String textValue(JsonNode node, String fieldName, String defaultValue) {
        return node != null && node.hasNonNull(fieldName) ? node.get(fieldName).asText() : defaultValue;
    }

    private record SolvedChipView(Chip chip, String svg, String chipJson) {
    }

    // Helper class for synthesis response
    public static class SynthesisResponse {
        private final String svg;
        private final String stlBase64; // optional, may be null if timeout occurs
        private final String chipJson;

        public SynthesisResponse(String svg, String stlBase64, String chipJson) {
            this.svg = svg;
            this.stlBase64 = stlBase64;
            this.chipJson = chipJson;
        }

        public String getSvg() {
            return svg;
        }

        public String getStlBase64() {
            return stlBase64;
        }

        public String getChipJson() {
            return chipJson;
        }
    }

    public static class ComsolResponse {
        private final String className;
        private final String dimension;
        private final String javaSource;

        public ComsolResponse(String className, String dimension, String javaSource) {
            this.className = className;
            this.dimension = dimension;
            this.javaSource = javaSource;
        }

        public String getClassName() {
            return className;
        }

        public String getDimension() {
            return dimension;
        }

        public String getJavaSource() {
            return javaSource;
        }
    }

    //20260624 comsol_zip
    public static class ComsolPackageResponse {
        private final String className;
        private final String dimension;
        private final byte[] zipBytes;

        public ComsolPackageResponse(String className, String dimension, byte[] zipBytes) {
            this.className = className;
            this.dimension = dimension;
            this.zipBytes = zipBytes;
        }

        public String getClassName() {
            return className;
        }

        public String getDimension() {
            return dimension;
        }

        public byte[] getZipBytes() {
            return zipBytes;
        }

        public String getZipFileName() {
            return className + ".zip";
        }
    }

}
