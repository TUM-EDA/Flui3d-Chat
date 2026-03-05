package com.fluidllm.fluidllm_backend.service;

import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Value;

import reactor.core.publisher.Mono;
import reactor.util.function.Tuples;

import com.fasterxml.jackson.databind.JsonNode;

import com.threedmf.designsynthesis.chip.Chip;
import com.threedmf.designsynthesis.connection.Connection;
import com.threedmf.designsynthesis.module.*;

import org.jfree.svg.ViewBox;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.time.Duration;

@Service
public class MicrofluidicChipSynthesisService {

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

            Chip chip = new Chip();
            Map<String, Object> components = new HashMap<>();
            Map<String, JsonNode> junctions = new HashMap<>();

            // Apply chip settings
            chip.setMargin(settings.get("chipMargin").asInt());
            chip.setDefaultModuleMargin(settings.get("moduleMargin").asInt());
            chip.setDefaultConnectionWidth(settings.get("channelWidth").asInt());
            chip.setDefaultConnectionMargin(settings.get("channelMargin").asInt());
            chip.setLayerHeight(settings.get("layerHeight").asInt());

            // Create components (Ports, Mixers, etc.) and add to chip
            createComponents(chipDesign, settings, chip, components);

            // Create junctions and add to chip
            createJunctions(chipDesign, settings, components, junctions, chip);

            // Create connections and add to chip
            createConnections(chipDesign, components, junctions, chip);

            // initialize and solve
            chip.initialize();

            chip.solve(120, mipGap, mipFocus, 0.6);

            // Generate SVG immediately
            String chipSvg = chip.drawSVG(1).getSVGElement(null, false,
                    new ViewBox(0, 0, chip.getWidth(), chip.getLength()), null, null);

            // Also return chip for STL generation
            return Tuples.of(chip, chipSvg);
        })
                .flatMap(tuple -> {
                    Chip chip = tuple.getT1();
                    String chipSvg = tuple.getT2();

                    // Async STL generation with timeout
                    CompletableFuture<String> stlFuture = CompletableFuture.supplyAsync(() -> {
                        try {
                            byte[] stlBytes = chip.generateSTL().stlToBytes();
                            return Base64.getEncoder().encodeToString(stlBytes);
                        } catch (Exception e) {
                            return null;
                        }
                    });

                    return Mono.fromFuture(stlFuture)
                            .timeout(Duration.ofSeconds(stlTimeout))
                            .onErrorResume(e -> Mono.empty())
                            .map(stlBase64 -> new SynthesisResponse(chipSvg, stlBase64))
                            .defaultIfEmpty(new SynthesisResponse(chipSvg, null));
                });
    }

    /**
     * Instantiates all non-junction microfluidic components from the JSON
     * definition and adds them to the 3M-DeSyn Chip object. This helper method
     * iterates through the 'connections' and 'component_params' sections of the
     * JSON to identify and create all necessary modules like Ports, Chambers,
     * Mixers, Filters, Delays, and Droplet Generators, translating their properties
     * into the corresponding 3M-DeSyn library objects.
     */
    private void createComponents(JsonNode chipDefinition, JsonNode settings, Chip chip,
            Map<String, Object> components) {

        // Create Ports (inlets and outlets) & DropletGenerators
        Set<String> ports = new HashSet<>();
        Set<String> dropletIds = new HashSet<>();
        chipDefinition.get("connections").forEach(conn -> {
            addPort(conn.get("source").asText(), ports);
            addPort(conn.get("target").asText(), ports);
            extractDropletId(conn.get("source").asText(), dropletIds);
            extractDropletId(conn.get("target").asText(), dropletIds);
        });
        ports.forEach(portName -> {
            Port port = new Port(portName, settings.get("portDiameter").asInt());
            components.put(portName, port);
            chip.addModule(port);
        });
        dropletIds.forEach(dropletId -> {
            DropletGenerator dg = DropletGenerator.builder().name(dropletId).channelMargin(200).build();
            components.put(dropletId, dg);
            chip.addModule(dg);
        });

        // Create Chambers
        chipDefinition.get("component_params").get("chambers").forEach(chamber -> {
            Chamber chamberModule = new Chamber(chamber.get("id").asText(),
                    chamber.get("dimensions").get("length").asInt(), chamber.get("dimensions").get("width").asInt());
            components.put(chamber.get("id").asText(), chamberModule);
            chip.addModule(chamberModule);
        });

        // Create Mixers
        chipDefinition.get("component_params").get("mixers").forEach(mixer -> {
            Mixer mixerModule = new Mixer(mixer.get("id").asText(), settings.get("serpentineWidth").asInt(),
                    mixer.get("num_turnings").asInt());
            components.put(mixer.get("id").asText(), mixerModule);
            chip.addModule(mixerModule);
        });

        // Create Filters
        chipDefinition.get("component_params").get("filters").forEach(filter -> {
            Filter filterModule = new Filter(filter.get("id").asText(), settings.get("filterWidth").asInt(),
                    settings.get("filterHeight").asInt(), filter.get("critical_particle_diameter").asDouble(),
                    settings.get("filterPillarRadius").asInt());
            components.put(filter.get("id").asText(), filterModule);
            chip.addModule(filterModule);
        });

        // Create Delays
        chipDefinition.get("component_params").get("delays").forEach(delay -> {
            Mixer delayModule = new Mixer(delay.get("id").asText(), settings.get("serpentineWidth").asInt(),
                    delay.get("num_turnings").asInt());
            components.put(delay.get("id").asText(), delayModule);
            chip.addModule(delayModule);
        });

    }

    /**
     * A helper method to identify and collect port names (inlets and outlets) from
     * a given component ID string.
     */
    private void addPort(String name, Set<String> ports) {
        if (name.startsWith("inlet_") || name.startsWith("outlet_")) {
            ports.add(name);
        }
    }

    /**
     * A helper method to extract the base ID of a DropletGenerator from a
     * port-specific string. For example, it extracts "droplet_1" from
     * "droplet_1_continuous".
     */
    private void extractDropletId(String str, Set<String> dropletIds) {
        if (str.startsWith("droplet_")) {
            String[] parts = str.split("_");
            if (parts.length >= 2) {
                dropletIds.add(parts[0] + "_" + parts[1]);
            }
        }
    }

    /**
     * Instantiates all junction components from the JSON definition and adds them
     * to the Chip object. It iterates through the 'junctions' array, creating a
     * Joint module from the 3M-DeSyn library for each entry. The angle of the joint
     * is set based on whether the type is specified as a "T-junction" (90 degrees)
     * or "Y-junction" (45 degrees).
     */
    private void createJunctions(JsonNode chipDefinition, JsonNode settings, Map<String, Object> components,
            Map<String, JsonNode> junctions, Chip chip) {

        for (JsonNode junction : chipDefinition.get("junctions")) {
            // Create Joint component
            int angle = "T-junction".equalsIgnoreCase(junction.get("type").asText()) ? 90 : 45;
            Joint joint = Joint.builder()
                    .name(junction.get("id").asText())
                    .angle(angle)
                    .channelMargin(settings.get("channelMargin").asInt())
                    .build();

            junctions.put(junction.get("id").asText(), junction);
            components.put(junction.get("id").asText(), joint);
            chip.addModule(joint);

        }
    }

    /**
     * Translates the abstract connections from the JSON definition into physical
     * connections in the 3M-DeSyn Chip object. This method is responsible for
     * creating all the Connection objects between the instantiated modules. It
     * handles both direct component-to-component connections and the more complex
     * routing through the junction network by resolving the correct source and
     * target pins for each connection.
     */
    private void createConnections(JsonNode chipDefinition, Map<String, Object> components,
            Map<String, JsonNode> junctions, Chip chip) {

        Set<Connection> connections = new HashSet<>();

        Set<String> sourceComponents = new HashSet<>();
        Set<String> targetComponents = new HashSet<>();

        // Create junction connections
        for (JsonNode junction : chipDefinition.get("junctions")) {

            String junctionId = junction.get("id").asText();

            Joint joint = (Joint) components.get(junctionId);

            // Resolve pins and create connections
            if (junction.has("source")) {

                // Resolve source pin (input pin)
                String sourceId = junction.get("source").asText();
                String sourcePinName = sourceId.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, sourceId, true)
                        : sourceId;
                Pin sourcePin = resolvePin(sourcePinName, components, true);

                // Resolve target_1 pin (output pin)
                String target1Id = junction.get("target_1").asText();
                String target1PinName = target1Id.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, target1Id, false)
                        : target1Id;
                Pin target1Pin = resolvePin(target1PinName, components, false);

                // Resolve target_2 pin (output pin)
                String target2Id = junction.get("target_2").asText();
                String target2PinName = target2Id.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, target2Id, false)
                        : target2Id;
                Pin target2Pin = resolvePin(target2PinName, components, false);

                sourceComponents.add(junction.get("source").asText());
                targetComponents.add(junction.get("target_1").asText());
                targetComponents.add(junction.get("target_2").asText());

                // Joints in 3-dmf-design-synthesis only have 2 inputs and 1 output so we have
                // to switch here
                connections.add(new Connection(sourcePin, joint.getOutputPin()));
                connections.add(new Connection(joint.getInput1Pin(), target1Pin));
                connections.add(new Connection(joint.getInput2Pin(), target2Pin));
            } else {

                // Resolve source_1 pin (input pin)
                String source1Id = junction.get("source_1").asText();
                String source1PinName = source1Id.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, source1Id, true)
                        : source1Id;
                Pin source1Pin = resolvePin(source1PinName, components, true);

                // Resolve source_2 pin (input pin)
                String source2Id = junction.get("source_2").asText();
                String source2PinName = source2Id.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, source2Id, true)
                        : source2Id;
                Pin source2Pin = resolvePin(source2PinName, components, true);

                // Resolve target pin (output pin)
                String targetId = junction.get("target").asText();
                String targetPinName = targetId.startsWith("junction")
                        ? getJunctionPinName(junctions, junctionId, targetId, false)
                        : targetId;
                Pin targetPin = resolvePin(targetPinName, components, false);

                sourceComponents.add(junction.get("source_1").asText());
                sourceComponents.add(junction.get("source_2").asText());
                targetComponents.add(junction.get("target").asText());

                connections.add(new Connection(source1Pin, joint.getInput1Pin()));
                connections.add(new Connection(source2Pin, joint.getInput2Pin()));
                connections.add(new Connection(joint.getOutputPin(), targetPin));
            }
        }

        // Create other connections
        for (JsonNode connection : chipDefinition.get("connections")) {
            Pin sourcePin = resolvePin(connection.get("source").asText(), components, true);
            Pin targetPin = resolvePin(connection.get("target").asText(), components, false);

            // Only add if not connected with junctions
            if (!sourceComponents.contains(connection.get("source").asText())
                    && !targetComponents.contains(connection.get("target").asText())) {
                connections.add(new Connection(sourcePin, targetPin));
            }
        }

        // Add all connections
        chip.addConnection(connections.toArray(new Connection[0]));
    }

    /**
     * Resolves the specific pin name for a junction-to-junction connection. Since
     * the 3M-DeSyn Joint has distinct input and output pins ("pin1", "pin2"), this
     * method determines which pin of the desiredJunctionId is connected to the
     * currentJunctionId based on the topology defined in the JSON.
     */
    private String getJunctionPinName(Map<String, JsonNode> junctions, String currentJunctionId,
            String desiredJunctionId, boolean isDesiredSource) {

        boolean isDesiredSplitting = junctions.get(desiredJunctionId).has("source");

        if (isDesiredSource) {

            if (!isDesiredSplitting) {
                // Default port (single)
                return desiredJunctionId;
            } else if (junctions.get(desiredJunctionId).get("target_1").asText().equalsIgnoreCase(currentJunctionId)) {
                return (desiredJunctionId + "_pin1");
            } else if (junctions.get(desiredJunctionId).get("target_2").asText().equalsIgnoreCase(currentJunctionId)) {
                return (desiredJunctionId + "_pin2");
            }
            throw new IllegalArgumentException(
                    "Invalid connection between junctions: " + currentJunctionId + " and " + desiredJunctionId);

        } else {

            if (isDesiredSplitting) {
                // Default port (single)
                return desiredJunctionId;
            } else if (junctions.get(desiredJunctionId).get("source_1").asText().equalsIgnoreCase(currentJunctionId)) {
                return (desiredJunctionId + "_pin1");
            } else if (junctions.get(desiredJunctionId).get("source_2").asText().equalsIgnoreCase(currentJunctionId)) {
                return (desiredJunctionId + "_pin2");
            }
            throw new IllegalArgumentException(
                    "Invalid connection between junctions: " + currentJunctionId + " and " + desiredJunctionId);

        }
    }

    /**
     * A helper to resolve a component ID string to a specific Pin object. This
     * method can handle both simple component IDs (e.g., "mixer_1") and compound
     * IDs that specify a particular port (e.g., "filter_1_smaller"). It acts as a
     * dispatcher to more specific pin resolution methods.
     */
    private Pin resolvePin(String str, Map<String, Object> components, boolean isSource) {
        Object component = components.get(str);
        if (component != null) {
            return getDefaultPin(component, isSource);
        }

        // Handle pins like "droplet_1_dispersed"
        String[] parts = str.split("_(?=[^_]+$)");
        String base = parts[0];
        String pinName = parts.length > 1 ? parts[1] : null;

        component = components.get(base);
        if (component == null) {
            throw new IllegalArgumentException("Component not found: " + base);
        }

        return getPin(component, pinName, isSource);
    }

    /**
     * Returns the default pin for a given component when no specific pin is named.
     * For most components, this will be the single input pin (if target) or output
     * pin (if source).
     */
    private Pin getDefaultPin(Object component, boolean isSource) {
        if (component instanceof Port) {
            return ((Port) component).getPortPin();
        } else if (component instanceof Mixer) {
            return isSource ? ((Mixer) component).getOutputPin() : ((Mixer) component).getInputPin();
        } else if (component instanceof Filter) {
            return ((Filter) component).getPin(Filter.Pins.In);
        } else if (component instanceof DropletGenerator) {
            return ((DropletGenerator) component).getOutputPin();
        } else if (component instanceof Chamber) {
            return isSource ? ((Chamber) component).getPin(Chamber.Pins.East)
                    : ((Chamber) component).getPin(Chamber.Pins.West);
        } else if (component instanceof Joint) {
            return ((Joint) component).getOutputPin();
        }

        throw new IllegalArgumentException("Unknown component type: " + component.getClass());
    }

    /**
     * Resolves a specific, named pin on a component. This method handles compound
     * IDs (e.g., "filter_1_smaller" or "droplet_1_continuous") by parsing the pin
     * name and returning the corresponding Pin object from the specialized
     * component (e.g., Filter, DropletGenerator, Joint).
     */
    private Pin getPin(Object component, String pinName, boolean isSource) {
        if (pinName == null) {
            return getDefaultPin(component, isSource);
        }

        // Handle specific pins (e.g., "dispersed", "continuous")
        if (component instanceof Filter) {
            switch (pinName) {
                case "smaller":
                    return ((Filter) component).getPin(Filter.Pins.SmallerOut);
                case "larger":
                    return ((Filter) component).getPin(Filter.Pins.LargerOut);
                default:
                    throw new IllegalArgumentException("Invalid pin for Filter: " + pinName);
            }
        } else if (component instanceof DropletGenerator) {
            switch (pinName) {
                case "dispersed":
                    return ((DropletGenerator) component).getDispersedInputPin();
                case "continuous":
                    return ((DropletGenerator) component).getContinuousInputPin();
                default:
                    throw new IllegalArgumentException("Invalid pin for DropletGenerator: " + pinName);
            }
        } else if (component instanceof Joint) {

            switch (pinName) {
                case "pin1":
                    return ((Joint) component).getInput1Pin();
                case "pin2":
                    return ((Joint) component).getInput2Pin();
                default:
                    throw new IllegalArgumentException("Invalid pin for Junction: " + pinName);
            }
        }
        throw new IllegalArgumentException("Unsupported component for pin resolution: " + component.getClass());
    }

    // Helper class for synthesis response
    public static class SynthesisResponse {
        private final String svg;
        private final String stlBase64; // optional, may be null if timeout occurs

        public SynthesisResponse(String svg, String stlBase64) {
            this.svg = svg;
            this.stlBase64 = stlBase64;
        }

        public String getSvg() {
            return svg;
        }

        public String getStlBase64() {
            return stlBase64;
        }
    }

}