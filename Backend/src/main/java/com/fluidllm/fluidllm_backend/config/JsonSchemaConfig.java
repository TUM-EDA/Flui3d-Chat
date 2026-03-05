package com.fluidllm.fluidllm_backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import java.io.IOException;
import java.util.Map;

@Configuration
public class JsonSchemaConfig {

    @Value("classpath:microfluidic_schema.json")
    private Resource schemaResource;

    @Bean
    public Map<String, Object> microfluidicJsonSchema(ObjectMapper objectMapper) throws IOException {
        // Reads the microfluidic json schema from resources
        return objectMapper.readValue(
                schemaResource.getInputStream(),
                new TypeReference<Map<String, Object>>() {
                });
    }
}