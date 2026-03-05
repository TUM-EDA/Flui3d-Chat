package com.fluidllm.fluidllm_backend.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Bean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.springframework.util.StreamUtils;

@Configuration
public class GrammarConfig {

    @Value("classpath:microfluidic_reasoning_grammar.gbnf")
    private Resource grammarResource;

    @Bean
    public String microfluidcReasoningGrammar() throws IOException {
        // Reads the microfluidic grammar from resources
        return StreamUtils.copyToString(grammarResource.getInputStream(), StandardCharsets.UTF_8);
    }
}