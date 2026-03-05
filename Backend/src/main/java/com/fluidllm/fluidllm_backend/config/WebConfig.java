package com.fluidllm.fluidllm_backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.AsyncSupportConfigurer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.lang.NonNull;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    // Inject stl timeout value from application.properties
    @Value("${desyn.stl.timeout}")
    private int requestTimeout;

    @Override
    public void configureAsyncSupport(@NonNull AsyncSupportConfigurer configurer) {
        // This sets the global timeout for all @Controller async requests
        configurer.setDefaultTimeout((long) (requestTimeout + 10) * 1000);
    }
}