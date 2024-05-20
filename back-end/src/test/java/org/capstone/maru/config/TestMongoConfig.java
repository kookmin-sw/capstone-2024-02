package org.capstone.maru.config;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@EnableJpaAuditing
@TestConfiguration
@EnableJpaRepositories(basePackages = "org.capstone.maru.repository.mongodb")
public class TestMongoConfig {

}
