package org.capstone.maru.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.repository.config.EnableMongoRepositories;

@Configuration
@EnableMongoRepositories(basePackages = "org.capstone.maru.repository.mongodb")
public class MongoConfig {

}
