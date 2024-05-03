package org.capstone.maru.config;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.buffer.MessageBuffer;
import org.capstone.maru.service.ViewCountService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;

@RequiredArgsConstructor
@Slf4j
@EnableScheduling
@Configuration
public class SchedulerConfig {

    private final ViewCountService viewCountService;
    private final MessageBuffer messageBuffer;

    @Scheduled(fixedDelay = 20 * 60000)
    public void syncViewCountToDataBase() {
        log.info("[Debug] 스케줄러 동작");
        viewCountService.syncToDatabase();
    }

    @Scheduled(fixedDelay = 30000)
    public void saveBufferedMessagesToDatabase() {
        log.info("saveBufferedMessagesToDatabase");
        messageBuffer.saveBufferedMessagesToDatabase();
    }

}
