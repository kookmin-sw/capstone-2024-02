package org.capstone.maru.buffer;

import java.util.concurrent.ConcurrentLinkedQueue;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Chat;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

@RequiredArgsConstructor
@Component
@Slf4j
public class MessageBuffer {

    private final MongoTemplate mongoTemplate;

    private long lastMessageReceivedTime = System.currentTimeMillis();
    private static final int BUFFER_SIZE = 100; // 버퍼 사이즈
    private static final long SAVE_INTERVAL_MS = 30000; // 저장 간격 (밀리초) 1초

    private final ConcurrentLinkedQueue<Chat> messageBuffer = new ConcurrentLinkedQueue<>();

    // 비동기 적으로 메시지를 저장하는 로직
    @Async
    public void addMessage(Chat chat) {
        // 이곳에 redis에 메시지를 저장하는 로직이 들어갈 수 있음
        messageBuffer.add(chat);
    }

    // 최근 채팅은 redis에서 긁어오기
    public void saveBufferedMessagesToDatabase() {
        
        messageBuffer.forEach(
            chat -> {
                log.info("chat : {}", chat);
            }
        );

        long currentTime = System.currentTimeMillis();
        // 버퍼에 있는 메시지를 일괄 저장하는 로직
        int currentBufferSize = messageBuffer.size();

        if (currentBufferSize == 0) {
            return;
        }

        if (currentBufferSize > BUFFER_SIZE) {
            // 1000개 이상 메시지가 쌓이면 데이터베이스에 저장
            // bulk insert 의 경우 한번에 여러개의 데이터를 저장하기 때문에 속도가 빠름, 8,388,608 문자를 저장할 수 있다.
            // 이때 발생하는 문제는 오랬동안 안찼을때
            log.info("사이즈 넘어서 저장");
            mongoTemplate.insertAll(messageBuffer);
            messageBuffer.clear();
            return;
        }

        if (currentTime - lastMessageReceivedTime >= SAVE_INTERVAL_MS * 3) {
            // 버퍼가 차지 않아도 3분이 지났을 경우 데이터베이스에 저장
            log.info("시간 넘어서 저장");
            mongoTemplate.insertAll(messageBuffer);
            messageBuffer.clear();
            lastMessageReceivedTime = currentTime;
        }
    }
//
//    public void removeMessages(Long chatId) {
//        messageBuffer.removeIf(chat -> chat.getId().equals(chatId));
//    }
}
