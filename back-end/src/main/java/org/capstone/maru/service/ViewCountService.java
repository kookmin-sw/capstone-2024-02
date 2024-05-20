package org.capstone.maru.service;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import org.capstone.maru.config.redis.SharedViewCountCacheKey;
import org.capstone.maru.domain.ViewPost;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.repository.postgre.ViewPostRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.core.ValueOperations;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Transactional
@Service
public class ViewCountService {

    private final ViewPostRepository viewPostRepository;

    private final RedisTemplate<String, String> stringRedisTemplate;
    private final ValueOperations<SharedViewCountCacheKey, Long> studioOperations;

    private static final String SHARED_PATTERN = "shared:*";

    public ViewCountService(
        @Autowired ViewPostRepository viewPostRepository,
        @Qualifier("sharedViewCountRedisTemplate") RedisTemplate<SharedViewCountCacheKey, Long> studioRedisTemplate,
        @Autowired RedisTemplate<String, String> redisTemplate
    ) {
        this.viewPostRepository = viewPostRepository;
        this.studioOperations = studioRedisTemplate.opsForValue();
        this.stringRedisTemplate = redisTemplate;
    }

    public void setValue(SharedViewCountCacheKey key, Long value) {
        studioOperations.set(key, value, Duration.ofMinutes(30L));
    }

    public Long increaseValue(SharedViewCountCacheKey key) {
        Long viewCount = studioOperations.get(key);

        if (viewCount == null) {
            ViewPost viewPost = viewPostRepository
                .findBySharedRoomPostId(key.getPostId())
                .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));
            viewCount = viewPost.getViewCount();
            setValue(key, viewCount);
        }

        return studioOperations.increment(key);
    }

    public Long getAndDeleteValue(SharedViewCountCacheKey key) {
        return studioOperations.getAndDelete(key);
    }

    public void syncToDatabase() {
        Set<Entry<Long, Long>> entries = getAllViewCount(SHARED_PATTERN).entrySet();

        for (Entry<Long, Long> entry : entries) {
            Optional<ViewPost> viewPost = viewPostRepository
                .findBySharedRoomPostId(entry.getKey());

            if (viewPost.isEmpty()) {
                continue;
            }

            viewPost.get().updateViewCount(entry.getValue());
        }
    }

    private Map<Long, Long> getAllViewCount(String pattern) {
        Set<String> keys = stringRedisTemplate.keys(pattern);

        Map<Long, Long> result = new HashMap<>();

        if (keys == null) {
            return result;
        }

        for (String key : keys) {
            // TODO: 안전하지 않은 코드
            Long postId = Long.parseLong(key.split(":")[1]);
            Long viewCount = getAndDeleteValue(SharedViewCountCacheKey.from(postId));

            if (viewCount == null) {
                continue;
            }

            result.put(postId, viewCount);
        }
        return result;
    }
}
