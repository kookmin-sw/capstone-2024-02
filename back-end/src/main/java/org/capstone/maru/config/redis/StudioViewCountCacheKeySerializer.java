package org.capstone.maru.config.redis;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.SerializationException;

public class StudioViewCountCacheKeySerializer implements RedisSerializer<StudioViewCountCacheKey> {

    private static final Charset UTF_8 = StandardCharsets.UTF_8;

    @Override
    public byte[] serialize(StudioViewCountCacheKey value) throws SerializationException {
        if (Objects.isNull(value)) {
            throw new SerializationException("StudioPostViewCount key is null.");
        }

        return value.toString().getBytes(UTF_8);
    }

    @Override
    public StudioViewCountCacheKey deserialize(byte[] bytes) throws SerializationException {
        if (Objects.isNull(bytes)) {
            throw new SerializationException("bytes is null.");
        }

        return StudioViewCountCacheKey.from(new String(bytes, UTF_8));
    }
}
