package org.capstone.maru.config.redis;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.SerializationException;

public class SharedViewCountCacheKeySerializer implements RedisSerializer<SharedViewCountCacheKey> {

    private static final Charset UTF_8 = StandardCharsets.UTF_8;

    @Override
    public byte[] serialize(SharedViewCountCacheKey value) throws SerializationException {
        if (Objects.isNull(value)) {
            throw new SerializationException("SharedPostViewCount key is null.");
        }

        return value.toString().getBytes(UTF_8);
    }

    @Override
    public SharedViewCountCacheKey deserialize(byte[] bytes) throws SerializationException {
        if (Objects.isNull(bytes)) {
            throw new SerializationException("bytes is null.");
        }

        return SharedViewCountCacheKey.from(new String(bytes, UTF_8));
    }
}
