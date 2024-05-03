package org.capstone.maru.config.redis;

import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import org.springframework.data.redis.serializer.RedisSerializer;
import org.springframework.data.redis.serializer.SerializationException;

public class LongRedisSerializer implements RedisSerializer<Long> {

    public static final Gson gson = new Gson();
    private final Charset UTF_8 = StandardCharsets.UTF_8;

    @Override
    public byte[] serialize(Long value) throws SerializationException {
        if (Objects.isNull(value)) {
            return null;
        }

        try {
            String json = gson.toJson(value, Long.class);
            return json.getBytes(UTF_8);
        } catch (JsonIOException e) {
            throw new SerializationException("json serialize error", e);
        }
    }

    @Override
    public Long deserialize(byte[] bytes) throws SerializationException {
        if (Objects.isNull(bytes)) {
            return null;
        }

        try {
            return gson.fromJson(new String(bytes, UTF_8), Long.class);
        } catch (JsonSyntaxException e) {
            throw new SerializationException("json deserialize error", e);
        }
    }
}
