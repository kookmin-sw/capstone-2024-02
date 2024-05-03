package org.capstone.maru.security.token;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.annotation.Id;
import org.springframework.data.redis.core.RedisHash;
import org.springframework.data.redis.core.TimeToLive;
import org.springframework.data.redis.core.index.Indexed;

@Slf4j
@Getter
@RedisHash(value = "refresh_token")
public class RefreshToken {

    @Id
    private Long id;

    @Indexed
    private String token;

    @TimeToLive
    private Long ttl;

    public RefreshToken(String token, Long ttl) {
        this.token = token;
        this.ttl = ttl;
    }
}
