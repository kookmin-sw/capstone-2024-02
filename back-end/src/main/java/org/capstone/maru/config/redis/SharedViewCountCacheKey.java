package org.capstone.maru.config.redis;

import java.util.Objects;
import lombok.Getter;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;

@Getter
public class SharedViewCountCacheKey {

    private static final String PREFIX = "shared:";

    private final Long postId;

    private SharedViewCountCacheKey(Long postId) {
        if (Objects.isNull(postId)) {
            throw new PostNotFoundException(RestErrorCode.POST_NOT_FOUND);
        }
        this.postId = postId;
    }

    public static SharedViewCountCacheKey from(String postId) {
        return new SharedViewCountCacheKey(Long.parseLong(postId));
    }

    public static SharedViewCountCacheKey from(Long postId) {
        return new SharedViewCountCacheKey(postId);
    }

    @Override
    public String toString() {
        return PREFIX + postId;
    }
}
