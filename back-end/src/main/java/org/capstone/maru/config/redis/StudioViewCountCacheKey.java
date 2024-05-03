package org.capstone.maru.config.redis;

import java.util.Objects;
import lombok.Getter;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;

@Getter
public class StudioViewCountCacheKey {

    private static final String PREFIX = "studio:";

    private final Long postId;

    private StudioViewCountCacheKey(Long postId) {
        if (Objects.isNull(postId)) {
            throw new PostNotFoundException(RestErrorCode.POST_NOT_FOUND);
        }
        this.postId = postId;
    }

    public static StudioViewCountCacheKey from(String postId) {
        return new StudioViewCountCacheKey(Long.parseLong(postId));
    }

    public static StudioViewCountCacheKey from(Long postId) {
        return new StudioViewCountCacheKey(postId);
    }

    @Override
    public String toString() {
        return PREFIX + postId;
    }
}
