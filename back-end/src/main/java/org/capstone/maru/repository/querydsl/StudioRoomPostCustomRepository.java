package org.capstone.maru.repository.querydsl;

import jakarta.annotation.Nonnull;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

public interface StudioRoomPostCustomRepository {

    Page<StudioRoomPost> findStudioRoomPostByDynamicFilter(
        String gender,
        @Nonnull SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        Pageable pageable
    );

    Page<StudioRoomPost> findStudioRoomPostBySearchKeyWords(
        String gender,
        String searchKeyWords,
        Pageable pageable
    );
}
