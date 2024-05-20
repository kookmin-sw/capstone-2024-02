package org.capstone.maru.repository.postgre.querydsl;

import jakarta.annotation.Nonnull;
import org.capstone.maru.domain.DormitoryRoomPost;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.dto.DormitoryRoomRecommendPost;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

public interface DormitoryRoomPostCustomRepository {

    Page<DormitoryRoomRecommendPost> findDormitoryRoomPostByPublisherGender(
        String memberId,
        String gender,
        String cardOption,
        Pageable pageable
    );

    Page<DormitoryRoomRecommendPost> findDormitoryRoomPostBySearchKeyWords(
        String memberId,
        String gender,
        String searchKeyWords,
        String cardOption,
        Pageable pageable
    );
}
