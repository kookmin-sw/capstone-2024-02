package org.capstone.maru.service;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.BDDMockito.given;
import static org.mockito.BDDMockito.then;


import java.util.List;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.ScrapPostRepository;
import org.capstone.maru.repository.postgre.StudioRoomPostRepository;
import org.capstone.maru.util.EntityCreator;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;

@DisplayName("Service - 게시글")
@ExtendWith(MockitoExtension.class)
class SharedRoomPostServiceTest {

    @InjectMocks
    private SharedRoomPostService sut;

    @Mock
    private StudioRoomPostRepository studioRoomPostRepository;
    @Mock
    private MemberAccountRepository memberAccountRepository;
    @Mock
    private ScrapPostRepository scrapPostRepository;

    @DisplayName("[StudioRoomPost] 검색어와 필터링 조건 없이 검색하면, 모든 게시글들을 불러온다.")
    @Test
    void givenNothing_whenSearchingStudioRoomPost_thenReturnsStudioRoomPosts() throws Exception {
        // given
        String memberId = "test_1";
        String gender = "MALE";
        Pageable pageable = Pageable.ofSize(10);

        given(scrapPostRepository.findScrapViewByScrapperMemberId(memberId)).willReturn(List.of());
        given(studioRoomPostRepository.findAllByPublisherGender(gender, pageable)).willReturn(
            Page.empty());

        // when
        Page<StudioRoomPostDto> result = sut.searchStudioRoomPosts(memberId, gender,
            null, null, pageable);

        // then
        assertThat(result).isEmpty();
        then(scrapPostRepository).should().findScrapViewByScrapperMemberId(memberId);
        then(studioRoomPostRepository).should().findAllByPublisherGender(gender, pageable);
    }

    @DisplayName("[StudioRoomPost] 필터링 조건만 있이 검색하면, 필터링 조건에 맞는 모든 게시글들을 불러온다.")
    @Test
    void givenFilterCondition_whenSearchingStudioRoomPost_thenReturnsStudioRoomPosts()
        throws Exception {
        // given
        SearchFilterRequest filterCondition = SearchFilterRequest.fromJson(
            "{\"rentalType\": [0]}");
        StudioRoomPost studioRoomPost = EntityCreator.createStudioRoomPost(1);

        String memberId = "test_1";
        String gender = "MALE";
        Pageable pageable = Pageable.ofSize(10);

        given(scrapPostRepository.findScrapViewByScrapperMemberId(memberId)).willReturn(List.of());
        given(studioRoomPostRepository.findStudioRoomPostByDynamicFilter(gender, filterCondition,
            null, pageable)).willReturn(new PageImpl<>(List.of(studioRoomPost)));

        // when
        Page<StudioRoomPostDto> result = sut.searchStudioRoomPosts(memberId, gender,
            filterCondition, null, pageable);

        // then
        assertThat(result)
            .hasSize(1)
            .contains(StudioRoomPostDto.from(studioRoomPost, List.of()));
        then(scrapPostRepository).should().findScrapViewByScrapperMemberId(memberId);
        then(studioRoomPostRepository).should()
            .findStudioRoomPostByDynamicFilter(gender, filterCondition,
                null, pageable);
    }
}