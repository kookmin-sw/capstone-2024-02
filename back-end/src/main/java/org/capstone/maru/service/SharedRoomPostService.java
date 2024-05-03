package org.capstone.maru.service;

import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.config.redis.StudioViewCountCacheKey;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Participation;
import org.capstone.maru.domain.ScrapPost;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.ViewPost;
import org.capstone.maru.dto.MemberCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.ParticipationRepository;
import org.capstone.maru.repository.postgre.ScrapPostRepository;
import org.capstone.maru.repository.postgre.StudioRoomPostRepository;
import org.capstone.maru.repository.postgre.ViewPostRepository;
import org.capstone.maru.repository.postgre.projection.ScrapPostView;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

@Slf4j
@RequiredArgsConstructor
@Transactional
@Service
public class SharedRoomPostService {

    private final StudioRoomPostRepository studioRoomPostRepository;
    private final MemberAccountRepository memberAccountRepository;
    private final ParticipationRepository participationRepository;
    private final ScrapPostRepository scrapPostRepository;
    private final ViewPostRepository viewPostRepository;

    private final ViewCountService viewCountService;

    private static final String PREFIX = "studio";

    @Transactional(readOnly = true)
    public Page<StudioRoomPostDto> searchStudioRoomPosts(
        String memberId,
        String gender,
        SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        Pageable pageable
    ) {
        List<ScrapPostView> scrapPostViews = scrapPostRepository
            .findScrapViewByScrapperMemberId(memberId);

        if (searchFilterRequest == null && !StringUtils.hasText(searchKeyWords)) {
            return studioRoomPostRepository
                .findAllByPublisherGender(gender, pageable)
                .map(studioRoomPost ->
                    StudioRoomPostDto.from(
                        studioRoomPost,
                        scrapPostViews
                    )
                );
        }

        if (searchFilterRequest == null) {
            return studioRoomPostRepository
                .findStudioRoomPostBySearchKeyWords(gender, searchKeyWords, pageable)
                .map(studioRoomPost ->
                    StudioRoomPostDto.from(
                        studioRoomPost,
                        scrapPostViews
                    )
                );
        }

        return studioRoomPostRepository
            .findStudioRoomPostByDynamicFilter(
                gender,
                searchFilterRequest,
                searchKeyWords,
                pageable
            )
            .map(studioRoomPost ->
                StudioRoomPostDto.from(
                    studioRoomPost,
                    scrapPostViews
                )
            );
    }

    @Transactional(readOnly = true)
    public StudioRoomPostDetailDto getStudioRoomPostDetail(String memberId, Long postId,
        String gender) {
        StudioRoomPost resultEntity = studioRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        // 스크랩 여부와 스크랩 개수
        final Boolean isScrapped = scrapPostRepository
            .findScrapViewByScrappedIdAndScrapperMemberId(postId, memberId)
            .map(ScrapPostView::getIsScrapped)
            .orElse(false);
        final Long scrapCount = scrapPostRepository.countByScrappedIdAndIsScrapped(postId);

        // 조회수 +1 & 게시글 총 조회수
        Long viewCount = viewCountService.increaseValue(StudioViewCountCacheKey.from(postId));

        return StudioRoomPostDetailDto.from(resultEntity, isScrapped, scrapCount, viewCount);
    }

    public void saveStudioRoomPost(
        String publisherMemberId,
        StudioRoomPostDto studioRoomPostDto,
        MemberCardDto roomMateCardDto,
        List<String> participationMemberIds,
        List<RoomImageDto> roomImagesDto,
        RoomInfoDto roomInfoDto
    ) {
        MemberAccount publisherAccount = memberAccountRepository.getReferenceById(
            publisherMemberId);

        StudioRoomPost studioRoomPost = studioRoomPostDto.toEntity(
            roomMateCardDto.toEntity(), publisherAccount, roomInfoDto.toEntity()
        );

        participationMemberIds
            .stream()
            .map(memberAccountRepository::getReferenceById)
            .forEach(memberAccount -> Participation.of(memberAccount, studioRoomPost));

        roomImagesDto
            .forEach(roomImageDto -> roomImageDto.toEntity(studioRoomPost));

        StudioRoomPost result = studioRoomPostRepository.save(
            studioRoomPost
        );

        viewPostRepository.save(ViewPost.of(result.getId(), 0L));
        viewCountService.setValue(StudioViewCountCacheKey.from(result.getId()), 0L);
    }

    public void deleteStudioRoomPost(
        Long postId,
        String memberId
    ) {
        studioRoomPostRepository.deleteByIdAndAndPublisherAccount_MemberId(postId, memberId);
    }

    public void scrapStudioRoomPost(
        String memberId,
        String gender,
        Long postId
    ) {
        MemberAccount memberAccount = memberAccountRepository.getReferenceById(memberId);
        StudioRoomPost studioRoomPost = studioRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .orElseThrow(() -> new IllegalArgumentException("게시물이 존재하지 않습니다."));

        Optional<ScrapPost> scrapPost = scrapPostRepository.findByScrappedIdAndScrapperMemberId(
            postId, memberId);

        if (scrapPost.isEmpty()) {
            scrapPostRepository.save(ScrapPost.of(memberAccount, studioRoomPost));
            return;
        }

        scrapPost.get().toggleScrap();
    }

}
