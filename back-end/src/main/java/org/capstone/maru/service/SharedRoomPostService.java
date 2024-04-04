package org.capstone.maru.service;

import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.ScrapPost;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.repository.MemberAccountRepository;
import org.capstone.maru.repository.RoomImageRepository;
import org.capstone.maru.repository.ScrapPostRepository;
import org.capstone.maru.repository.StudioRoomPostRepository;
import org.capstone.maru.repository.projection.ScrapPostView;
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
    private final RoomImageRepository roomImageRepository;
    private final ScrapPostRepository scrapPostRepository;

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
        final Boolean isScrapped = scrapPostRepository
            .findScrapViewByScrappedIdAndScrapperMemberId(postId, memberId)
            .map(ScrapPostView::getIsScrapped)
            .orElse(false);

        final Long scrapCount = scrapPostRepository.countByScrappedIdAndIsScrapped(postId);

        return studioRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .map(studioRoomPost -> StudioRoomPostDetailDto
                .from(studioRoomPost, isScrapped, scrapCount)
            )
            .orElseThrow(() -> new IllegalArgumentException("그런 게시물은 존재하지 않습니다."));
    }

    public void saveStudioRoomPost(
        String publisherMemberId,
        StudioRoomPostDto studioRoomPostDto,
        List<RoomImageDto> roomImagesDto,
        RoomInfoDto roomInfoDto
    ) {
        MemberAccount publisherAccount = memberAccountRepository.getReferenceById(
            publisherMemberId);

        StudioRoomPost studioRoomPost = studioRoomPostDto.toEntity(
            publisherAccount, roomInfoDto.toEntity()
        );

        roomImagesDto
            .forEach(roomImageDto -> roomImageDto.toEntity(studioRoomPost));

        studioRoomPostRepository.save(
            studioRoomPost
        );
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
