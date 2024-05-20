package org.capstone.maru.service;

import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.config.redis.SharedViewCountCacheKey;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Participation;
import org.capstone.maru.domain.ScrapPost;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.ViewPost;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.StudioRoomRecommendPost;
import org.capstone.maru.dto.StudioRoomRecommendPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.repository.postgre.FollowRepository;
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
public class StudioRoomPostService {

    private final StudioRoomPostRepository studioRoomPostRepository;
    private final MemberAccountRepository memberAccountRepository;
    private final ParticipationRepository participationRepository;
    private final ScrapPostRepository scrapPostRepository;
    private final ViewPostRepository viewPostRepository;
    private final FollowRepository followRepository;

    private final ViewCountService viewCountService;
    private final RecommendService recommendService;
    private final S3FileService s3FileService;

    @Transactional(readOnly = true)
    public Page<StudioRoomRecommendPostDto> searchStudioRoomPosts(
        String memberId,
        String gender,
        SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        String cardOption,
        Pageable pageable
    ) {
        recommendService.updateRecommendation(
            memberId,
            cardOption,
            "post"
        ).block();

        List<ScrapPostView> scrapPostViews = scrapPostRepository
            .findScrapViewByScrapperMemberId(memberId);

        if (searchFilterRequest == null && !StringUtils.hasText(searchKeyWords)) {
            log.info("둘다 없음");
            Page<StudioRoomRecommendPost> resultPage = studioRoomPostRepository
                .findAllRecommendByPublisherGender(memberId, gender, cardOption, pageable);

            log.info("resultPage: {}", resultPage);

            return resultPage
                .map(studioRoomPost -> {
                    studioRoomPost
                        .getRoomImages()
                        .forEach(roomImage ->
                            roomImage
                                .updateFileName(
                                    s3FileService
                                        .getPreSignedUrlForLoad(roomImage.getFileName())
                                )
                        );

                    studioRoomPost
                        .getPublisherAccount()
                        .getProfileImage()
                        .updateFileName(
                            s3FileService.getPreSignedUrlForLoad(
                                studioRoomPost.getPublisherAccount().getProfileImage().getFileName()
                            ));

                    return StudioRoomRecommendPostDto.from(
                        studioRoomPost,
                        scrapPostViews
                    );
                });
        }

        if (searchFilterRequest == null) {
            log.info("searchKeyWords만 있음");
            Page<StudioRoomRecommendPost> resultPage = studioRoomPostRepository
                .findStudioRoomRecommendPostBySearchKeyWords(
                    memberId,
                    gender,
                    searchKeyWords,
                    cardOption,
                    pageable
                );

            return resultPage.map(studioRoomPost -> {
                    studioRoomPost
                        .getRoomImages()
                        .forEach(roomImage ->
                            roomImage
                                .updateFileName(
                                    s3FileService
                                        .getPreSignedUrlForLoad(roomImage.getFileName())
                                )
                        );
                    studioRoomPost
                        .getPublisherAccount()
                        .getProfileImage()
                        .updateFileName(
                            s3FileService.getPreSignedUrlForLoad(
                                studioRoomPost.getPublisherAccount().getProfileImage().getFileName()
                            ));
                    return StudioRoomRecommendPostDto.from(
                        studioRoomPost,
                        scrapPostViews
                    );
                }
            );
        }

        log.info("둘다 있음");

        Page<StudioRoomRecommendPost> resultPage = studioRoomPostRepository
            .findStudioRoomPostByRecommendDynamicFilter(
                gender, searchFilterRequest, searchKeyWords, memberId, cardOption, pageable
            );

        return resultPage.map(studioRoomPost -> {
                studioRoomPost
                    .getRoomImages()
                    .forEach(roomImage ->
                        roomImage
                            .updateFileName(
                                s3FileService
                                    .getPreSignedUrlForLoad(roomImage.getFileName())
                            )
                    );
                studioRoomPost
                    .getPublisherAccount()
                    .getProfileImage()
                    .updateFileName(
                        s3FileService.getPreSignedUrlForLoad(
                            studioRoomPost.getPublisherAccount().getProfileImage().getFileName()
                        ));
                return StudioRoomRecommendPostDto.from(
                    studioRoomPost,
                    scrapPostViews
                );
            }
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
        List<String> scrappedMemberIds = followRepository.findFollowingIdsByFollowerId(memberId);

        // 조회수 +1 & 게시글 총 조회수
        Long viewCount = viewCountService.increaseValue(SharedViewCountCacheKey.from(postId));

        resultEntity
            .getSharedRoomPostRecruits()
            .stream()
            .map(Participation::getRecruitedMemberAccount)
            .map(MemberAccount::getProfileImage)
            .forEach(
                profileImage -> profileImage.updateFileName(
                    s3FileService.getPreSignedUrlForLoad(profileImage.getFileName())
                )
            );
        resultEntity
            .getRoomImages()
            .forEach(
                roomImage -> roomImage.updateFileName(
                    s3FileService
                        .getPreSignedUrlForLoad(roomImage.getFileName())
                )
            );
        return StudioRoomPostDetailDto.from(resultEntity, isScrapped, scrappedMemberIds, scrapCount,
            viewCount);
    }

    public void saveStudioRoomPost(
        String publisherMemberId,
        StudioRoomPostDto studioRoomPostDto,
        FeatureCardDto roomMateCardDto,
        List<String> participationMemberIds,
        List<RoomImageDto> roomImagesDto,
        RoomInfoDto roomInfoDto
    ) {
        MemberAccount publisherAccount = memberAccountRepository.getReferenceById(
            publisherMemberId);

        StudioRoomPost studioRoomPost = studioRoomPostDto.toEntity(
            roomMateCardDto.toEntity(), publisherAccount, roomInfoDto.toEntity()
        );
        studioRoomPost.addRoomImages(roomImagesDto);

        participationMemberIds
            .stream()
            .map(memberAccountRepository::getReferenceById)
            .forEach(memberAccount -> Participation.of(memberAccount, studioRoomPost));

        StudioRoomPost result = studioRoomPostRepository.save(
            studioRoomPost
        );

        viewPostRepository.save(ViewPost.of(result.getId(), 0L));
        viewCountService.setValue(SharedViewCountCacheKey.from(result.getId()), 0L);
    }

    public void updateStudioRoomPost(
        Long postId,
        String publisherMemberId,
        StudioRoomPostDto studioRoomPostDto,
        FeatureCardDto roomMateCardDto,
        List<String> participationMemberIds,
        List<RoomImageDto> roomImagesDto,
        RoomInfoDto roomInfoDto
    ) {
        StudioRoomPost studioRoomPost = studioRoomPostRepository
            .findByIdAndPublisherAccount_MemberId(postId, publisherMemberId)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        studioRoomPost.updateStudioRoomPost(
            studioRoomPostDto,
            roomMateCardDto,
            roomInfoDto,
            roomImagesDto
        );

        // TODO: 별로 좋은 코드 같지 않음
        studioRoomPost.initParticipants();
        participationMemberIds
            .stream()
            .map(memberAccountRepository::getReferenceById)
            .forEach(memberAccount -> Participation.of(memberAccount, studioRoomPost));

        studioRoomPostRepository.save(studioRoomPost);
    }

    public void deleteStudioRoomPost(
        Long postId,
        String memberId
    ) {
        studioRoomPostRepository.deleteByIdAndPublisherAccount_MemberId(postId, memberId);
    }

    public void scrapStudioRoomPost(
        String memberId,
        String gender,
        Long postId
    ) {
        MemberAccount memberAccount = memberAccountRepository.getReferenceById(memberId);
        StudioRoomPost studioRoomPost = studioRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        Optional<ScrapPost> scrapPost = scrapPostRepository.findByScrappedIdAndScrapperMemberId(
            postId, memberId);

        if (scrapPost.isEmpty()) {
            scrapPostRepository.save(ScrapPost.of(memberAccount, studioRoomPost));
            return;
        }

        scrapPost.get().toggleScrap();
    }

}
