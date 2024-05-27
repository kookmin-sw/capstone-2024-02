package org.capstone.maru.service;

import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.config.redis.SharedViewCountCacheKey;
import org.capstone.maru.domain.DormitoryRoomPost;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.Participation;
import org.capstone.maru.domain.ScrapPost;
import org.capstone.maru.domain.ViewPost;
import org.capstone.maru.dto.DormitoryRoomPostDetailDto;
import org.capstone.maru.dto.DormitoryRoomPostDto;
import org.capstone.maru.dto.DormitoryRoomRecommendPostDto;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.exception.PostNotFoundException;
import org.capstone.maru.exception.RestErrorCode;
import org.capstone.maru.repository.postgre.DormitoryRoomPostRepository;
import org.capstone.maru.repository.postgre.FollowRepository;
import org.capstone.maru.repository.postgre.MemberAccountRepository;
import org.capstone.maru.repository.postgre.ScrapPostRepository;
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
public class DormitoryRoomPostService {

    private final ViewPostRepository viewPostRepository;
    private final DormitoryRoomPostRepository dormitoryRoomPostRepository;
    private final MemberAccountRepository memberAccountRepository;
    private final ScrapPostRepository scrapPostRepository;
    private final FollowRepository followRepository;

    private final S3FileService s3FileService;
    private final ViewCountService viewCountService;
    private final RecommendService recommendService;

    @Transactional(readOnly = true)
    public Page<DormitoryRoomRecommendPostDto> searchDormitoryRoomPosts(
        String memberId,
        String gender,
        String searchKeyWords,
        String cardOption,
        Pageable pageable
    ) {
        recommendService.updateRecommendation(
            memberId,
            cardOption,
            "post"
        ).subscribe();

        List<ScrapPostView> scrapPostViews = scrapPostRepository
            .findScrapViewByScrapperMemberId(memberId);

        if (!StringUtils.hasText(searchKeyWords)) {
            return dormitoryRoomPostRepository
                .findDormitoryRoomPostByPublisherGender(
                    memberId, gender, cardOption, pageable
                )
                .map(dormitoryRoomPost -> {
                    dormitoryRoomPost
                        .getRoomImages()
                        .forEach(roomImage ->
                            roomImage
                                .updateFileName(
                                    s3FileService.getPreSignedUrlForLoad(roomImage.getFileName())
                                )
                        );
                    dormitoryRoomPost
                        .getPublisherAccount()
                        .getProfileImage()
                        .updateFileName(
                            dormitoryRoomPost
                                .getPublisherAccount()
                                .getProfileImage()
                                .getFileName()
                        );
                    return DormitoryRoomRecommendPostDto.from(
                        dormitoryRoomPost,
                        scrapPostViews
                    );
                });
        }

        return dormitoryRoomPostRepository
            .findDormitoryRoomPostBySearchKeyWords(memberId, gender, searchKeyWords, cardOption,
                pageable)
            .map(dormitoryRoomPost -> {
                    dormitoryRoomPost
                        .getRoomImages()
                        .forEach(roomImage ->
                            roomImage
                                .updateFileName(
                                    s3FileService.getPreSignedUrlForLoad(roomImage.getFileName())
                                )
                        );
                    dormitoryRoomPost
                        .getPublisherAccount()
                        .getProfileImage()
                        .updateFileName(
                            dormitoryRoomPost
                                .getPublisherAccount()
                                .getProfileImage()
                                .getFileName()
                        );
                    return DormitoryRoomRecommendPostDto.from(
                        dormitoryRoomPost,
                        scrapPostViews
                    );
                }
            );
    }

    @Transactional(readOnly = true)
    public DormitoryRoomPostDetailDto getDormitoryRoomPostDetail(String memberId, Long postId,
        String gender) {
        DormitoryRoomPost resultEntity = dormitoryRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        final Boolean isScrapped = scrapPostRepository
            .findScrapViewByScrappedIdAndScrapperMemberId(postId, memberId)
            .map(ScrapPostView::getIsScrapped)
            .orElse(false);
        final Long scrapCount = scrapPostRepository.countByScrappedIdAndIsScrapped(postId);

        List<String> followingIds = followRepository.findFollowingIdsByFollowerId(memberId);

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
                    s3FileService.getPreSignedUrlForLoad(roomImage.getFileName())
                )
            );
        return DormitoryRoomPostDetailDto.from(resultEntity, isScrapped, followingIds,
            scrapCount, viewCount);
    }

    public void saveDormitoryRoomPost(
        String publisherMemberId,
        DormitoryRoomPostDto dormitoryRoomPostDto,
        FeatureCardDto roomMateCardDto,
        List<String> participationMemberIds,
        List<RoomImageDto> roomImagesDto
    ) {
        MemberAccount publisherAccount = memberAccountRepository.getReferenceById(
            publisherMemberId);

        DormitoryRoomPost dormitoryRoomPost = dormitoryRoomPostDto.toEntity(
            roomMateCardDto.toEntity(), publisherAccount
        );
        dormitoryRoomPost.addRoomImages(roomImagesDto);

        participationMemberIds
            .stream()
            .map(memberAccountRepository::getReferenceById)
            .forEach(memberAccount -> Participation.of(memberAccount, dormitoryRoomPost));

        DormitoryRoomPost result = dormitoryRoomPostRepository.save(dormitoryRoomPost);

        viewPostRepository.save(ViewPost.of(result.getId(), 0L));
        viewCountService.setValue(SharedViewCountCacheKey.from(result.getId()), 0L);
    }

    public void updateDormitoryRoomPost(
        Long postId,
        String publisherMemberId,
        DormitoryRoomPostDto dormitoryRoomPostDto,
        FeatureCardDto roomMateCardDto,
        List<String> participationMemberIds,
        List<RoomImageDto> roomImagesDto
    ) {
        DormitoryRoomPost dormitoryRoomPost = dormitoryRoomPostRepository
            .findByIdAndPublisherAccount_MemberId(postId, publisherMemberId)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        dormitoryRoomPost.updateDormitoryRoomPost(
            dormitoryRoomPostDto,
            roomMateCardDto,
            roomImagesDto
        );

        dormitoryRoomPost.initParticipants();
        participationMemberIds
            .stream()
            .map(memberAccountRepository::getReferenceById)
            .forEach(memberAccount -> Participation.of(memberAccount, dormitoryRoomPost));

        dormitoryRoomPostRepository.save(dormitoryRoomPost);
    }

    public void deleteDormitoryRoomPost(Long postId, String memberId) {
        dormitoryRoomPostRepository.deleteByIdAndPublisherAccount_MemberId(postId, memberId);
    }

    public void scrapDormitoryRoomPost(
        String memberId,
        String gender,
        Long postId
    ) {
        MemberAccount memberAccount = memberAccountRepository.getReferenceById(memberId);
        DormitoryRoomPost dormitoryRoomPost = dormitoryRoomPostRepository
            .findByIdAndPublisherGender(postId, gender)
            .orElseThrow(() -> new PostNotFoundException(RestErrorCode.POST_NOT_FOUND));

        Optional<ScrapPost> scrapPost = scrapPostRepository.findByScrappedIdAndScrapperMemberId(
            postId, memberId);

        if (scrapPost.isEmpty()) {
            scrapPostRepository.save(ScrapPost.of(memberAccount, dormitoryRoomPost));
            return;
        }

        scrapPost.get().toggleScrap();
    }
}
