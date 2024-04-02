package org.capstone.maru.service;

import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDetailDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.repository.MemberAccountRepository;
import org.capstone.maru.repository.RoomImageRepository;
import org.capstone.maru.repository.StudioRoomPostRepository;
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
    private final RoomImageRepository roomImageRepository;
    private final MemberAccountRepository memberAccountRepository;

    @Transactional(readOnly = true)
    public Page<StudioRoomPostDto> searchStudioRoomPosts(
        String gender,
        SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        Pageable pageable
    ) {
        if (searchFilterRequest == null && !StringUtils.hasText(searchKeyWords)) {
            return studioRoomPostRepository
                .findAllByPublisherGender(gender, pageable)
                .map(StudioRoomPostDto::from);
        }

        if (searchFilterRequest == null) {
            return studioRoomPostRepository
                .findStudioRoomPostBySearchKeyWords(gender, searchKeyWords, pageable)
                .map(StudioRoomPostDto::from);
        }

        return studioRoomPostRepository
            .findStudioRoomPostByDynamicFilter(
                gender,
                searchFilterRequest,
                searchKeyWords,
                pageable
            )
            .map(StudioRoomPostDto::from);
    }

    @Transactional(readOnly = true)
    public StudioRoomPostDetailDto getStudioRoomPostDetail(Long postId, String gender) {
        return studioRoomPostRepository
            .findById(postId)
            .filter(studioRoomPost -> studioRoomPost.getPublisherGender().equals(gender))
            .map(StudioRoomPostDetailDto::from)
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
            .stream()
            .map(RoomImageDto::toEntityWithoutStudioRoomPost)
            .forEach(studioRoomPost::addRoomImage);

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
}
