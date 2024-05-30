package org.capstone.maru.controller;

import jakarta.validation.Valid;
import java.util.List;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.annotation.RequestQueryString;
import org.capstone.maru.dto.DormitoryRoomPostDto;
import org.capstone.maru.dto.FeatureCardDto;
import org.capstone.maru.dto.RoomImageDto;
import org.capstone.maru.dto.RoomInfoDto;
import org.capstone.maru.dto.StudioRoomPostDto;
import org.capstone.maru.dto.request.DormitoryRoomPostRequest;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.capstone.maru.dto.request.StudioRoomPostRequest;
import org.capstone.maru.dto.response.APIResponse;
import org.capstone.maru.dto.response.DormitoryRoomPostDetailResponse;
import org.capstone.maru.dto.response.DormitoryRoomPostResponse;
import org.capstone.maru.dto.response.DormitoryRoomRecommendPostResponse;
import org.capstone.maru.dto.response.StudioRoomPostDetailResponse;
import org.capstone.maru.dto.response.StudioRoomPostResponse;
import org.capstone.maru.dto.response.StudioRoomRecommendPostResponse;
import org.capstone.maru.security.principal.MemberPrincipal;
import org.capstone.maru.service.DormitoryRoomPostService;
import org.capstone.maru.service.StudioRoomPostService;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort.Direction;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RestController
@RequestMapping("/shared/posts")
public class SharedRoomPostController {

    private final StudioRoomPostService studioRoomPostService;
    private final DormitoryRoomPostService dormitoryRoomPostService;

    @GetMapping("/studio")
    public ResponseEntity<APIResponse> studioRoomPosts(
        @AuthenticationPrincipal MemberPrincipal principal,
        @RequestQueryString(name = "filter", required = false) SearchFilterRequest searchFilterRequest,
        @RequestParam(name = "search", required = false) String searchKeyWords,
        @RequestParam(name = "cardOption") String cardOption,
        @PageableDefault(size = 10, sort = "score", direction = Direction.DESC) Pageable pageable
    ) {
        Page<StudioRoomRecommendPostResponse> result = studioRoomPostService
            .searchStudioRoomPosts(
                principal.memberId(),
                principal.gender(),
                searchFilterRequest,
                searchKeyWords,
                cardOption,
                pageable
            )
            .map(StudioRoomRecommendPostResponse::from);

        log.info("result size : {}", result.getSize());

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @GetMapping("/studio/{postId}")
    public ResponseEntity<APIResponse> studioRoomPostDetail(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        StudioRoomPostDetailResponse result = StudioRoomPostDetailResponse.from(
            studioRoomPostService.getStudioRoomPostDetail(
                principal.memberId(),
                postId,
                principal.gender()
            )
        );

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PostMapping("/studio")
    public void postNewStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @Valid @RequestBody StudioRoomPostRequest studioRoomPostRequest
    ) {
        StudioRoomPostDto studioRoomPostDto = studioRoomPostRequest.toBaseStudioRoomPostDto(
            principal.gender()
        );
        FeatureCardDto roomMateCardDto = studioRoomPostRequest.toMemberCardDto();
        List<RoomImageDto> roomImagesDto = studioRoomPostRequest.toRoomImagesDto();
        RoomInfoDto roomInfoDto = studioRoomPostRequest.toRoomInfoDto();
        List<String> participationMemberIds = studioRoomPostRequest
            .participationData()
            .participationMemberIds();

        studioRoomPostService.saveStudioRoomPost(
            principal.memberId(),
            studioRoomPostDto,
            roomMateCardDto,
            participationMemberIds,
            roomImagesDto,
            roomInfoDto
        );
    }

    @PutMapping("/studio/{postId}")
    public void updateStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId,
        @Valid @RequestBody StudioRoomPostRequest studioRoomPostRequest
    ) {
        StudioRoomPostDto studioRoomPostDto = studioRoomPostRequest.toBaseStudioRoomPostDto(
            principal.gender()
        );
        FeatureCardDto roomMateCardDto = studioRoomPostRequest.toMemberCardDto();
        List<RoomImageDto> roomImagesDto = studioRoomPostRequest.toRoomImagesDto();
        RoomInfoDto roomInfoDto = studioRoomPostRequest.toRoomInfoDto();
        List<String> participationMemberIds = studioRoomPostRequest
            .participationData()
            .participationMemberIds();

        studioRoomPostService.updateStudioRoomPost(
            postId,
            principal.memberId(),
            studioRoomPostDto,
            roomMateCardDto,
            participationMemberIds,
            roomImagesDto,
            roomInfoDto
        );
    }

    @DeleteMapping("/studio/{postId}")
    public void deleteStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        studioRoomPostService.deleteStudioRoomPost(postId, principal.memberId());
    }

    @PostMapping("/studio/{postId}/scrap")
    public ResponseEntity<APIResponse> scrapStudioRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        studioRoomPostService.scrapStudioRoomPost(principal.memberId(), principal.gender(), postId);

        return ResponseEntity.ok(APIResponse.success());
    }

    @GetMapping("/dormitory")
    public ResponseEntity<APIResponse> dormitoryRoomPosts(
        @AuthenticationPrincipal MemberPrincipal principal,
        @RequestParam(name = "search", required = false) String searchKeyWords,
        @RequestParam(name = "cardOption") String cardOption,
        @PageableDefault(size = 10, sort = "score", direction = Direction.DESC) Pageable pageable
    ) {
        Page<DormitoryRoomRecommendPostResponse> result = dormitoryRoomPostService
            .searchDormitoryRoomPosts(
                principal.memberId(),
                principal.gender(),
                searchKeyWords,
                cardOption,
                pageable
            )
            .map(DormitoryRoomRecommendPostResponse::from);

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @GetMapping("/dormitory/{postId}")
    public ResponseEntity<APIResponse> dormitoryRoomPostDetail(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        DormitoryRoomPostDetailResponse result = DormitoryRoomPostDetailResponse.from(
            dormitoryRoomPostService.getDormitoryRoomPostDetail(
                principal.memberId(),
                postId,
                principal.gender()
            )
        );

        return ResponseEntity.ok(APIResponse.success(result));
    }

    @PostMapping("/dormitory")
    public void postNewDormitoryRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @Valid @RequestBody DormitoryRoomPostRequest dormitoryRoomPostRequest
    ) {
        DormitoryRoomPostDto dormitoryRoomPostDto = dormitoryRoomPostRequest.toBaseDormitoryRoomPostDto(
            principal.gender()
        );
        FeatureCardDto roomMateCardDto = dormitoryRoomPostRequest.toMemberCardDto();
        List<RoomImageDto> roomImagesDto = dormitoryRoomPostRequest.toRoomImagesDto();
        List<String> participationMemberIds = dormitoryRoomPostRequest
            .participationData()
            .participationMemberIds();

        dormitoryRoomPostService.saveDormitoryRoomPost(
            principal.memberId(),
            dormitoryRoomPostDto,
            roomMateCardDto,
            participationMemberIds,
            roomImagesDto
        );
    }

    @PutMapping("/dormitory/{postId}")
    public void updateDormitoryRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId,
        @Valid @RequestBody DormitoryRoomPostRequest dormitoryRoomPostRequest
    ) {
        DormitoryRoomPostDto dormitoryRoomPostDto = dormitoryRoomPostRequest.toBaseDormitoryRoomPostDto(
            principal.gender()
        );
        FeatureCardDto roomMateCardDto = dormitoryRoomPostRequest.toMemberCardDto();
        List<RoomImageDto> roomImagesDto = dormitoryRoomPostRequest.toRoomImagesDto();
        List<String> participationMemberIds = dormitoryRoomPostRequest
            .participationData()
            .participationMemberIds();

        dormitoryRoomPostService.updateDormitoryRoomPost(
            postId,
            principal.memberId(),
            dormitoryRoomPostDto,
            roomMateCardDto,
            participationMemberIds,
            roomImagesDto
        );
    }

    @DeleteMapping("/dormitory/{postId}")
    public void deleteDormitoryRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        dormitoryRoomPostService.deleteDormitoryRoomPost(postId, principal.memberId());
    }

    @PostMapping("/dormitory/{postId}/scrap")
    public ResponseEntity<APIResponse> scrapDormitoryRoomPost(
        @AuthenticationPrincipal MemberPrincipal principal,
        @PathVariable("postId") Long postId
    ) {
        dormitoryRoomPostService.scrapDormitoryRoomPost(principal.memberId(), principal.gender(),
            postId);

        return ResponseEntity.ok(APIResponse.success());
    }

}
