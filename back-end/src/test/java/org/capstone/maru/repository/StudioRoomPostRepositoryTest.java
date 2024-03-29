package org.capstone.maru.repository;

import static org.assertj.core.api.Assertions.assertThat;

import com.github.gavlyukovskiy.boot.jdbc.decorator.DataSourceDecoratorAutoConfiguration;
import java.time.LocalDate;
import java.util.List;
import org.capstone.maru.config.P6spyConfig;
import org.capstone.maru.config.TestJpaConfig;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.constant.Gender;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.ImportAutoConfiguration;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase.Replace;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.context.annotation.Import;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.test.context.TestPropertySource;

@Disabled("initial dummy data 개발 중")
@DisplayName("JPA StudioRoomPostRepository 테스트")
@Import({TestJpaConfig.class, P6spyConfig.class})
@DataJpaTest(showSql = false)
@TestPropertySource(locations = "classpath:application-test.yaml")
@ImportAutoConfiguration(DataSourceDecoratorAutoConfiguration.class)
@AutoConfigureTestDatabase(replace = Replace.NONE)
class StudioRoomPostRepositoryTest {

    private final StudioRoomPostRepository studioRoomPostRepository;

    private final int PAGE_NUMBER = 0;
    private final int PAGE_SIZE = 10;
    private final Gender MALE = Gender.MALE;
    private final Gender FEMALE = Gender.FEMALE;
    private Pageable pageable;


    public StudioRoomPostRepositoryTest(
        @Autowired StudioRoomPostRepository studioRoomPostRepository
    ) {
        this.studioRoomPostRepository = studioRoomPostRepository;
    }

    @BeforeEach
    public void setUpPagingData() {
        this.pageable = PageRequest.of(PAGE_NUMBER, PAGE_SIZE, Sort.by(
            Sort.Order.desc("createdAt")
        ));
    }

    @DisplayName("[StudioRoomPost] select 테스트 (전체)")
    @Test
    void given_when_then() throws Exception {
        // given
        int expected = 1000;

        // when
        List<StudioRoomPost> studioRoomPosts = studioRoomPostRepository.findAll();

        // then
        assertThat(studioRoomPosts).isNotNull().hasSize(expected);
    }

    @DisplayName("[StudioRoomPost] select 테스트 (남자)")
    @Test
    void givenGenderMale_whenQueryingSelect_thenReturnStudioRoomPosts() throws Exception {
        // given
        int expectedTotalSize = 750;
        int expectedTotalPage = calculateTotalPage(expectedTotalSize);

        // when
        Page<StudioRoomPost> studioRoomPostsPage = studioRoomPostRepository.findAllByPublisherGender(
            MALE.name(), pageable);

        // then
        assertThat(studioRoomPostsPage.getContent()).hasSize(pageable.getPageSize());
        assertThat(studioRoomPostsPage.getTotalElements()).isEqualTo(expectedTotalSize);
        assertThat(studioRoomPostsPage.getTotalPages()).isEqualTo(expectedTotalPage);
    }

    @DisplayName("[StudioRoomPost] select 테스트 (여자)")
    @Test
    void givenGenderFemale_whenQueryingSelect_thenReturnStudioRoomPosts() throws Exception {
        // given
        int expectedTotalSize = 250;
        int expectedTotalPage = calculateTotalPage(expectedTotalSize);

        // when
        Page<StudioRoomPost> studioRoomPostsPage = studioRoomPostRepository.findAllByPublisherGender(
            FEMALE.name(), pageable);

        // then
        assertThat(studioRoomPostsPage.getContent()).hasSize(pageable.getPageSize());
        assertThat(studioRoomPostsPage.getTotalElements()).isEqualTo(expectedTotalSize);
        assertThat(studioRoomPostsPage.getTotalPages()).isEqualTo(expectedTotalPage);
    }

    @DisplayName("[StudioRoomPost] select 테스트 필터링 조건만 있는 경우")
    @Test
    void givenSearchFilter_whenQuerying_thenReturnStudioRoomPosts() throws Exception {
        // given
        int expectedTotalSize = 499;
        int expectedTotalPage = calculateTotalPage(expectedTotalSize);
        SearchFilterRequest searchFilter = SearchFilterRequest
            .fromJson("{\"roomTypes\": [0, 1], \"rentalTypes\": [0]}");

        // when
        Page<StudioRoomPost> studioRoomPostsPage = studioRoomPostRepository
            .findStudioRoomPostByDynamicFilter(MALE.name(), searchFilter, null, pageable);

        // then
        assertThat(studioRoomPostsPage.getContent()).hasSize(pageable.getPageSize());
        assertThat(
            studioRoomPostsPage.getContent().get(0).getRoomInfo().getRoomType().getDescription())
            .containsAnyOf(RoomType.VILLA.getDescription(), RoomType.OFFICE_TEL.getDescription());
        assertThat(
            studioRoomPostsPage.getContent().get(0).getRoomInfo().getRentalType().getDescription())
            .isEqualTo(RentalType.MONTHLY.getDescription());
        assertThat(studioRoomPostsPage.getTotalElements()).isEqualTo(expectedTotalSize);
        assertThat(studioRoomPostsPage.getTotalPages()).isEqualTo(expectedTotalPage);
    }

    @DisplayName("[StudioRoomPost] select 테스트 검색어만 있는 경우")
    @Test
    void givenSearchKeyWords_whenQuerying_thenReturnsStudioRoomPosts() throws Exception {
        // given
        String searchKeyWords = "lorem";

        // when
        Page<StudioRoomPost> studioRoomPostsPage = studioRoomPostRepository
            .findStudioRoomPostBySearchKeyWords(MALE.name(), searchKeyWords, pageable);

        // then
        assertThat(studioRoomPostsPage.getContent()).hasSize(pageable.getPageSize());
        assertThat(
            studioRoomPostsPage.getContent().get(0).getTitle() + " " +
                studioRoomPostsPage.getContent().get(0).getContent()
        ).contains(searchKeyWords);
    }

    @DisplayName("[StudioRoomPost] select join publisher member account")
    @Test
    void givenNoting_whenJoinQuerying_thenReturnStudioRoomPosts() throws Exception {
        // given

        // when
        List<StudioRoomPost> studioRoomPosts = studioRoomPostRepository.findAll();
        StudioRoomPost studioRoomPost = studioRoomPosts.get(0);
        MemberAccount publisherAccount = studioRoomPost.getPublisherAccount();

        // then
        assertThat(publisherAccount.getGender()).isEqualTo(Gender.MALE.name().toLowerCase());
    }


    // -- 편의 메서드 -- //
    private int calculateTotalPage(int pageSize) {
        return (int) Math.ceil((double) pageSize / PAGE_SIZE);
    }
}