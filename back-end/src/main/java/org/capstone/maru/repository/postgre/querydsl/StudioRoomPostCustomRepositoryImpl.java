package org.capstone.maru.repository.postgre.querydsl;

import com.querydsl.core.types.ExpressionUtils;
import com.querydsl.core.types.Order;
import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.jpa.JPAExpressions;
import com.querydsl.jpa.JPQLQuery;
import com.querydsl.jpa.impl.JPAQuery;
import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.annotation.Nonnull;
import jakarta.persistence.EntityManager;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.capstone.maru.domain.Recommend;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.dto.QStudioRoomRecommendPost;
import org.capstone.maru.dto.StudioRoomRecommendPost;
import org.capstone.maru.dto.request.RangeRequest;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.support.PageableExecutionUtils;
import org.springframework.util.StringUtils;

import static org.capstone.maru.domain.QRecommend.recommend;
import static org.capstone.maru.domain.QStudioRoomPost.studioRoomPost;
import static org.capstone.maru.domain.QRoomInfo.roomInfo;

@Slf4j
public class StudioRoomPostCustomRepositoryImpl implements
    org.capstone.maru.repository.postgre.querydsl.StudioRoomPostCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    public StudioRoomPostCustomRepositoryImpl(@Autowired EntityManager em) {
        this.jpaQueryFactory = new JPAQueryFactory(em);
    }

    @Override
    public Page<StudioRoomPost> findStudioRoomPostByDynamicFilter(
        String gender,
        @Nonnull SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        Pageable pageable
    ) {
        List<StudioRoomPost> content = jpaQueryFactory
            .selectFrom(studioRoomPost)
            .join(studioRoomPost.roomInfo, roomInfo)
            .fetchJoin()
            .where(
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
                eqHasLivingRoom(searchFilterRequest.hasLivingRoom()),
                eqNumberOfRoom(searchFilterRequest.numberOfRoom()),
                eqNumberOfBathRoom(searchFilterRequest.numberOfBathRoom()),
                betweenRoomSize(searchFilterRequest.roomSizeRange()),
                inFloorTypes(searchFilterRequest.floorTypes()),
                betweenExpectedPayment(searchFilterRequest.expectedPaymentRange()),
                containSearchKeyWords(searchKeyWords)
            )
            .orderBy(postSort(pageable))
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .fetch();

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .where(
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
                eqHasLivingRoom(searchFilterRequest.hasLivingRoom()),
                eqNumberOfRoom(searchFilterRequest.numberOfRoom()),
                eqNumberOfBathRoom(searchFilterRequest.numberOfBathRoom()),
                betweenRoomSize(searchFilterRequest.roomSizeRange()),
                inFloorTypes(searchFilterRequest.floorTypes()),
                betweenExpectedPayment(searchFilterRequest.expectedPaymentRange()),
                containSearchKeyWords(searchKeyWords)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    @Override
    public Page<StudioRoomPost> findStudioRoomPostBySearchKeyWords(
        String gender,
        String searchKeyWords,
        Pageable pageable
    ) {
        List<StudioRoomPost> content = jpaQueryFactory
            .selectFrom(studioRoomPost)
            .where(
                eqGender(gender),
                containSearchKeyWords(searchKeyWords)
            )
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .orderBy(postSort(pageable))
            .fetch();

        // 개수
        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .where(
                eqGender(gender),
                containSearchKeyWords(searchKeyWords)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    @Override
    public Page<StudioRoomRecommendPost> findStudioRoomPostByRecommendDynamicFilter(
        String gender,
        @Nonnull SearchFilterRequest searchFilterRequest,
        String searchKeyWords,
        String memberId,
        String cardOption,
        Pageable pageable
    ) {
        List<StudioRoomRecommendPost> content = jpaQueryFactory
            .select(new QStudioRoomRecommendPost(studioRoomPost, recommend))
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
                eqHasLivingRoom(searchFilterRequest.hasLivingRoom()),
                eqNumberOfRoom(searchFilterRequest.numberOfRoom()),
                eqNumberOfBathRoom(searchFilterRequest.numberOfBathRoom()),
                betweenRoomSize(searchFilterRequest.roomSizeRange()),
                inFloorTypes(searchFilterRequest.floorTypes()),
                betweenExpectedPayment(searchFilterRequest.expectedPaymentRange()),
                containSearchKeyWords(searchKeyWords)
            )
            .orderBy(recommend.score.desc())
            .fetch();

        // 개수
        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
                eqHasLivingRoom(searchFilterRequest.hasLivingRoom()),
                eqNumberOfRoom(searchFilterRequest.numberOfRoom()),
                eqNumberOfBathRoom(searchFilterRequest.numberOfBathRoom()),
                betweenRoomSize(searchFilterRequest.roomSizeRange()),
                inFloorTypes(searchFilterRequest.floorTypes()),
                betweenExpectedPayment(searchFilterRequest.expectedPaymentRange()),
                containSearchKeyWords(searchKeyWords)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    @Override
    public Page<StudioRoomRecommendPost> findStudioRoomRecommendPostBySearchKeyWords(
        String memberId,
        String gender,
        String searchKeyWords,
        String cardOption,
        Pageable pageable
    ) {
        List<StudioRoomRecommendPost> content = jpaQueryFactory
            .select(new QStudioRoomRecommendPost(studioRoomPost, recommend))
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                eqGender(gender),
                containSearchKeyWords(searchKeyWords),
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption)
            )
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .orderBy(postSort(pageable))
            .fetch();

        // 개수
        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                eqGender(gender),
                containSearchKeyWords(searchKeyWords),
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    @Override
    public Page<StudioRoomRecommendPost> findAllRecommendByPublisherGender(
        String memberId,
        String gender,
        String cardOption,
        Pageable pageable
    ) {
        List<StudioRoomRecommendPost> content = jpaQueryFactory
            .select(new QStudioRoomRecommendPost(studioRoomPost, recommend))
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender)
            )
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .orderBy(postSort(pageable))
            .fetch();

        log.info("[Debug] content: {}", content);

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .join(recommend)
            .on(studioRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender)
            );

        log.info("[Debug] count: {}", countQuery);

        log.info("[Debug] countQuery: {}", countQuery.fetchOne());

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    private BooleanExpression eqGender(String gender) {
        return studioRoomPost.publisherGender.eq(gender);
    }

    private BooleanExpression inRoomTypes(List<RoomType> roomTypes) {
        if (roomTypes == null || roomTypes.isEmpty()) {
            return null;
        }

        return studioRoomPost.roomInfo.roomType.in(roomTypes);
    }

    private BooleanExpression inRentalTypes(List<RentalType> rentalTypes) {
        if (rentalTypes == null || rentalTypes.isEmpty()) {
            return null;
        }

        return studioRoomPost.roomInfo.rentalType.in(rentalTypes);
    }

    private BooleanExpression eqHasLivingRoom(Boolean hasLivingRoom) {
        if (hasLivingRoom == null) {
            return null;
        }

        return studioRoomPost.roomInfo.hasLivingRoom.eq(hasLivingRoom);
    }

    private BooleanExpression eqNumberOfRoom(Short numberOfRoom) {
        if (numberOfRoom == null) {
            return null;
        }

        return studioRoomPost.roomInfo.numberOfRoom.eq(numberOfRoom);
    }

    private BooleanExpression eqNumberOfBathRoom(Short numberOfBathRoom) {
        if (numberOfBathRoom == null) {
            return null;
        }

        return studioRoomPost.roomInfo.numberOfBathRoom.eq(numberOfBathRoom);
    }

    private BooleanExpression betweenRoomSize(RangeRequest roomSizeRange) {
        if (roomSizeRange == null) {
            return null;
        }

        return studioRoomPost.roomInfo.size.between(roomSizeRange.start(), roomSizeRange.end());
    }

    private BooleanExpression inFloorTypes(List<FloorType> floorTypes) {
        if (floorTypes == null || floorTypes.isEmpty()) {
            return null;
        }

        return studioRoomPost.roomInfo.floorType.in(floorTypes);
    }

    private BooleanExpression betweenExpectedPayment(RangeRequest expectedPaymentRange) {
        if (expectedPaymentRange == null) {
            return null;
        }

        return studioRoomPost.roomInfo.expectedPayment.between(expectedPaymentRange.start(),
            expectedPaymentRange.end());
    }

    private BooleanExpression containSearchKeyWords(String searchKeyWords) {
        if (!StringUtils.hasText(searchKeyWords)) {
            return null;
        }

        return containsTitle(searchKeyWords).or(containsContent(searchKeyWords));
    }

    private BooleanExpression containsTitle(String searchKeyWords) {
        return studioRoomPost.title.contains(searchKeyWords);
    }

    private BooleanExpression containsContent(String searchKeyWords) {
        return studioRoomPost.content.contains(searchKeyWords);
    }

    private OrderSpecifier<?>[] postSort(Pageable page) {
        List<OrderSpecifier<?>> result = new ArrayList<>();
        if (!page.getSort().isEmpty()) {
            for (Sort.Order order : page.getSort()) {
                Order direction = order.getDirection().isAscending() ? Order.ASC : Order.DESC;
                result.add(
                    switch (order.getProperty()) {
                        case "createdAt" ->
                            new OrderSpecifier<>(direction, studioRoomPost.createdAt);
                        case "modifiedAt" ->
                            new OrderSpecifier<>(direction, studioRoomPost.modifiedAt);
                        default -> new OrderSpecifier<>(direction, recommend.score);
                    }
                );
            }
        }

        return result.isEmpty() ? null : result.toArray(OrderSpecifier[]::new);
    }
}
