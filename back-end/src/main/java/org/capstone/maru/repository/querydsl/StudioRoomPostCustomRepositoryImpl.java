package org.capstone.maru.repository.querydsl;

import com.querydsl.core.types.Order;
import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.jpa.impl.JPAQuery;
import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.annotation.Nonnull;
import jakarta.persistence.EntityManager;
import java.util.ArrayList;
import java.util.List;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.dto.request.SearchFilterRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.support.PageableExecutionUtils;
import org.springframework.util.StringUtils;

import static org.capstone.maru.domain.QStudioRoomPost.studioRoomPost;

public class StudioRoomPostCustomRepositoryImpl implements StudioRoomPostCustomRepository {

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
            .where(
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
                containSearchKeyWords(searchKeyWords)
            )
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .orderBy(postSort(pageable))
            .fetch();

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .where(
                eqGender(gender),
                inRoomTypes(searchFilterRequest.roomTypes()),
                inRentalTypes(searchFilterRequest.rentalTypes()),
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

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(studioRoomPost.count())
            .from(studioRoomPost)
            .where(
                eqGender(gender),
                containSearchKeyWords(searchKeyWords)
            );

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
                        default -> new OrderSpecifier<>(direction, studioRoomPost.id);
                    }
                );
            }
        }

        return result.isEmpty() ? null : result.toArray(OrderSpecifier[]::new);
    }
}
