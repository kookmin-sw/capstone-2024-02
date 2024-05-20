package org.capstone.maru.repository.postgre.querydsl;

import com.querydsl.core.types.Order;
import com.querydsl.core.types.OrderSpecifier;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.jpa.impl.JPAQuery;
import com.querydsl.jpa.impl.JPAQueryFactory;
import jakarta.persistence.EntityManager;
import java.util.ArrayList;
import java.util.List;
import org.capstone.maru.dto.DormitoryRoomRecommendPost;
import org.capstone.maru.dto.QDormitoryRoomRecommendPost;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.support.PageableExecutionUtils;
import org.springframework.util.StringUtils;

import static org.capstone.maru.domain.QDormitoryRoomPost.dormitoryRoomPost;
import static org.capstone.maru.domain.QRecommend.recommend;

public class DormitoryRoomPostCustomRepositoryImpl implements DormitoryRoomPostCustomRepository {

    private final JPAQueryFactory jpaQueryFactory;

    public DormitoryRoomPostCustomRepositoryImpl(@Autowired EntityManager em) {
        this.jpaQueryFactory = new JPAQueryFactory(em);
    }


    @Override
    public Page<DormitoryRoomRecommendPost> findDormitoryRoomPostByPublisherGender(
        String memberId,
        String gender,
        String cardOption,
        Pageable pageable
    ) {
        List<DormitoryRoomRecommendPost> content = jpaQueryFactory
            .select(new QDormitoryRoomRecommendPost(dormitoryRoomPost, recommend))
            .from(dormitoryRoomPost)
            .join(recommend)
            .on(dormitoryRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender)
            )
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .orderBy(postSort(pageable))
            .fetch();

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(dormitoryRoomPost.count())
            .from(dormitoryRoomPost)
            .join(recommend)
            .on(dormitoryRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    @Override
    public Page<DormitoryRoomRecommendPost> findDormitoryRoomPostBySearchKeyWords(
        String memberId,
        String gender,
        String searchKeyWords,
        String cardOption,
        Pageable pageable
    ) {
        List<DormitoryRoomRecommendPost> content = jpaQueryFactory
            .select(new QDormitoryRoomRecommendPost(dormitoryRoomPost, recommend))
            .from(dormitoryRoomPost)
            .join(recommend)
            .on(dormitoryRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender),
                containSearchKeyWords(searchKeyWords)
            )
            .orderBy(postSort(pageable))
            .offset(pageable.getOffset())
            .limit(pageable.getPageSize())
            .fetch();

        JPAQuery<Long> countQuery = jpaQueryFactory
            .select(dormitoryRoomPost.count())
            .from(dormitoryRoomPost)
            .join(recommend)
            .on(dormitoryRoomPost.id.stringValue().eq(recommend.recommendationId))
            .where(
                recommend.userId.eq(memberId),
                recommend.cardType.eq(cardOption),
                eqGender(gender),
                containSearchKeyWords(searchKeyWords)
            );

        return PageableExecutionUtils.getPage(content, pageable, countQuery::fetchOne);
    }

    private BooleanExpression eqGender(String gender) {
        return dormitoryRoomPost.publisherGender.eq(gender);
    }

    private BooleanExpression containSearchKeyWords(String searchKeyWords) {
        if (!StringUtils.hasText(searchKeyWords)) {
            return null;
        }

        return containsTitle(searchKeyWords).or(containsContent(searchKeyWords));
    }

    private BooleanExpression containsTitle(String searchKeyWords) {
        return dormitoryRoomPost.title.contains(searchKeyWords);
    }

    private BooleanExpression containsContent(String searchKeyWords) {
        return dormitoryRoomPost.content.contains(searchKeyWords);
    }

    private OrderSpecifier<?>[] postSort(Pageable page) {
        List<OrderSpecifier<?>> result = new ArrayList<>();
        if (!page.getSort().isEmpty()) {
            for (Sort.Order order : page.getSort()) {
                Order direction = order.getDirection().isAscending() ? Order.ASC : Order.DESC;
                result.add(
                    switch (order.getProperty()) {
                        case "createdAt" ->
                            new OrderSpecifier<>(direction, dormitoryRoomPost.createdAt);
                        case "modifiedAt" ->
                            new OrderSpecifier<>(direction, dormitoryRoomPost.modifiedAt);
                        default -> new OrderSpecifier<>(direction, recommend.score);
                    }
                );
            }
        }

        return result.isEmpty() ? null : result.toArray(OrderSpecifier[]::new);
    }
}
