package org.capstone.maru.util;

import java.util.List;
import java.util.Random;
import java.util.UUID;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.FeatureCard;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.domain.RoomImage;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.constant.FloorType;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;
import org.capstone.maru.domain.jsonb.ExtraOption;
import org.capstone.maru.domain.jsonb.MemberFeatures;

public class EntityCreator {

    // -- 회원 관련 생성 메서드 -- //
    public static FeatureCard createMemberCard(MemberFeatures memberFeatures, int i) {
        return FeatureCard.of(
            null,
            memberFeatures
        );
    }

    public static MemberFeatures createMemberFeatures(int i) {
        switch (i % 3) {
            case 0 -> {
                return MemberFeatures.of("흡연", "같은 방", (short) (i % 10), null);
            }
            case 1 -> {
                return MemberFeatures.of("비흡연", "다른 방", (short) (i % 10), null);
            }
            default -> {
                return MemberFeatures.of("상관없어요", "상관없어요", null, null);
            }
        }
    }

    public static ProfileImage createProfileImage(int i) {
        return ProfileImage.of("test" + i + ".jpg", "tester" + i);
    }

    public static MemberAccount createMemberAccount(int i) {
        return MemberAccount.of(
            "test_" + i,
            "test" + i + "@email.com",
            "tester" + i,
            "1980" + i % 11,
            i % 2 == 0 ? "MALE" : "FEMALE",
            "010-1234-" + String.format("%04d", i),
            "test_" + i,
            true,
            createMemberCard(createMemberFeatures(i), i),
            createMemberCard(createMemberFeatures(i), i),
            null,
            null,
            createProfileImage(i),
            null
        );
    }


    // -- 게시글 관련 생성 메서드 -- //
    public static Address createAddress(int i) {
        return Address.of(
            "test old address" + i,
            "test road address" + i
        );
    }

    public static RoomType randomRoomType() {
        Random random = new Random();
        switch (random.nextInt(7)) {
            case 1 -> {
                return RoomType.TWO_ROOM_VILLA;
            }
            case 2 -> {
                return RoomType.OFFICE_TEL;
            }
            case 3 -> {
                return RoomType.APT;
            }
            default -> {
                return RoomType.ONE_ROOM;
            }
        }
    }

    public static FloorType randomFloorType() {
        Random random = new Random();
        switch (random.nextInt(3)) {
            case 1 -> {
                return FloorType.GROUND;
            }
            case 2 -> {
                return FloorType.SEMI_BASEMENT;
            }
            default -> {
                return FloorType.PENTHOUSE;
            }
        }
    }

    public static RentalType randomRentalType() {
        Random random = new Random();
        if (random.nextInt(2) == 1) {
            return RentalType.MONTHLY;
        }
        return RentalType.JEONSE;
    }

    public static ExtraOption createExtraOption(int i) {
        return ExtraOption.of(
            i % 2 == 0,
            i % 2 == 1,
            i % 2 == 0,
            i % 2 == 1,
            i % 2 == 0
        );
    }

    public static RoomInfo createRoomInfo(int i) {
        return RoomInfo.of(
            randomRoomType(),
            randomFloorType(),
            (short) (i % 20),
            (short) (i % 5),
            (short) (i % 5),
            i % 2 == 0,
            randomRentalType(),
            (long) (10_0000 * (i % 11) / 2),
            createExtraOption(i)
        );
    }

    public static StudioRoomPost createStudioRoomPost(int i) {
        return StudioRoomPost.of(
            "test title" + i,
            "test content" + i,
            i % 2 == 0 ? "MALE" : "FEMALE",
            (short) (i % 4),
            createMemberAccount(i % 100),
            createMemberCard(createMemberFeatures(i), i),
            createAddress(i),
            createRoomInfo(i)
        );
    }

    public static RoomImage createRoomImage(int j, StudioRoomPost studioRoomPost) {
        return RoomImage.of(
            UUID.randomUUID().toString() + ".jpg",
            j % 3 == 1,
            (short) (j % 3),
            studioRoomPost
        );
    }
}
