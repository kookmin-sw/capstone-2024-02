package org.capstone.maru.util;

import java.util.List;
import java.util.Random;
import java.util.UUID;
import org.capstone.maru.domain.Address;
import org.capstone.maru.domain.Address.CITY;
import org.capstone.maru.domain.MemberAccount;
import org.capstone.maru.domain.MemberCard;
import org.capstone.maru.domain.ProfileImage;
import org.capstone.maru.domain.RoomImage;
import org.capstone.maru.domain.RoomInfo;
import org.capstone.maru.domain.StudioRoomPost;
import org.capstone.maru.domain.constant.RentalType;
import org.capstone.maru.domain.constant.RoomType;

public class EntityCreator {

    // -- 회원 관련 생성 메서드 -- //
    public static MemberCard createMemberCard(List<String> memberFeatures, int i) {
        return MemberCard.of(
            null,
            "test location" + i % 11,
            memberFeatures
        );
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
            "tester" + i,
            true,
            createMemberCard(List.of(), i),
            createMemberCard(List.of(), i),
            null,
            null,
            createProfileImage(i)
        );
    }


    // -- 게시글 관련 생성 메서드 -- //
    public static Address createAddress(int i) {
        return Address.of(
            CITY.SEOUL,
            "test old address" + i,
            "test road address" + i,
            "test detail address" + i,
            "test station" + i,
            (short) (i % 60),
            (short) (i % 60),
            "test school" + i,
            (short) (i % 60),
            (short) (i % 60)
        );
    }

    public static RoomType randomRoomType() {
        Random random = new Random();
        switch (random.nextInt(7)) {
            case 1 -> {
                return RoomType.TWO_ROOM_VILLA;
            }
            case 2 -> {
                return RoomType.THREE_ROOM_VILLA;
            }
            case 3 -> {
                return RoomType.OFFICE_TEL;
            }
            case 4 -> {
                return RoomType.TWO_ROOM_OFFICE_TEL;
            }
            case 5 -> {
                return RoomType.THREE_ROOM_OFFICE_TEL;
            }
            case 6 -> {
                return RoomType.APT;
            }
            default -> {
                return RoomType.ONE_ROOM_VILLA;
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

    public static RoomInfo createRoomInfo(int i) {
        return RoomInfo.of(
            createAddress(i),
            randomRoomType(),
            (short) (i % 20),
            (short) (i % 5),
            randomRentalType(),
            (long) (500_0000 + (i % 11) * 1000000),
            (long) (10_0000 * (i % 11)),
            (long) (10000 * (i % 11)),
            (long) (10_0000 * (i % 11) / 2),
            (short) (i % 4)
        );
    }

    public static StudioRoomPost createStudioRoomPost(int i) {
        return StudioRoomPost.of(
            "test title" + i,
            "test content" + i,
            i % 2 == 0 ? "MALE" : "FEMALE",
            createMemberAccount(i % 100),
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
