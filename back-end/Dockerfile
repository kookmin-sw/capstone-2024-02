FROM amazoncorretto:17-alpine-jdk
ARG JAR_FILE=./build/libs/*.jar
COPY ${JAR_FILE} /maru.jar
ENTRYPOINT ["java","-jar","/maru.jar"]