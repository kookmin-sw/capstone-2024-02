# 🏠 공간 공유 서비스앱 MARU

<br>

## 프로젝트 소개

> 저희는 공간을 함께 공유하며 사용할 사람들을 찾아주는 플랫폼을 제공하고자 합니다.  
> 공급의 감소, 전셋값의 상승 등 도시에서 여러 이유로 주거비가 상승하고 있습니다.   
> 이는 특히 경제적으로 어려운 젊은 세대에게 부담으로 다가오고 있습니다.
>
> 저희의 목표는 젊은 세대의 이러한 경제적 어려움을 공유 경제를 통해 해결하는 것입니다.   
> 즉, 젊은 세대들이 함께 주거 비용을 나누고, 남는 공간을 효율적으로 활용하여 부담을 줄이는 것을 돕고자 합니다.  
> 각자의 생활 방식에 따라 함께 살 룸메이트 또는 하우스메이트를 찾을 수 있도록 도와주며, 남는 공간을 공유하고 싶어하는 사람들을 연결시켜주어 비용 부담의 문제를 해결하고자
> 합니다.

<br>

## 팀원 구성

<div align="center">

|                                                              **최정민 (FE)**                                                              |                                                          **조희정 (FE)**                                                          |                                                           **이준호 (BE)**                                                           |                                                                **정연수 (BE)**                                                                |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------:|
| [<img src="https://avatars.githubusercontent.com/u/55117867?v=4" height=150 width=150> <br/> @cjeongmin](https://github.com/cjeongmin) | [<img src="https://avatars.githubusercontent.com/u/66050038?v=4" height=150 width=150> <br/> @he2e2](https://github.com/he2e2) | [<img src="https://avatars.githubusercontent.com/u/39540595?v=4" height=150 width=150> <br/> @leejh7](https://github.com/leejh7) | [<img src="https://avatars.githubusercontent.com/u/52970725?v=4" height=150 width=150> <br/> @cheesecrust](https://github.com/cheesecrust) |

</div>

<br>

## 1. 개발 환경

- Intellij IDEA Ultimate 2023.3.3 ~
- Java 17
- Gradle 8.5
- Spring Boot 3.2.2
- H2 Database (Dev)
- PostgreSQL
- Github, Github Issues, Github Project, GitKraken  
  Notion, Github Wiki, Slack
- [Figma](https://www.figma.com/file/7kNRrVNUAK9KZdMAPVBPgU/wire-frame?type=design&node-id=0-1&mode=design&t=oZJiMTmm1bsBSgXw-0)
- [컨벤션](document/CONVENTION.md)
  <br>

## 2. 채택한 프레임워크, 인프라, 브랜치 전략

### 🍃 Spring Boot

- 자동 구성 기능을 제공하여 복잡한 설정 없이도 빠르게 프로젝트를 시작하고 프로토타입을 만들 수 있어 사용했습니다.
- Spring 프레임워크의 기능과 라이브러리 (Spring Security, Spring Data, Spring MVC 등) 를 활용할 수 있어 사용했습니다.
- YAML 파일을 사용하여 애플리케이션의 환경을 쉽게 설정할 수 있어 사용했습니다.

### 🐘 PostgreSQL

- 오픈 소스이며 무료로 사용할 수 있어 사용했습니다.
- Postgre DBMS의 구조를 공부하고 서비스에 최적화된 설정으로 커스터마이즈 하여 사용했습니다.

### 🖊️ SonarLint, GoogleStyle

- 정해진 규칙에 따라 자동적으로 코드 스타일을 정리해 코드의 일관성을 유지하고자 했습니다.
- 코드 품질 관리는 SonarLint에, 코드 포맷팅은 GoogleStyle에 일임하였습니다.
- Google의 코딩 컨벤션을 참고해 사용했고, 예외 규칙은 팀원들과 협의했습니다.
- 협업 시 매번 컨벤션을 신경 쓸 필요 없이 빠르게 개발하는 데에 목적을 두었습니다.

### 🐳 Docker

- 

### ☁️ AWS

- ec2, rds, s3

### 🪵 브랜치 전략

- Git-flow 전략을 기반으로 main, dev 브랜치와 feature 보조 브랜치를 운용했습니다.
- main, develop, feat 브랜치로 나누어 개발을 하였습니다.
    - **main** 브랜치는 배포 단계에서만 사용하는 브랜치입니다.
    - **dev** 브랜치는 개발 단계에서 git-flow의 master 역할을 하는 브랜치입니다.
    - **feat** 브랜치는 기능 단위로 독립적인 개발 환경을 위하여 사용하고 merge 후 각 브랜치를 삭제해주었습니다.
      <br>

## 3. 프로젝트 구조

```
├── README.md
├── build.gradle
├── .gitignore
├── gradlew
├── gradlew.bat
├── settings.gradle
│
└── src
     ├── main
     |     ├── java
     |     |     └── maru
     |     |           ├── config
     |     |           ├── controller
     |     |           ├── domain
     |     |           ├── dto
     |     |           ├── repository
     |     |           ├── service
     |     |           └── BackEndApplication.java
     |     |
     |     └── resources 
     |             └── application.yaml
     ├── test
     |     └── java
     |           └── maru
     |                 ├── config
     |                 ├── controller
     |                 ├── domain
     |                 ├── dto
     |                 ├── repository
     |                 ├── service
     |                 └── BackEndApplicationTests.java
```

<br>

## 4. 역할 분담

### 👻 이준호

<br>

### 🐬 정연수

<br>

## 5. 개발 기간 및 작업 관리

### 개발 기간

- 전체 개발 기간 : 2024-03-04 ~

<br>

### 작업 관리

- GitHub Projects와 Issues를 사용하여 진행 상황을 공유했습니다.
- 주간 회의를 진행하며 작업 순서와 방향성에 대한 고민을 나누고 Notion에 회의 내용을 기록했습니다.

<br>

## 6. 신경 쓴 부분

<br>

## 7. 참고한 자료

### [🗄️ 백엔드 아카이브](https://www.notion.so/Archive-256df43d0e6d4b148bf366cc7312e20b?pvs=4)

프로젝트를 진행하면서 참고한 문서 및 블로그 등을 스크랩하여 스크럼 시간에 팀원과 자신이 알게된 지식들을 공유하며 리뷰하는 시간을 가졌습니다.
<br>

## 8. 트러블 슈팅

### [⚠️ 이슈 트래커](https://www.notion.so/Issue-Tracker-bc6d59d8d90a40f0a03edac83b9d595b?pvs=4)

프로젝트를 진행하면서 직면한 버그와 이슈를 정리하고 관리하여 다른 팀원이 동일한 문제에 직면했을 때 빠르게 해결할 수 있도록 문제의 원인, 해결방법을 정리하였습니다.

### [📈 포퍼먼스](https://www.notion.so/Performance-09735be60ecb42899546424654390f93?pvs=4)

프로젝트를 진행하면서 서비스의 성능을 향상시킬 수 있는 부분에 대해서 고민하고 테스팅한 후 성능 향상 전/후의 측정값들을 기록하고 성능 향상 원인에 대해 공부한 것을
정리하였습니다.

## 9. 설계 다이어그램

### [시퀀스 다이어그램] (https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1#G1BFeex-2cN2aTEROFH5kdMMmAsgxf-xuQ)
<br>
