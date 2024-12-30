# Sentimental Analysis

2024학년도 2학기 확장형 고성능 컴퓨팅 수업의 기말 프로젝트로 진행한 **Sentiment Analysis**입니다.

* [문제 설명](submit/2024SHPC-Fall-Project.pdf)
* [보고서](submit/report.pdf)

## 프로젝트 목표
- 딥 러닝 기반 Sentiment Analysis 모델을 **병렬화 및 최적화**하여 높은 Throughput을 달성
- 기존의 순차 코드로 구현된 추론 프로그램을 **Pthread**, **OpenMP**, **MPI**, **CUDA** 등을 사용해 고성능 코드로 변환
- 프로젝트 환경:
  - **4개의 계산 노드**와 **총 16개의 NVIDIA V100 GPU**를 사용
  - 외부 라이브러리(CUBLAS, CUDNN 등)는 사용 불가
- **Throughput (sentences/sec)** 기준으로 성능 평가:
  - 입력된 문장을 가능한 빠르게 처리하여 긍정(Positive)인지 부정(Negative)인지 판별

---

## 모델 개요

<img width="1256" alt="image" src="https://github.com/user-attachments/assets/a4677cea-db0c-46a8-a3d9-3bea5c365082" />

- **CNN 기반 Sentiment Analysis 모델**을 사용
- 입력된 문장이 긍정(Positive)인지 부정(Negative)인지 분류

---

## 최적화 전략

### HW-aware Conv1D Kernel
- Conv1D 연산을 개선하기 위해 GPU 하드웨어 구조에 맞춘 Kernel 설계:
  - Input Channel(C)의 값이 크고 독립적으로 처리되는 점을 고려하여 이를 기반으로 Thread Block을 분할
  - Shared Memory 크기를 최적화하기 위해 블록 안의 데이터를 별도로 타일링함
  - Kernel Fusion을 통해 Conv1D와 ReLU를 하나의 Kernel로 통합하여 실행에 드는 오버헤드 감소

### HW-aware Linear Kernel
- 기존 GEMM 커널([Siboehm Kernel 5](https://siboehm.com/articles/22/CUDA-MMM))을 활용하여 Linear 연산을 최적화:
  - GEMM과 다르게 B matrix에서 Transpose가 필요했는데, 이에 메모리 접근 패턴을 간단히 수정하여 처리
  - Kernel Fusion을 통해 Linear와 ReLU를 하나의 Kernel로 통합하여 실행에 드는 오버헤드 감소

### Batch Input
- 데이터를 Batch 단위로 동시에 처리하여 커널의 파라미터 로드 과정을 최소화

### End-to-End CUDA Optimization
- Tensor를 GPU로 전송한 뒤, GPU 내부에서 모든 연산이 수행되어 결과만 반환하도록 함
- 네 종류의 Conv1D 연산에 각각 CUDA Streams를 할당하여 GPU 하나에서 각 Conv1D 연산이 병렬적으로 처리되도록 함

### Multi-GPU Implementation
- CUDA Streams를 활용하여 코드가 4개의 GPU에서 병렬적으로 실행하도록 함:
  - 동일한 모델을 사용하며, 입력 데이터만 분리하여 병렬 처리
  - OpenMP를 활용하여 GPU 한 개를 CPU thread 한 개가 관리하도록 하여 병렬성 확보

### Multi-Node Implementation
- 코드를 4개의 노드에서 병렬 처리되도록 MPI를 활용하여 확장함
- 데이터 전송을 최소화하기 위해 필요한 데이터 전체를 미리 전송하도록 함(pre-loading)
