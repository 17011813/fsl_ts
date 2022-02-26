# few shot learning (fsl) time series predict / anomaly detection based on MAML

**코드 구조**
```
-- fsl_ts_dataloader.py：meta train의 batch task 훈련 데이터를 얻기 위한 코드
-- fsl_ts_maml.py：few shot learning 메인 제어 프로그램，train、evaluate、test、predict 다중 프로세스를 포함，하이퍼파라미터 의미와 기본값을 포함
-- lstm_learner.py：lstm 모델，MAML 가운데 base model
-- maml_higher_learner：MAML 알고리즘의 주요 구현 모듈，meta_train과 fine_tune 포함
-- plot_tools.py：시각화
```

**데이터셋**

4개의 KPI 데이터 세트 사용 중，사전 처리되어 아래의 디렉토리에 배치되어있음：
```
-- fsl_generator
---- fsl_pool
------- xxx_spt_x_pool.npy
------- xxx_spt_y_pool.npy
------- xxx_qry_x_pool.npy
------- xxx_qry_y_pool.npy
------- ......
```
사용자가 정의한 데이터 세트를 사용하는것도 가능 dataloader，입력 형식은 MAML 모듈을 따름、lstm 모듈의 support set과 query set의 요구사항은 아래와 같음。

**하이퍼파라미터 설명**

*경로 클래스 매개변수*
```
model_params_dir: 모델을 저장하는 경로
figure_dir: 훈련 loss 그래프
logs_dir: 로그 저장 경로
logs_name: 로그의 이름. 기본 출력은 로그에 저장. 명령줄의 출력은 없음. 출력을 관찰해야 하는 경우 cat 명령을 사용하여 로그 내용을 캡처할 수 있음
check_point: 모델 매개변수에 해당하는 체크포인트 이름 초기화
```
*meta-training 매개변수*
```
epoch: meta-training 학습 횟수
n_ways: （일시적으로 쓸모 없음 ??，분류 작업에서 n-way-k-shot 정의）
k_spt: 각 task 당 support set 갯수
k_qry: 각 task 당 query set 갯수
task_num: meta_train 단계의 task batch에 해당하는 batch 크기
task_num_eval: meta_evaluate 단계의 task batch에 해당하는 batch 크기
task_num_test: meta_test 단계의 task batch에 해당하는 batch 크기
meta_lr: meta_update(outer loop) 학습 속도
update_lr: inner loop 학습 속도
update_step: meta_train 단계의 inner loop 업데이트 횟수
update_step_test: meta_test 단계의 fine_tune 업데이트 횟수
clip_val: gradient clipping을 위한 매개변수（그라디언트 폭발 방지）
```
