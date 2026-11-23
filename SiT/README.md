# VA-Diffusion · SiT-S/2

ImageNet 256에서 고정한 SiT-S/2에 CNN addon을 학습한다. 데이터와 모델 가중치의 위치는 실행 환경에서 지정한다.

| 구성 | 현재 실행 |
|---|---|
| 입력 | ImageNet train의 256×256 RGB 이미지 → 고정 VAE → 4×32×32 clean latent |
| Backbone | 원본 SiT-S/2, 12 blocks, patch 2, hidden 384. 파라미터 32,963,360개 고정 |
| Addon | 다중 해상도 CNN encoder, prior, residual posterior, bias 없는 projection. 학습 파라미터 합계 104,270,592개 |
| 주입 | 4·8·12번째 SiT block 뒤. 각 위치에서 e4·e8·e16 encoder 조건을 16×16 token grid에 맞춤 |
| 잠재변수 | 그룹마다 64×16×16. posterior 평균·log 표준편차는 prior에 각각 보정값을 더함 |
| 최적화 | 물리 GPU 1 한 장, AdamW, 학습률 0.0001, batch 128. 고정 이미지 VAE는 8장씩 인코딩 |
| 샘플링 | Step 0과 매 체크포인트에서 Euler NFE 1·3·5 × prior 온도 0·0.3·0.6 × 4장. CFG 1.5 |
| 저장 | 30분마다 addon FP32 텐서를 단일 safetensors 파일에 원자적으로 교체. 위치와 RNG는 메타데이터에 저장 |

## 학습 목표

선형 경로는 `xt = (1-t) * noise + t * clean`이다. 모델의 속도 출력으로부터 `pred_clean = xt + (1-t) * velocity`를 구한다. 각 그룹의 KL(q‖p)를 샘플별로 모든 공간·채널 좌표에서 합산하고, 배치 평균을 구한 뒤 데이터 latent의 원소 수 4096으로 나눈다.

```python
data_mse = mean((pred_clean - clean) ** 2)
rate = e4 + e8 + e16
beta = clip((completed_steps - 21) / 5021, 0.0001, 1.0)
loss = data_mse + beta * rate
```

대시보드의 STEP은 업데이트를 완료한 횟수다. 예를 들어 101번째 업데이트에 사용한 β는 `completed_steps=100`으로 계산한다. `completed_steps=5042`부터 β는 1이다. [Efficient-VDVAE 스케줄](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/schedules.py)

## 검증과 재개

실행 환경에서 `VA_SIT_WEIGHTS`는 SiT transformer와 VAE가 있는 폴더, `VA_IMAGENET_TRAIN`은 train parquet 폴더, `CUDA_VISIBLE_DEVICES`는 사용할 물리 GPU 1의 UUID로 지정한다.

`train.py --verify`는 실제 이미지를 이용한 두 번의 업데이트, freeze, 저장된 키와 크기, 새 모델에 복원한 다음 업데이트를 검사한다. 검증용 체크포인트는 임시 디렉터리에서 검사 후 제거한다.

운영 진입점은 같은 환경의 `train.py`다. 매니페스트로 게시된 Step 0이 있는지 확인하고 첫 optimizer 업데이트를 실행한다. 정상 종료 신호를 받으면 진행 중인 업데이트를 마치고 가중치와 해당 step의 36장을 저장한다. `--resume`은 저장한 데이터 위치·가중치·난수 상태를 복원하고, AdamW 모멘트는 새로 시작한다. 위 검증은 이 재개 조건을 기준으로 한다.

## 대시보드

대시보드는 별도 프로세스로 실행한다. 로컬 PRX 원본을 `--prx-html`로 지정하면 헤더, 체크포인트 표시, step 선택기, 확대 보기와 차트 코드를 읽는다. `--bind`로 접근할 네트워크 인터페이스를 명시한다. 학습 상태는 프로세스의 실제 동작 여부와 PID를 함께 확인한다.

`samples/<step>/manifest.json`이 게시된 세트만 선택할 수 있다. 같은 class와 NFE에서는 모든 온도에 동일한 초기 noise와 prior 난수를 사용한다. Step 0에는 projection이 0이어서 온도를 바꿔도 같은 그림이 나온다.
