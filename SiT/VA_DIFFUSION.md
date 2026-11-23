# VA-Diffusion / SiT-S/2

이 디렉터리는 VA-Diffusion 모델 패밀리의 SiT instance다.

## 고정된 base

- 모델: `SiT-S/2`
- 체크포인트: `/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors`
- 입력 latent: `[B,4,32,32]`
- hidden: `[B,256,384]`
- patch embedding, 12개 SiT block, final head는 모두 frozen

## 학습하는 addon

- VDVAE식 bottleneck residual CNN encoder
- block 4, 8, 12 뒤의 spatial CNN Prior
- NVAE식 residual spatial CNN Posterior
- `Conv2d(64,384,1,bias=False)` ZeroProject

| 구성 | parameters |
|---|---:|
| CNN Encoder | 42,864,384 |
| Encoder feature condition | 590,592 |
| Prior | 27,763,584 |
| Posterior | 33,072,000 |
| ZeroProject | 73,728 |
| **addon 합계** | **104,364,288** |

각 adapter에서:

```python
mu_p, logstd_p = Prior(h_map)
delta = PosteriorDelta(concat(h_map, encoder_condition))
mu_q = mu_p + delta_mu
logstd_q = clamp(logstd_p + delta_logstd, -5, 2)
z = sample(q)
h_map = h_map + ZeroProject(z)
```

shape:

| 항목 | shape |
|---|---|
| SiT hidden | `[B,256,384]` |
| encoder e16/e8/e4 | `[B,512,16,16]` / `[B,768,8,8]` / `[B,1024,4,4]` |
| encoder condition | `[B,256,16,16]` |
| prior/posterior latent | `[B,64,16,16]` |
| projected residual map | `[B,384,16,16]` |
| projected residual tokens | `[B,256,384]` |

## Loss

Free bits와 KL balancing은 사용하지 않는다.

```python
x_t = (1 - t) * noise + t * clean
pred_clean = x_t + (1 - t) * pred_velocity
data_mse = mean((pred_clean - clean)**2)
# kl_sum[level]은 이미지별로 spatial 및 latent 차원을 합산한 KL(q || p)
kl_rate = mean(kl_sum["e4"] + kl_sum["e8"] + kl_sum["e16"]) / (4 * 32 * 32)
beta = max(1e-4, min(1.0, (step - 21) / 5021))
loss = data_mse + beta * kl_rate
```

## 67번 서버 입력

- ImageNet Parquet: `/home/scpark/data/imagenet-1k-adm-crop-256/data/train-*.parquet`
- local VAE: `/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/vae`
- 학습 GPU: 물리 `GPU 1`의 RTX 5090 한 장만 사용

`train_va_diffusion.py`의 기본 장치는 `--device 1`이다. 물리 GPU 1만 노출한 환경에서는 `--device 0`으로 실행한다.
`step`은 해당 업데이트 직전까지 완료한 횟수다. 5,042에서 β가 1에 도달한다. [Efficient-VDVAE 설정](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/hparams.cfg) · [스케줄 구현](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/efficient_vdvae_torch/model/schedules.py)
학습은 epoch 상한 없이 계속되며 SIGINT 또는 SIGTERM을 받으면 현재 step을 마친 뒤
checkpoint와 샘플을 저장하고 정상 중지한다.

대시보드 생성 샘플은 Euler, CFG 1.5, NFE `1·3·5`, prior temperature
`0.0·0.3·0.6`, 조합당 4장을 사용한다. Temperature는
`z = mu_p + temperature * sigma_p * epsilon`에만 적용한다.
고정 class label은 `207·360·387·974`이며 모든 조합에서 초기 noise와 prior epsilon을 공유한다.
첫 업데이트 전 STEP 0과 이후 30분마다 Euler 샘플 36장을 생성한다.
`latest.pt`의 모델 텐서는 FP32 addon 가중치 104,364,288개다.
학습 위치와 난수 상태도 기록한다. 재개 시 AdamW 모멘트는 새로 시작한다.

검증 진입점 `verify_va_diffusion.py`는 실제 ImageNet 배치로 초기 출력의 일치,
frozen SiT 파라미터 불변, 빈 AdamW로 재개한 다음 업데이트의 재현성, 36개 PNG의 수와 해상도를 검사한다.
