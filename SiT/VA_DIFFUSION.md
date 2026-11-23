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
p = Prior(h)
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
kl = mean([kl_e4, kl_e8, kl_e16])
beta = 0.03 * min(step / (0.30 * total_steps), 1.0)
loss = flow_mse + beta * kl
```

## 67번 서버 입력

- ImageNet Parquet: `/home/scpark/data/imagenet-1k-adm-crop-256/data/train-*.parquet`
- local VAE: `/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/vae`
- 학습 GPU: 물리 `GPU 1`의 RTX 5090 한 장만 사용

`train_va_diffusion.py`는 단일 GPU 학습만 수행하며 기본 장치는 `--gpu 1`이다.
checkpoint에는 frozen SiT를 복사하지 않고 addon과 optimizer만 저장한다.

대시보드 생성 샘플은 Euler, CFG 1.5, NFE `1·3·5·10`, prior temperature
`0.0·0.3·0.6`, 조합당 4장을 사용한다. Temperature는
`z = mu_p + temperature * sigma_p * epsilon`에만 적용한다.
고정 class label은 `207·360·387·974`이며 모든 조합에서 초기 noise와 prior epsilon을 공유한다.
Addon·optimizer·epoch/batch 위치·CPU/CUDA RNG는 30분마다 `latest.pt`에 저장하고,
각 checkpoint 직후 같은 STEP의 48장 Euler 샘플을 생성한다.

67번 서버 실측 검증:

- ImageNet 294 shards와 local VAE를 이용한 실제 학습 step 통과
- batch 192 peak GPU memory: 25,575.6 MiB
- checkpoint: 1,194.9 MiB, 저장 0.53초
- frozen base gradient 0, addon gradient 정상
