# VA-Diffusion / SiT-S/2

이 디렉터리는 VA-Diffusion 모델 패밀리의 SiT instance다.

## 고정된 base

- 모델: `SiT-S/2`
- 체크포인트: `/data/RBF-Solver/weights/SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors`
- 입력 latent: `[B,4,32,32]`
- hidden: `[B,256,384]`
- patch embedding, 12개 SiT block, final head는 모두 frozen

## 학습하는 addon

- VDVAE식 bottom-up encoder
- block 4, 8, 12 뒤의 Prior
- NVAE식 residual Posterior
- `Linear(16,384,bias=False)` ZeroProject

각 adapter에서:

```python
p = Prior(h)
delta = PosteriorDelta(concat(h, encoder_tokens))
mu_q = mu_p + delta_mu
logstd_q = clamp(logstd_p + delta_logstd, -5, 2)
z = sample(q)
h = h + ZeroProject(z)
```

shape:

| 항목 | shape |
|---|---|
| SiT hidden | `[B,256,384]` |
| encoder tokens | `[B,256,128]` |
| prior/posterior latent | `[B,256,16]` |
| projected residual | `[B,256,384]` |

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
- GPU: RTX 5090 2장

`train_va_diffusion.py`는 `torchrun` 단일·다중 GPU를 모두 지원하며, checkpoint에는
frozen SiT를 복사하지 않고 addon과 optimizer만 저장한다.

대시보드 생성 샘플은 Euler, CFG 1.5, NFE `1·3·5·10`, prior temperature
`0.0·0.3·0.6`, 조합당 4장을 사용한다. Temperature는
`z = mu_p + temperature * sigma_p * epsilon`에만 적용한다.
