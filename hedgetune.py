from evodiff.pretrained import OA_DM_38M
from evodiff.pretrained import OA_DM_640M
from evodiff.generate import generate_oaardm
from evodiff.generate import generate_GA_reward_metrics, generate_oaardm_reward_metrics, generate_oaardm_reward_metrics_edit, likelihood, generate_oaardm_reward_metrics_edit_initial
import os.path
import datetime
from datetime import date
import logging
import warnings
import pandas as pd
import numpy as np
import torch
import os, sys
import warnings
import os.path
from types import SimpleNamespace
import utils
import esm

from utils import set_seed
from args_file import get_args
from reward import *
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
# seaborn 쓰지 말라는 제약이 있을 수 있으니 안 씀.

# ----------------------------------------------------------------------
# 0. Amino acid encoding
# ----------------------------------------------------------------------

MASKED_TOKEN = 'Z'
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
ALPHABET_WITH_MASK = ALPHABET + MASKED_TOKEN
MASK_TOKEN_INDEX = ALPHABET_WITH_MASK.index(MASKED_TOKEN)

def encode_one_hot(seq: str, max_len: int = None) -> torch.Tensor:
    """
    seq: amino acid string (20 AA)
    returns: [L, V] one-hot tensor
    """
    if max_len is None:
        max_len = len(seq)
    x = torch.zeros(max_len, len(ALPHABET_WITH_MASK), dtype=torch.float32)
    for i, ch in enumerate(seq[:max_len]):
        if ch in ALPHABET_WITH_MASK:
            x[i, ALPHABET_WITH_MASK.index(ch)] = 1.0
    return x

# ----------------------------------------------------------------------
# 1. Reward model (simple 1D CNN over one-hot)
# ----------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    매우 심플한 seq -> scalar 회귀 모델
    (원하면 여기 대신 ESM2 feature + MLP로 바꿔도 됨)
    """
    def __init__(self, vocab_size: int = len(ALPHABET_WITH_MASK), max_len=512, hidden_dim: int = 256):
        super().__init__()
        # self.conv = nn.Conv1d(vocab_size, hidden_dim, kernel_size=3, padding=1)
        # self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Linear(vocab_size * max_len, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, V]
        returns: [B] (scalar reward)
        """
        x = x.view(x.size(0), -1)  # flatten [B, L*V]
        h = F.relu(self.mlp(x))
        # h = self.pool(h).squeeze(-1)  # [B, hidden_dim]
        r = self.head(h).squeeze(-1)  # [B]
        return r

# ----------------------------------------------------------------------
# 2. Dataset for reward model training 
# ----------------------------------------------------------------------

class SeqRewardDataset(Dataset):
    def __init__(self, path: Path, max_len: int = 512):
        """
        read csv
        """
        df = pd.read_csv('log/'+path)
        y_series = df['plddt']
        seq_series = df['sequence']
        self.max_len = max_len
        self.ys = y_series.tolist()
        self.seqs = seq_series.tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        y = self.ys[idx]
        seq = self.seqs[idx]
        x = encode_one_hot(seq, max_len=self.max_len)
        return x, y
        
def collate_fn(batch):
    xs, ys = zip(*batch)  # xs: list [L, V], ys: float
    # 여기서는 간단하게 모두 동일 길이(max_len)로 encode 했다고 가정
    x = torch.stack(xs, dim=0)  # [B, L, V]
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y

# ----------------------------------------------------------------------
# 3. ESMFold true reward 함수 (HuggingFace 버전 예시)
# ----------------------------------------------------------------------
# 실제로는 ProDifEvo-Refinement의 reward.py에 있는 esmfold wrapper를
# import해서 쓰는 게 더 안정적일 거야.
# (여기 코드는 self-contained 예시)

try:
    from transformers import AutoTokenizer, EsmForProteinFolding
    _HAS_ESMFOLD = True
except ImportError:
    _HAS_ESMFOLD = False

class ESMFoldScorer:
    def __init__(self, device: str = "cuda"):
        if not _HAS_ESMFOLD:
            raise ImportError("transformers[esm] 가 설치되어 있어야 합니다.")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score_plddt(self, seqs: List[str]) -> List[float]:
        """
        seqs: list of AA sequences
        returns: list of mean pLDDT (0~1 scaled)
        """
        batch = self.tokenizer(seqs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**batch)
        # outputs.plddt: [B, L] in 0~100
        plddt = outputs.plddt.mean(dim=-1) / 100.0
        return plddt.cpu().tolist()

# ----------------------------------------------------------------------
# 4. Phase 1: true reward dataset 만들기
# ----------------------------------------------------------------------

def random_seq(seq_len: int) -> str:
    return "".join(np.random.choice(list(ALPHABET_WITH_MASK), size=seq_len))

# ----------------------------------------------------------------------
# 5. Phase 2: reward model 학습
# ----------------------------------------------------------------------

def train_reward_model(
    data_path: Path,
    max_len: int = 256,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 10,
):
    dataset = SeqRewardDataset(data_path, max_len=max_len)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = RewardModel().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_val = float("inf")
    save_path = Path("model/reward_model_ckpt.pth")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = mse(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                pred = model(x)
                loss = mse(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        print(f"[epoch {epoch}] train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, save_path)
            # torch.save(model, save_path)
            print(f"  -> best model updated (val {best_val:.4f})")

# ----------------------------------------------------------------------
# 6. Phase 3: RGIR + reward model + 다양한 theta로 샘플 수집
# ----------------------------------------------------------------------

def load_reward_model(ckpt_path: Path, device: str = "cuda") -> RewardModel:
    model = RewardModel()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def reward_model_score(model: RewardModel, seqs: List[str], max_len: int = 256, device: str = "cuda") -> List[float]:
    xs = [encode_one_hot(s, max_len=max_len) for s in seqs]
    x = torch.stack(xs, dim=0).to(device)
    r = model(x)
    return r.cpu().tolist()

def run_rgir_with_reward_model(
    theta: float,
    num_samples: int,
    reward_model_ckpt: Path,
    out_jsonl: Path,
    seq_len: int = 100,
    max_len: int = 256,
    device: str = "cuda",
):
    """
    이 함수 안에서 실제로는 ProDifEvo-Refinement의 RGIR(SVDD_edit)를 호출해야 함.
    여기서는 간단한 placeholder로 random seq에서 reward_model gradient 없이
    그냥 "theta"를 기록하는 형태로만 만들어 놓음.

    실제로는:
      - refinement.py 내부에 reward model을 호출하는 hook을 추가하거나
      - reward 함수 안에서 reward_model_score(seq)를 사용하게 만들고
      - theta를 test-time hyperparameter (예: reward_scale)로 넘겨주면 됨.
    """
    device = device
    model = load_reward_model(reward_model_ckpt, device=device)
    esm_scorer = ESMFoldScorer(device=device)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(out_jsonl, "w") as f:
        # 이 부분을 "실제 RGIR로 theta 세팅해서 시퀀스 뽑기" 로 교체
        for i in range(num_samples):
            # TODO: 여기 대신
            # seq = run_refinement_with_theta(theta, ...)
            seq = random_seq(seq_len)

            proxy = reward_model_score(model, [seq], max_len=max_len, device=device)[0]
            true_reward = esm_scorer.score_plddt([seq])[0]

            rec = {
                "seq": seq,
                "theta": float(theta),
                "proxy_reward": float(proxy),
                "true_reward": float(true_reward),
            }
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 10 == 0:
                print(f"[theta={theta}] {i+1}/{num_samples} samples done.")

def collect_rgir_grid(
    thetas: List[float],
    reward_model_ckpt: Path,
    out_dir: Path,
    num_samples_per_theta: int = 200,
    seq_len: int = 100,
    max_len: int = 256,
    device: str = "cuda",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for th in thetas:
        out_path = out_dir / f"samples_theta_{th:.3f}.jsonl"
        run_rgir_with_reward_model(
            theta=th,
            num_samples=num_samples_per_theta,
            reward_model_ckpt=reward_model_ckpt,
            out_jsonl=out_path,
            seq_len=seq_len,
            max_len=max_len,
            device=device,
        )

# ----------------------------------------------------------------------
# 7. Phase 4: HedgeTune + Figure 3-style plot
# ----------------------------------------------------------------------

def load_all_samples(sample_dir: Path) -> Dict[float, Dict[str, np.ndarray]]:
    """
    sample_dir 안에 있는 jsonl 파일들에서
    theta별 proxy_reward, true_reward 배열을 모아옴.
    returns:
      {
        theta_value: {
          "proxy": np.array([...]),
          "true": np.array([...]),
        }, ...
      }
    """
    result = {}
    for path in sample_dir.glob("samples_theta_*.jsonl"):
        thetastr = path.stem.split("_")[-1]
        theta = float(thetastr)
        proxies, trues = [], []
        with open(path, "r") as f:
            for line in f:
                d = json.loads(line)
                proxies.append(d["proxy_reward"])
                trues.append(d["true_reward"])
        result[theta] = {
            "proxy": np.array(proxies, dtype=np.float32),
            "true": np.array(trues, dtype=np.float32),
        }
    return result

def plot_theta_vs_true_reward(
    stats: Dict[float, Dict[str, np.ndarray]],
    out_path: Path,
):
    thetas = sorted(stats.keys())
    mean_true = [stats[th]["true"].mean() for th in thetas]
    std_true  = [stats[th]["true"].std() for th in thetas]

    plt.figure()
    plt.errorbar(
        thetas,
        mean_true,
        yerr=std_true,
        fmt="-o",
    )
    plt.xlabel("theta (test-time hyperparameter)")
    plt.ylabel("True reward (e.g., mean pLDDT)")
    plt.title("True reward vs theta (RGIR + reward model)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def prepare_arrays_for_hedgetune(
    stats: Dict[float, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HedgeTune에 넘길 수 있도록
    (theta, proxy_reward, true_reward)들을 flatten해서 만든다.

    returns:
      thetas_all: [N]  (각 샘플에 대응되는 theta 값)
      proxy_all:  [N]
      true_all:   [N]
    """
    theta_list = []
    proxy_list = []
    true_list = []
    for th, d in stats.items():
        n = len(d["proxy"])
        theta_list.append(np.full(n, th, dtype=np.float32))
        proxy_list.append(d["proxy"])
        true_list.append(d["true"])
    thetas_all = np.concatenate(theta_list, axis=0)
    proxy_all  = np.concatenate(proxy_list, axis=0)
    true_all   = np.concatenate(true_list, axis=0)
    return thetas_all, proxy_all, true_all

def run_hedgetune_placeholder(
    thetas_all: np.ndarray,
    proxy_all: np.ndarray,
    true_all: np.ndarray,
):
    """
    여기는 실제 Hedgetune 라이브러리를 가져다가 쓰는 자리.

    예를 들면 (실제 함수 이름은 hedging repo README 확인 필요):

      from hedging import HedgeTune

      tuner = HedgeTune()
      theta_star = tuner.fit(
          proxy_scores=proxy_all,
          true_scores=true_all,
          theta_values=thetas_all,
      )

    같은 형식으로 넘기면 됨.

    아래는 아주 naive한 baseline: 단순히 theta별 true reward 평균이 가장 큰 것 선택.
    (oracle tuning; 실제 HedgeTune과는 다른 baseline)
    """
    unique_thetas = np.unique(thetas_all)
    best_theta = None
    best_true = -1e9
    for th in unique_thetas:
        mask = (thetas_all == th)
        mean_true = true_all[mask].mean()
        if mean_true > best_true:
            best_true = mean_true
            best_theta = th
    print(f"[Naive oracle baseline] best theta = {best_theta}, mean true reward = {best_true:.4f}")
    return best_theta, best_true

# ----------------------------------------------------------------------
# 8. CLI
# ----------------------------------------------------------------------

def main():

    path = '2025-12-12 08:04:32.657928random_plddt_1_None/output.csv'
    plot_path = 'log/'

    plot_path = Path()

    train_reward_model(
        data_path=path,
        max_len=512,
        batch_size=32,
        lr=1e-3,
        num_epochs=30,
    )

    # elif args.mode == "collect_rgir":
    #     # 여기서 thetas는 RGIR의 test-time hyperparameter grid
    #     # (예: guidance scale, iteration 수 등)에 맞게 수정해서 사용
    #     thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #     collect_rgir_grid(
    #         thetas=thetas,
    #         reward_model_ckpt=ckpt_path,
    #         out_dir=sample_dir,
    #         num_samples_per_theta=200,
    #         seq_len=100,
    #         max_len=128,
    #         device=args.device,
    #     )

    # elif args.mode == "analyze":
    #     stats = load_all_samples(sample_dir)
    #     plot_theta_vs_true_reward(stats, plot_path)
    #     thetas_all, proxy_all, true_all = prepare_arrays_for_hedgetune(stats)
    #     _ = run_hedgetune_placeholder(thetas_all, proxy_all, true_all)
    #     # 여기서 thetas_all, proxy_all, true_all를 numpy로 저장한 뒤
    #     # Hedgetune 라이브러리가 기대하는 형식으로 넘기면 됨.
    #     np.savez(sample_dir / "hedgetune_data.npz",
    #              thetas=thetas_all,
    #              proxy=proxy_all,
    #              true=true_all)

if __name__ == "__main__":
    main()