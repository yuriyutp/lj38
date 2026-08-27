# 契約 A — LJ38 履歴機構監査 v0.3.6（導出規則）

状態語彙は `{CERT_FAIL, UNVERIFIED}`。本契約下のいかなる測定も `PASS` を発行しない。
（器械検査のラベルのみ `INSTRUMENT_CHECK_OK` / `INSTRUMENT_CHECK_FAIL`。器械のラベルであって科学的証明書ではない。）

**導出規則・方式選択・検定方式・検証許容差・有意水準・同等性幅・乱数導出規則は、本文書と
規範的 companion `CALIBRATION_PROTOCOL_V036.json` の組で確定する。契約 B が凍結するのは
較正由来の数値のみであり、方式選択を一切含まない。**

- 契約 ID: `lj38-audit-v0.3.6-contractA`
- 前身: `lj38-audit-v0.3.5-contractA`。それ以前は `v0.2.2` および未凍結ドラフト `v0.3`–`v0.3.4`

---

## §A-0 v0.3.5 からの修正一覧

| # | v0.3.5 の欠陥／停止事由 | v0.3.6 の対応 |
|---|---|---|
| 1 | §6.1/§12 は較正出力を列挙したが、候補集合・選択写像・較正予算・停止規則が未定義で、契約 A 単独から契約 B が一意に決まらなかった | 規範的 companion `CALIBRATION_PROTOCOL_V036.json` を契約の一部とし、anchor-1 manifest で本文と同時に固定 |
| 2 | 補助プロトコル v0.3.5 の PT 軌道予算 180,000 では、中間 anneal 層で IPS thinning 後 40 点に対し 37 点しか得られず `CALIBRATION_FAIL` | PT 較正軌道の探索予算を 300,000 に固定。`n_per_traj=40` は変更しない |
| 3 | seed 100–105 は v0.3.5 較正で既に使用され、未観測の較正集合ではない | seed 100–105 を開発証拠へ降格し、v0.3.6 較正は seed 106–111 に固定 |
| 4 | §12 項目 8 に burn-in 長が残る一方、本文は burn-in 不使用を確定していた | 項目 8 から burn-in 長を削除。burn-in=0 は方式として本文で固定 |

v0.3.5 の 37 完了セル、失敗行、raw/checkpoint は削除せず開発証拠として保存する。科学的所見および契約 B は発行しない。

### §A-1 v0.3.4 からの修正一覧

| # | v0.3.4 の欠陥 | 対応 |
|---|---|---|
| 1 | 冒頭は「契約 B は数値のみ」としながら、§12 に方式選択（burn-in の有無、`source × routing` の主 estimand、自己相関推定法、Hessian の対称化・射影法、壁時計を独立予算とするか）が残っていた | すべて本契約で確定（§3.3.4、§2.1、§1.2、§11.4、§4） |
| 2 | `δ_equiv` を契約 B（較正後）へ送っていた。観測された差に合わせて同等性幅を選べる | §8.4 で**外部根拠に基づき本契約で固定**。根拠が拒否される場合の代替も明記 |
| 3 | 「固定テスト用閾値」の数値が未指定 | §11.3 で `g_test = 1e−4` を固定し、`0.9 g_test / g_test / 1.1 g_test` の三点検査と `<` の境界意味論を明記 |

---

## §B 実測による凍結事項

**B-1. `bin_key` はプロセス間で非決定的。** Python は `bytes` のハッシュを `PYTHONHASHSEED` で撹乱する。
同一入力の実測: `-2201530587008039063` / `-1150182246883363621` / `-5124579102905009035`。
v0.2.2 の `basin_key` 由来の計数の地位は
**「run 内の近似診断値として使用可能。衝突は未検証。cross-run の同一性判定は無効。」**

**B-2. `assert` は `python -O` で完全に除去される。**
実測: `[normal] AssertionError: integrity violated` / `[python3 -O] {'raw_equals_store': True, 'duplicates': 0}`。

**B-3. `Q_inst` の有効性はドメイン依存。** §1.2 参照。

### §B-4 自己 penalty の作用の向き

`H_t` を更新前履歴、`m` を現在状態、`m'` を提案とする。受理判定より前に `m'` を登録すると、
追加された `m'` が `m'` 自身へ `h·K(m',m')=h` を、`m` へ `h·K(D(m),D(m'))` を**同時に**加えるため

$$D_B - D_A = h\bigl(1 - K(D(m),D(m'))\bigr) \;\ge\; 0$$

だけ指数内エネルギー差が増える。**相殺は「現在状態の既存自己項」によるものではない**
（既存自己項は正誤どちらの順序にも存在し、この差分から消えている）。
残る `h(1−K)` は遠い提案ほど大きく、taboo 機構が意図する方向と**逆向き**である。
定数の初回罰ではない。

---

## 0. 凍結する問い

拘束付き LJ38 fixture 上で、履歴依存の再訪抑制機構の**配置**（どこで記憶を読み、どこに介入するか）が、
同一 PES 予算の basin hopping との差を説明するか。
副次的に、慣性を持つ提案カーネルが履歴とは独立の寄与を持つか。

---

## 1. 写像と用語の定義

### 1.1 `Q_inst` — 固定アルゴリズム

| 項目 | 値 |
|---|---|
| 実装 | `scipy.optimize.minimize(method='L-BFGS-B', jac=True)` |
| `maxiter` / `ftol` / `gtol`(=`pgtol`) | 400 / 1e-14 / 1e-8 |
| 返り値規則 | 探索中に評価した全点の最小エネルギー点と `res.x` のうちエネルギーが低い方 |
| tie-break | エネルギー同値なら `res.x`（決定的） |
| **環境（本契約で凍結）** | Python 3、numpy **2.4.4**、scipy **1.17.1**。差し替えは契約 A の改訂を要する |

**`Q_inst` を「極小写像」と呼ぶことを禁止する。** 呼称は `Q_inst`（部分緩和写像）。

### 1.2 `p_μ` — 有効性・採取設計・区間推定方式

$$p_\mu = \Pr_{x\sim\mu}\bigl[\ \mathrm{complete}(x)\wedge\|\nabla E(Q_{\rm inst}x)\|<g_{\rm tol}\ \bigr]$$

`complete(x)` は予算枯渇による中断が起きなかったこと。`status==0` のみでは判定しない。

| ドメイン | 入力分布 | 採取設計 | 再標本化単位 |
|---|---|---|---|
| `mu_sample` | `P.sample(rng)` | 独立生成 n=300 | i.i.d. |
| `mu_bh` | 収束済み点 + `step`·N(0,I) | `n_traj` × `n_per_traj`、thinning `τ_bh` | **軌道** |
| `mu_pt` | PT 冷レプリカの熱的配置 | `n_traj` × anneal 3 層（`[1,2/3)`,`[2/3,1/3)`,`[1/3,0]`）× `n_per_traj`、thinning `τ_pt` | **軌道（層内）** |

**自己相関時間の推定法（本契約で確定）:**
各軌道について、ゲート通過を表す二値指標列に **Geyer の initial positive sequence (IPS) 推定量**を適用し、
積分自己相関時間 `τ_int` を得る。thinning 間隔は

$$\tau = \bigl\lceil 2\,\tau_{\rm int} \bigr\rceil$$

とする。他の推定法（窓付き推定、ブロック平均、AR 当てはめ等）は用いない。
`τ_bh` / `τ_pt` の**数値**のみ契約 B で凍結する。

**区間推定方式（本契約で確定）:**

- `mu_sample`: Wilson 区間
- `mu_bh` / `mu_pt`: **cluster bootstrap、B = 2000、percentile 法**（BCa 不使用）。
  再標本化単位は軌道。`mu_pt` は層内で再標本化。片側 95% 下限を `LCB₉₅(p_μ)` と書く。
- **bootstrap seed:** 昇格判定が bootstrap 出力に依存するため、anchor-1 manifest に列挙された
  **`CONTRACT_A_V036.md` component digest** から決定的に導出する。

  ```
  seed = int(SHA256(CONTRACT_A_V036.md).hexdigest()[:16], 16) mod 2**63
  rng  = numpy.random.default_rng(seed)
  ```

  digest は本契約の**実バイト列**（UTF-8、LF 改行）に対して計算する。
  この値は凍結時点で一意に決まり、較正データを見てから選ぶ経路は存在しない。
- `mu_pt` は**層別しない `p_μ` を報告しない**。層別値と重み明示の加重平均を併記する。

診断 quench の PES 呼び出しは `diag_grad_budget`（§4.1）。

**参考値（n=60 ずつ、単一軌道、`g_tol` 未固定。探索的・契約 B で置換）:**

| ドメイン | `maxiter` 到達率 | ‖∇E‖ 中央値 | p90 | 最大 |
|---|---:|---:|---:|---:|
| `mu_sample` | 98.3% | 1.51 | 1.12e+1 | 2.69e+1 |
| `mu_bh` (step=0.38) | 1.7% | 2.03e−5 | 3.81e−5 | 1.02e−3 |
| `mu_pt` (T=0.05, dt=1e−3) | 0.0% | 1.92e−5 | 4.53e−5 | 1.03e−4 |

### 1.3 `best_x` 分岐カウンタ

`res.x` 以外が返された回数を全 run で計数し raw に保存。本監査中は 0/120（`mu_sample`）。
到達不能の証明ではない（95% 上限 3/120 = 2.5%）。カウンタは削除しない。

### 1.4 記述子・量子化・識別子

- `desc(x)` = ソート済みペア距離（LJ38 で 703 次元）。E(3) × S_38 不変。**完全不変量としては未証明。**
- $q_r(x)=\operatorname{round}(\operatorname{desc}(x),\mathrm{decimals}=r)$。`r` は小数点以下桁数。
- 識別子: 実行中は `q_r(x)` の rounded tuple、保存用 ID は正準バイト列
  （`float64`, little-endian, C-order, `+0.0` 正規化）の **SHA-256**。rounded vector も保存。
- Python `hash` の識別子利用を**禁止**。

| フィールド | 定義 |
|---|---|
| `n_descriptor_bins` | 全 `Q_inst` 出力の相異なる `q_r` 数 |
| `n_converged_descriptor_bins` | `complete ∧ ‖∇E‖ < g_tol` を通った出力のみ |
| `n_hessian_positive_descriptor_bins` | 上記かつ代表点の射影 Hessian が正定値（§11.4） |

**三者いずれも「局所極小の個数」と読み替えることを禁止する。**

### 1.5 中心登録ゲート

ゲートは**純粋述語**として実装する。

```
gate(complete: bool, gnorm: float, g_tol_hill: float) -> bool
    return complete and (gnorm < g_tol_hill)
```

比較は**厳密不等号 `<`** である。`gnorm == g_tol_hill` は**棄却**される。
`status==0` のみでは受理しない。ゲート判定の PES 呼び出しは `pes_oracle` に計上（§4.1）。
𝓜 観測セルでは deposit を quench の後段へ移し、**X 観測セルにも同じ順序を適用**して揃える。

---

## 2. 機構の型付け

$$(\text{観測空間})\times(\text{介入箇所})\times(\text{堆積源})\times(\text{共有経路})\times(\text{近傍核})$$

| 軸 | 値域 |
|---|---|
| `obs_space` | `X` ／ `M` |
| `intervene` | `force` ／ `accept` |
| `source` | `cold_only` ／ `all_walkers` |
| `routing` | `shared` ／ `private` |
| `kernel` | `gaussian(σ)` ／ `hard_bin(r)` |

v0.2.2 の現行実装: `(X, force, cold_only, shared, gaussian(2.0))`（コードで確認済み）。

### 2.1 交絡の宣言と範囲（本契約で確定）

- **`source × routing` は本契約の範囲外とする。** 全 walker 私有化は「ensemble 全体の生成数」と
  「各 walker が受ける数」を同時に固定できず、「共有解除の効果」と「堆積率増加の効果」が
  分離できないためである。本ラウンドでは全アームを `(cold_only, shared)` に固定し、
  当該 2 軸は**次契約の感度解析へ送る**。主 estimand の選択も次契約で行う。
- `kernel` を替える比較は「記憶の有無」「soft/hard な一般化」「σ または r」を同時に動かす。
  σ・r は較正対象、holdout で再調整しない。

### 2.2 2×2

|  | `force` | `accept` |
|---|---|---|
| `obs_space=X` | `A1`（現行 V_hist） | 本契約では扱わない |
| `obs_space=M` | **`A2`（未実装・主対比）** | `A3`（soft）／`A5`（hard, visit-count） |

---

## 3. アーム定義

### 3.1 履歴機構ブロック

| ID | 座標 | `role` |
|---|---|---|
| `A0_basinhop` | 履歴なし | 基準（必須） |
| `A1_hist_X_force` | `(X, force, cold_only, shared, gaussian(σ))` | 親アーム（必須） |
| `A2_hist_M_force` | `(M, force, cold_only, shared, gaussian(σ))` | **treatment**（必須・主対比） |
| `A3_soft_taboo_bh` | `(M, accept, —, —, gaussian(σ))` | **treatment** |
| `A4_yoked_sham_bh` | `A3` の yoked donor 版 | **control** |
| `A5_visitcount_bh` | `(M, accept, —, —, hard_bin(r))` visit-count | **treatment** |
| `A6_pt_only_same_ladder` | 履歴なし PT、`A1` と同一梯子 | **comparator** |
| `A7_pt_matched_accept` | 履歴なし PT、受理率を `A1` に整合 | **comparator** |

`A6`/`A7` は comparator であり estimand ではない。estimand は §8.5 の `θ`。
`sensitivity envelope` として報告し、`identification interval` / `上下界` の語を禁止する。

### 3.2 慣性ブロック

| ID | 内容 | `role` |
|---|---|---|
| `B1_md_proposal` | MD 提案 + `Q_inst` + 固定受理則、履歴なし | **treatment** |
| `B2_rand_proposal` | ランダム変位提案 + `Q_inst` + **同一**受理則、履歴なし | **control** |
| `B3_minima_hopping` | 完全な MH（`E_kin` と `E_diff` の二重フィードバック） | treatment（本契約では昇格対象外） |

- **`B1` 対 `B2` のみが慣性のアブレーションである。** `B3` は履歴回避を内蔵するため
  慣性の単独アブレーションではない（Goedecker, *J. Chem. Phys.* **120**, 9911 (2004), DOI 10.1063/1.1724816）。
- `B1`/`B2` は提案あたりの記述子空間 RMS 変位 `rms_desc_disp` で揃える（等価点は契約 B）。
- MD の `dt` は過減衰の `dt=1e-3` から移送しない。NVE ドリフト検査（相対ドリフト < 1e-3 / 提案）で較正。
- `B1` は 1 提案あたり `n_md` 回の力評価を消費し `B2` は 0。**`nquench` を必ず併記する。**

### 3.3 `accept` 介入の導出規則

**3.3.1 penalty**

履歴 `H_t = (m_1,\dots,m_{t-1})` に対し $P_t(m) = h \sum_{j<t} K(D(m),D(m_j))$。

| アーム | `K` |
|---|---|
| `A0` | `K ≡ 0`（`h = 0`） |
| `A3`, `A4` | `K_gauss(u,v)=exp(−‖u−v‖²/(2σ²))` |
| `A5` | `K_hard(u,v)=1[q_r(u)=q_r(v)]`（＝訪問回数） |

`K_gauss(u,u)=K_hard(u,u)=1`、`h` は全アーム共通の単一値、加算則も共通
（履歴全体の単純和、減衰なし、上限なし）。`A3` 対 `A5` の差は**近傍への一般化の有無のみ**に帰着する。

**3.3.2 受理**

$$a_t(m\to m')=\min\Bigl[1,\exp\bigl\{-\beta\bigl[(E(m')+P_t(m'))-(E(m)+P_t(m))\bigr]\bigr\}\Bigr]$$

`β = 1/T_BH`。`T_BH` は `A0` の較正値を `A0`/`A3`/`A4`/`A5` で共有する。

**3.3.3 履歴更新の順序**

1. 提案 `x'` を生成し `m' = Q_inst(x')` を得る
2. **更新前の `H_t` から** `P_t(m)` と `P_t(m')` を計算する
3. `a_t` により受理判定する
4. **受理・棄却にかかわらず**、§1.5 のゲートを通った `m'` を追加して `H_{t+1}` を作る

報告に `history = visited (not accepted)` と明記する。
「受理後にのみ登録する」変種は本契約の範囲外。`P_t(m)` は毎ステップ再評価する。

**3.3.4 `A4` の donor / recipient と yoke（本契約で確定）**

- donor は **`A3` と同一アルゴリズムの run**。recipient と同一初期配置、異なる雑音 seed で、
  **先に完走して固定**する。recipient は**自身の履歴を一切利用しない**。

- **yoke は提案番号ではなく累積 PES 予算比率で行う。** quench 反復数は軌道ごとに異なり、
  同一予算でも提案数は一致しないためである。donor の `j` 番目の中心登録時点の累積予算から

  $$b_j = \frac{\text{donor の } \texttt{search\_grad\_used} \text{ at deposit } j}{B}$$

  （`B` は当該段階の `pes_oracle`）。recipient は、`P_t` を評価する時点の自身の
  `search_grad_used / B` が `b_j` **以上**となった中心 `j` をすべて有効とする。
  活性化は単調であり、途中で無効化しない。すべての `b_j ≤ 1` なので
  「donor 列が先に尽きる」問題は生じない。**提案番号による yoke は用いない。**

- **burn-in と時間シフトは用いない（本契約で確定）。** donor 列は `b_j = 0` から全件を用い、
  時間シフトも入れない。yoke の時間整合を厳密に保つためである。
  初期軌道の重なりは `donor_overlap_rate` として測定し、§5.2 の操作チェック帯を外れた場合は
  `sham_failed` として §5.3 の ITT 規則で処理する。**重なりを理由に burn-in を後から導入しない。**

- **壁時計:** `A4` の `P3abs`（§6.3）には **recipient 単体**の壁時計を用いる。
  `donor_wall_seconds` は**別掲**し、合算しない。

**3.3.5 `accept_flip_rate`**

共通乱数（CRN）による反実仮想。各提案 `t` で単一の一様乱数 `u_t` を引き

- `d_t^{pen} = 1[u_t < a_t(penalty あり)]`
- `d_t^{nul} = 1[u_t < a_t(penalty なし、同一 E 値)]`

`accept_flip_rate = mean_t 1[d_t^{pen} ≠ d_t^{nul}]`。`u_t` は penalty の有無で**共有**する。

---

## 4. 予算とコスト（本契約で確定）

| ID | 種別 | 定義 |
|---|---|---|
| `pes_oracle` | **予算** | 力・勾配評価回数。確認 500,000／スクリーニング 150,000。超過不可（厳密一致） |
| `diag_grad_budget` | **別枠予算** | 事後検査専用 |
| `end_to_end_cost_outcome` | **測定結果** | 固定機材上の壁時計 |

**壁時計を独立予算にしない。** 本契約における探索予算は `pes_oracle` のみであり、
壁時計は結果側の量として Pareto 報告する。壁時計制限 run は実施しない。
（`P3abs` は昇格ゲートであって探索予算ではない。）

### 4.1 予算境界

**`pes_oracle`（探索へフィードバックする全て）:** 状態遷移の `Q_inst` 呼び出し、提案生成の力評価
（`B1` の MD 積分を含む）、受理判定の `E`／`P_t` 評価、**中心生成と §1.5 ゲート判定の勾配評価**。

**`diag_grad_budget`（探索へ一切フィードバックしないもののみ）:** 事後の射影 Hessian 検査、
`p_μ` 測定用の診断 quench、復元診断。

**禁止:** 診断結果を見て archive・履歴・hill を変更すること。診断は読み出し専用である。

### 4.2 コスト報告

`end_to_end_cost_outcome` は**この実装の性能**であって手法の性能ではない。
将来の実装改善を「手法の改善」と読み替えることを禁止する。
参考: v0.2.2 実測で `hist_nq100` は `pt_nq100` と同一勾配数で 197.6 s vs 62.0 s = 3.19 倍。

---

## 5. yoked sham（`A4`）

### 5.1 無関係中心が対照にならない理由

LJ38 の相異なる極小間の記述子距離は中央値 9.71（実測: min 0.715 / max 32.7）。
σ=2.0 では `exp(−9.71²/8) ≈ 7e−6` となり penalty がほぼ消え、「penalty 無し」アームの複製になる。

### 5.2 操作チェック

`bias_ratio_mean` は力の比であり `accept` アームには定義されない。`A3`/`A4` では以下を用いる。

| 量 | 定義 |
|---|---|
| `penalty_energy_mean` | 提案点で評価した `P_t` の平均 |
| `penalty_fire_rate` | `P_t` が閾値以上となった提案の割合 |
| `accept_flip_rate` | §3.3.5（CRN 反実仮想） |
| `nearest_center_mean` | 提案点から最近傍中心までの記述子距離の平均 |
| `donor_overlap_rate` | 初期軌道の重なり率 |
| `nhills_donor` / `donor_wall_seconds` | donor 側の中心数・壁時計（別掲、合算しない） |

等価性帯の**数値**は契約 B。これは結果を一致させる条件ではなく操作チェックである。

### 5.3 `sham_failed` の扱い — ITT 型

- 固定した**全 donor–recipient ペアを主解析に残す**（intention-to-treat）。
- 失敗率が `f_sham_max`（契約 B）以上なら、**`A3` 対 `A4` の対比全体を
  `UNVERIFIED (control invalid)`** とする。個々の run を落として続行しない。
- 成立 run のみの解析は **per-protocol 参考値**に限定する。
- **`sham_failed` を通常の `incomplete` と同列に扱わない**（§8.1 の除外規則から分離）。
- donor–recipient 対応は較正段階で決定し、anchor-4 後の交換を禁止する。

### 5.4 判定規則

- 「構造化された訪問履歴が効く」の証拠が強まるのは **`A3` が `A0` と `A4` の双方を上回るとき**に限る。
- `A3 ≈ A0` から「履歴は効かない」を導くことを禁止する。無差の主張には §8.4 の TOST を要する。
- `A3` 対 `A5` から、近傍への一般化が必要か完全一致の再訪抑制で足りるかを分離する。

---

## 6. 段階構成と anchor

### 6.0 時系列（7 段階・4 anchor）

| # | 行為 | anchor |
|---|---|---|
| 1 | 契約 A（本文書）と規範的較正 companion を凍結 | **anchor-1** `SHA256(ANCHOR1_MANIFEST_V036.json)` |
| 2 | **器械 freeze**: コード commit 固定、§11.3 のクロスチェックを本契約の許容差で実行 | **anchor-2** `SHA256(INSTRUMENT_MANIFEST_V036.json)` |
| 3 | 較正段階を実行（seed 106–111、PT 軌道予算 300,000） | — |
| 4 | 契約 B で較正由来の数値を凍結 | **anchor-3** `SHA256(MANIFEST_B_V036.json)`（正準 manifest） |
| 5 | スクリーニング段階を実行（150k、seed 106–111） | — |
| 6 | 昇格規則から**機械的に** `SELECTION_V036.json` を生成 | **anchor-4** `SHA256(SELECTION_V036.json)` |
| 7 | 確認段階を実行（500k、seed 200–211）、実行後 `SHA256(manifest)` を記録 | 記録のみ |

- anchor-1 manifest に列挙された `CONTRACT_A_V036.md` component digest は §1.2 の bootstrap seed を決定する。
- **anchor-2 は較正の前に置く。** 許容差は §11.3 で凍結済みなので、結果を見てから許容差を選ぶ経路はない。
  `INSTRUMENT_MANIFEST_V036.json` が収めるのは環境・コード commit・**クロスチェックの結果**のみ。
- **v0.3.5 器械の carry-forward:** `md_search_v035.py` の実バイト列 SHA-256 が
  `4f411d1b54440aa33ac0403fe1a7fb0655b951f59bc13f70c3dff81b70657429` と一致する場合に限り、
  アルゴリズム実装を変更せず再利用できる。ただし §11.3 の全 11 検査を再実行し、
  v0.3.6 の契約・companion・較正 driver の digest とともに新しい anchor-2 manifest へ記録する。
- **複数対象を一つの digest で暗黙に指すことを禁止する。** anchor-3 は対象ごとの `path` と `sha256` を
  列挙した正準 manifest（キー昇順、UTF-8、LF）を作り、その manifest を anchor する。
- **`SELECTION_V036.json` の必須内容:**
  1. `screening_raw` の `path` と `sha256`
  2. `INSTRUMENT_MANIFEST_V036.json` の `sha256`
  3. `MANIFEST_B_V036.json` の `sha256`
  4. `selection` 生成コードの `path` と `sha256`
  5. 各ゲート（P1 / P2 / P3rel / P3abs）のセル別入力値と判定結果
  6. 昇格結果と、昇格されなかったセルの `UNVERIFIED` 事由
- 器械検査が `INSTRUMENT_CHECK_FAIL` を出した場合、較正段階へ進まない。修正は契約 A の改訂を要する。

### 6.1 較正段階（seed 106–111）

出力は契約 B に凍結される**数値のみ**。ここで得た結論を科学的所見として報告しない。
候補集合・選択写像・実行順序・停止規則は規範的 companion `CALIBRATION_PROTOCOL_V036.json` に従う。
PT の `p_μ` 較正軌道は各 seed 300,000 `pes_oracle`、`n_traj=6`、`n_per_traj=40` とする。

- `p_μ` の 3 ドメイン測定（§1.2）
- `g_tol`, `g_tol_hill`, `diag_grad_budget`, `p_min`, `k_max`, `t_max_screen`, `f_sham_max`
- `q_r` の桁数 `r`、hill 幅 `σ`、penalty 高さ `h`、`T_BH`
- PT 梯子 2 種（`A6` 用・`A7` 用）
- `B1`/`B2` の `rms_desc_disp` 等価点、MD の `dt`、`n_md`
- yoked sham の等価性帯の数値、`δ_tie`
- `τ_bh` / `τ_pt`（推定法は §1.2 で確定済み）
- 射影 Hessian の有限差分幅・固有値閾値（方法は §11.4 で確定済み）

`α`、bootstrap seed、`δ_equiv` は本契約で確定済みであり較正対象ではない。

### 6.2 スクリーニング段階（`pes_oracle` = 150,000、seed 106–111）

$$\Delta_s = E_{\rm best}(c,s) - E_{\rm best}(\mathrm{parent}(c),s)$$

| セル | `parent` | `role` |
|---|---|---|
| `A2` | `A1` | treatment（必須。screening 対象外） |
| `A3`, `A5` | `A0` | treatment |
| `A4` | `A3` | control |
| `B1` | `B2` | treatment |
| `B2` | **parent なし** | control |
| `A6`, `A7` | **parent なし** | comparator |
| `B3` | `A0` | treatment（本契約では昇格対象外） |

**禁止:** 150k で落ちたセルについて「500k でも劣る」と主張しない。150k の結果を科学的所見として報告しない。

### 6.3 昇格規則（ブロック単位・機械的）

- **P1（効果、treatment のみ）** `median_s Δ_s < 0` かつ `#{s : Δ_s < −δ_tie} ≥ 4`
- **P2（器械、全 role）** `LCB₉₅(p_μ) ≥ p_min`。`mu_pt` は**全層**について満たすこと
  （層別 LCB の**最小値**で判定。加重平均では判定しない）
- **P3rel（treatment）** `end_to_end_cost_outcome` が `parent` の `k_max` 倍以下
- **P3abs（control / comparator）** `end_to_end_cost_outcome` が**絶対上限 `t_max_screen`** 以下。
  `A4` では **recipient 単体**の壁時計を用い、`donor_wall_seconds` は別掲して合算しない

| `role` | 適用ゲート |
|---|---|
| `treatment` | P1 ∧ P2 ∧ P3rel |
| `control` | **P2 ∧ P3abs のみ** |
| `comparator` | **P2 ∧ P3abs のみ** |

control / comparator に P1 を課すと論理が反転する。

**ブロック単位 eligibility:**

```
BLK1_eligible := A3.passes(P1,P2,P3rel) AND A4.passes(P2,P3abs)
BLK2_eligible := B1.passes(P1,P2,P3rel) AND B2.passes(P2,P3abs)
BLK3_eligible := A6.passes(P2,P3abs) AND A7.passes(P2,P3abs)
```

treatment が通っても control が P2/P3abs を落とせば、**treatment だけを昇格させず、
ブロック全体を `UNVERIFIED (control invalid/infeasible)`** とする。

**優先順位:** `BLK1`（介入箇所）→ `BLK2`（慣性）→ `BLK3`（history 効果の比較対照）

```
mandatory = [A0, A1, A2]        # 常に確認段階へ。screening 対象外
slots     = 2                   # 既定 cap = 5
selected  = []

if A3.passes(P1,P2,P3rel) and not A4.passes(P2,P3abs):
    record("BLK1: UNVERIFIED (control invalid/infeasible)")
if B1.passes(P1,P2,P3rel) and not B2.passes(P2,P3abs):
    record("BLK2: UNVERIFIED (control invalid/infeasible)")

if BLK1_eligible:
    selected += [A3, A4]
    if A5.passes(P1,P2,P3rel):
        selected += [A5]                      # cap を 6 に引き上げる唯一の例外
    else:
        record("A5: UNVERIFIED (not attempted)")
elif BLK2_eligible:
    selected += [B1, B2]
elif BLK3_eligible:
    selected += [A6, A7]
else:
    record("no block promoted; confirmation runs mandatory arms only")
```

- `A3` が P1 を落とせば `A4`/`A5` も昇格させない。
- 上限 6 への引き上げは **`A5` が P1 を通過した場合に限る**。
- `A5` を黙って落とすことを禁止する（`UNVERIFIED (not attempted)` と明記）。
- 必須アームは screening 対象外だが **P2 診断は測定して報告する**。
  `A2` が P2 を落とした場合、主対比に `INSTRUMENT_CHECK_FAIL` を付し `UNVERIFIED` とする。

### 6.4 確認段階（`pes_oracle` = 500,000、seed 200–211、n=12）

- **seed 200–211 は anchor-4 まで一切実行しない。**
- winner's curse は **holdout 結果のみ**で判定する。

---

## 7. seed とストリームの割当

| 用途 | seed | 地位 |
|---|---|---|
| 較正 | 106–111 | 数値決定のみ |
| 開発証拠 | 0–5、100–105 | **既知データ。参考のみ。較正集合・確認集合として再利用しない。** |
| 最終 holdout | 200–211 | anchor-4 以降変更不可 |

固定すべきもの: 初期配置生成法、疑似乱数ストリームの割当（ただし §3.3.5 の CRN は penalty 有無で
**同一**ストリームを共有）、yoked donor 対応表、途中停止規則、`PYTHONHASHSEED`。
bootstrap seed は §1.2 により anchor-1 manifest 内の契約本文 component digest から導出される。

---

## 8. 解析計画

### 8.1 判定規則

**向き。** `Δ_s = E_best(control, s) − E_best(treatment, s)`。**`Δ_s > 0` は treatment 優位**。

**有意水準（本契約で確定）:** `α = 0.05`（両側）。

**検定。** ペア符号検定（両側、exact 二項）。

| 対比 | p 値 |
|---|---|
| **主対比**（`A2` 対 `A0`、1 件のみ） | **未補正の exact p 値**。多重補正しない |
| **副次対比**（§8.2 の族） | **Holm 調整後 p 値** |

同一表に混在させる場合、列に補正の有無を明記する。

**同点。** `|Δ_s| < δ_tie` を tie とし符号検定から除外、tie 数を必ず報告。`δ_tie` は契約 B（候補 1e−6）。
`δ_tie` は数値精度に基づく**器械閾値**であり、科学的意味を持つ `δ_equiv`（§8.4）とは別物である。
§8.4 の中央値 CI では tie を**除外しない**。

**除外規則。**

| 事象 | 扱い |
|---|---|
| `search_grad_used ≠ 500000`、最終 quench が `incomplete` | そのペアの両側を除外、除外数と理由を報告 |
| 除外後に有効ペア < 10 | 当該対比を `UNVERIFIED (insufficient pairs)` |
| `sham_failed` | **run 単位で除外しない**（§5.3 の ITT 規則） |

**ラベル写像。**

| 観測 | ラベル |
|---|---|
| 逆向きの符号検定が有意（control 側優位、当該対比の p 値 ≤ α） | **`CERT_FAIL(C)`** |
| 上記以外（有意差なし、または treatment 側有意） | **`UNVERIFIED(C)`** |

「有意差なし」は `CERT_FAIL` ではない。treatment 側が有意でも `PASS` は発行しない。
`UNVERIFIED` に `k/n` と p 値を必ず添付する。

### 8.2 検定族

主対比は 1 件: **`A2` 対 `A0`**。副次対比は `SELECTION_V036.json` が確定したアーム集合から
**実際に実行可能なものだけ**を族とし、その族にのみ Holm 補正を適用する。

| 条件 | 副次対比 |
|---|---|
| 常に | `A2` 対 `A1` |
| `BLK1` 昇格時 | `A3` 対 `A0`、`A3` 対 `A4`（`A5` も昇格なら `A3` 対 `A5`） |
| `BLK2` 昇格時 | `B1` 対 `B2` |
| `BLK3` 昇格時 | `A1` 対 `A6`、`A1` 対 `A7` を**並記**、統合しない |

### 8.3 候補媒介機構の列挙（効果分解ではない）

`history 効果 = 脱出 + 交換 + 結合 + 交互作用` は因果モデルと尺度を定義しない限り成立しない。
**本契約では効果分解を主張しない。**

| 候補機構 | 診断量 |
|---|---|
| 盆地内脱出 | `n_converged_descriptor_bins`、初到達時刻 |
| 交換促進 | 隣接温度対ごとの交換試行数・受理数、anneal 区間別受理率、round-trip 回数 |
| walker 間結合 | `routing` を振ったアームの結果（本契約では走らせないため診断量のみ） |
| 近傍核の一般化 | `nearest_center_mean`、`penalty_fire_rate` |

`exchange=off` は「交換なし」であって「walker 間結合なし」ではない。

### 8.4 無差の主張に用いる TOST（本契約で確定）

**分布自由の中央値同等性検定**を用いる。順序統計量による区間は離散であるため、
**被覆率をちょうど 90% にはできない。「coverage ≥ 90% の保守的 CI」と記述する。**

- 統計量: ペア差 `Δ_s` の母中央値
- 順序統計量の添字規則（有効ペア数 `n`）:

  $$k(n)=\max\Bigl\{k:\ 2\,F_{\mathrm{Bin}(n,1/2)}(k-1)\le 0.10\Bigr\},\qquad
  \mathrm{CI}_n=\bigl[\Delta_{(k(n))},\ \Delta_{(n-k(n)+1)}\bigr]$$

- 判定: `CI_n ⊂ (−δ_equiv, +δ_equiv)` のときのみ同等と報告する
- tie 処理: `|Δ_s| < δ_tie` の値も**除外せず**順序統計量に含める
- paired t-TOST および Wilcoxon 型 TOST は**用いない**（正規性・対称性の仮定を避けるため）

**実現被覆率（本規則から一意に決まる。事前計算値）:**

| `n` | `k(n)` | `CI` | 実現被覆率 | 実効片側 α |
|---:|---:|---|---:|---:|
| 10 | 2 | `[Δ₍₂₎, Δ₍₉₎]` | 97.8516% | 1.0742% |
| 11 | 3 | `[Δ₍₃₎, Δ₍₉₎]` | 93.4570% | 3.2715% |
| 12 | 3 | `[Δ₍₃₎, Δ₍₁₀₎]` | 96.1426% | 1.9287% |

実効片側 α は `n` に対し単調でない。**各対比の実現被覆率を必ず報告する。**
「90%」とだけ書くことを禁止する。

**`δ_equiv`（本契約で固定・外部根拠）:**

$$\delta_{\rm equiv} = 0.676049\ \varepsilon$$

これは LJ38 の fcc 大域最小（−173.928427 ε）と二十面体ファネル底（−173.252378 ε）の
**エネルギー差**であり、本 fixture が識別しようとしている当の量である。
較正データからではなく fixture の既知構造から決まるため、観測された差に合わせて選ぶ経路がない。
意味は「二つの手法の中央値差が、本 fixture の中心的な識別対象であるファネル間ギャップより小さい」である。

留意事項:
- 参照値は**非拘束** LJ38 の文献値である。本 fixture の拘束項は当該構造で不活性でなければならない。
  **anchor-2 の器械 manifest に、両参照構造で `kconf·Σ max(‖r_i−r̄‖−R_c,0)² = 0` であることを記録する。**
  不活性でない場合、`δ_equiv` の根拠は失われ、下記の代替に落とす。
- この根拠が拒否される場合、本ラウンドでは同等性主張を行わず、TOST を
  **`UNVERIFIED (margin not justified)`** と記録し記述統計のみ報告する。
- **`δ_equiv` を較正結果から選ぶことを禁止する。**

TOST 未実施の無差主張は `UNVERIFIED` とし、記述統計のみ報告する。

### 8.5 estimand の定義

$$\theta_6 = \operatorname{median}_s\bigl[E_{\rm best}(A6,s) - E_{\rm best}(A1,s)\bigr],\qquad
\theta_7 = \operatorname{median}_s\bigl[E_{\rm best}(A7,s) - E_{\rm best}(A1,s)\bigr]$$

`θ_6` と `θ_7` は**異なる estimand**であり統合しない。両者を並記し `sensitivity envelope` として報告する。
単一の「history 効果」を点推定として報告することを禁止する。

### 8.6 媒介変数の併記義務

`A1` 対 `A2` は「履歴を読む場所」の比較だが、中心の分布も penalty 強度も変わる。
`nearest_hill_mean` / `bias_energy_mean` / `bias_ratio_mean` / 中心数を必ず併記する。

---

## 9. 既存の凍結所見（v0.2.2 から持ち越す）

1. 同一勾配予算の LJ38 で、履歴反発付き過減衰 PT は basin hopping に勝てなかった
   （`hist_nq100` 1/6、`hist_nq25` 0/6 のペア勝利）。
2. この差は慣性の不在では説明できない。勝った対照も慣性を持たない。
3. 末尾前受理率の上界: `pt_nq100` ≤ 8.0%、`pt_nq25` ≤ 5.5%、`hist_nq100` ≤ 21.4%、`hist_nq25` ≤ 22.2%。
   固定梯子が二つの有効 Hamiltonian に対して同等に較正されていなかった。
4. 全アームで fcc 大域最小（−173.928427）到達は 0/6。
5. ソート gap < 1e−6 が 22.5–25.9% のステップで発生。`V_hist ∘ desc` は局所 Lipschitz だが C¹ ではない。
6. fixture は拘束付き LJ38（`Rc = 2.25·38^(1/3)`, `kconf = 20`）。非拘束 LJ38 の文献値と厳密には比較できない。
7. v0.2.2 の `basin_key` 由来の計数は **run 内の近似診断値としてのみ有効。衝突未検証。
   cross-run の同一性判定は無効**（§B-1）。

---

## 10. 禁止事項（要約）

- `PASS` の発行
- `Q_inst` を無条件の極小写像として扱うこと
- 3 種の bin 計数を局所極小数と読み替えること
- `status==0` のみでの中心登録／ゲートを `≤` で実装すること（§1.5 は厳密 `<`）
- 受理判定より前に `m'` を履歴へ登録すること（§B-4）
- 提案番号による yoke、`A4` の壁時計に donor 生成費を合算すること、
  **重なりを理由に burn-in や時間シフトを後から導入すること**
- `A6`/`A7` を区間・上下界と呼ぶこと、`θ_6`/`θ_7` を統合すること
- 「有意差なし」を `CERT_FAIL` と記録すること／TOST 抜きの無差主張
- 中央値 CI を「90% exact」と記述すること
- **`δ_equiv` を較正結果から選ぶこと**
- 主対比に多重補正を適用すること、副次対比を未補正で報告すること
- `α`・bootstrap seed・`δ_equiv` を較正後に決めること
- 150k で落ちたセルへの 500k 主張
- seed 0–5 および 100–105 の較正集合・確認集合としての再利用
- anchor-4 後の donor 交換、σ・r・`h`・`g_tol` の再調整
- `pes_oracle` と `end_to_end_cost_outcome` の混合報告、後者を予算と呼ぶこと、
  **壁時計を独立予算として扱うこと**
- 診断結果に基づく archive・履歴・hill の変更
- `A3 ≈ A0` からの「履歴は効かない」の導出
- `B3` を慣性の単独アブレーションと呼ぶこと
- `control` / `comparator` に P1 を適用すること
- treatment のみを昇格させ control を落とすこと
- `A5` の暗黙の除外、`sham_failed` の run 単位除外
- Python `hash` の識別子利用
- 対応アームを走らせずに効果分解を主張すること、
  **本契約で範囲外とした `source × routing` について結論を述べること**
- アーム選択に人手の裁量を挟むこと
- 複数ファイルを一つの digest で暗黙に指すこと
- 器械検査が `INSTRUMENT_CHECK_FAIL` の状態で次段階へ進むこと
- ゼロ近傍で相対誤差のみの許容差を用いること（§11.3 の混合基準を使う）

---

## 11. anchor と計装

### 11.1 外部 anchor の保存先・形式・時刻証明（本契約で確定）

| 項目 | 確定内容 |
|---|---|
| 対象 | anchor-1: `ANCHOR1_MANIFEST_V036.json`（本文＋規範的 companion）／anchor-2: `INSTRUMENT_MANIFEST_V036.json`／anchor-3: `MANIFEST_B_V036.json`／anchor-4: `SELECTION_V036.json` |
| digest | SHA-256（16 進小文字、64 文字）。**対象の実バイト列**（UTF-8、LF 改行）に対して計算する |
| 一次手段 | **OpenTimestamps**。`.ots` レシートを対象と同一ディレクトリに保存し、成果物に含める |
| 二次手段（併用） | 公開 Git リポジトリへのコミットと、そのコミットハッシュの記録 |
| 記録形式 | `ANCHORS.jsonl`。フィールド: `anchor_id`, `target_path`, `sha256`, `method`, `receipt_path`, `recorded_at_utc`, `external_ref` |
| 複数対象 | 正準 manifest（`path` と `sha256` の配列、キー昇順、UTF-8、LF）を作り、その manifest を anchor する |
| ラベル規約 | 外部手段を経ない digest 記録を anchor と呼ばない。内部のみは `TEMPORAL_PRECOMMITMENT_UNVERIFIED`、OpenTimestamps 確認後に `TEMPORAL_PRECOMMITMENT_OK`。真正性は `AUTHENTICITY_UNVERIFIED` |

**改行コードの変換（CRLF 化など）は digest を変える。** 転送後に digest を再計算し、
一致しない場合は anchor しない。

### 11.2 計装要件

- `INSTRUMENT_VERSION` 文字列。出力スキーマ変更でも bump。
- raw は全 run JSONL 保存、checkpoint は atomic write、両者の一致を毎回検証。
- **整合性フィールドはリテラルで書かない。測定値から導出する。**
- **検査に `assert` を使わない**（`python -O` で除去される。§B-2）。
  `if not cond: raise IntegrityError(...)` を用いる。
- 残差が 0 でなければ解釈より先に器械を疑う。

### 11.3 独立実装クロスチェック — 対象と許容差（本契約で凍結）

**判定基準は混合形とする。** 勾配成分・`P_t`・`a_t`・方向微分はゼロを取りうるため、
相対誤差のみでは検査不能である。

$$|a-b| \;\le\; \mathrm{atol} + \mathrm{rtol}\cdot\max(|a|,|b|)$$

anchor-2（較正前）に実行する。

| # | 対象 | 検査 | `atol` | `rtol` |
|---|---|---|---:|---:|
| 1 | LJ エネルギー・勾配 | 独立実装との値照合 | 1e−10 | 1e−12 |
| 1b | 同上 | 有限差分による方向微分 | 1e−6 | 2e−5 |
| 2 | 記述子の不変性 | 乱択 E(3) × S_38 作用下の `desc` | 1e−10 | 0 |
| 2b | 正準バイト列 ID | 往復同一性 | **バイト厳密一致** | — |
| 3 | replica-exchange 受理式 | 符号検査＋独立実装との一致 | 1e−12（符号は厳密） | 1e−12 |
| 4 | PES 予算カウンタ | 予算を意図的に枯渇させ `search_grad_used` と `incomplete` を確認 | **0（厳密一致）** | 0 |
| 5 | `Q_inst` ゲート述語 | 下記の三点検査 | **判定の厳密一致** | — |
| 6 | raw / checkpoint 一致検査 | 不整合を注入し例外送出を確認（**`python -O` 下でも**） | **例外が必ず送出されること** | — |
| 7 | `P_t` / `a_t` | 解析式との照合 | 1e−12 | 1e−12 |
| 7b | `accept_flip_rate` | 総当たり CRN 再生との照合 | **厳密一致** | — |
| 8 | 拘束項の不活性 | §8.4 の 2 参照構造で `kconf·Σ max(‖r_i−r̄‖−R_c,0)² = 0` | **厳密に 0** | — |

**`Q_inst` ゲートの二段検査:**

- **anchor-2 段階（テスト用閾値、本契約で固定）:** $g_{\rm test} = 10^{-4}$。
  ゲートは §1.5 の純粋述語として、`gnorm ∈ {0.9 g_test,\ g_test,\ 1.1 g_test}` かつ
  `complete ∈ {True, False}` の 6 通りで検査する。期待結果は

  | `complete` | `gnorm` | 期待 |
  |---|---|---|
  | True | `0.9 g_test` | **受理** |
  | True | `g_test` | **棄却**（条件は厳密 `<` であり、ちょうど閾値は通さない） |
  | True | `1.1 g_test` | 棄却 |
  | False | いずれも | 棄却 |

- **anchor-3 段階:** 契約 B で確定した `g_tol` / `g_tol_hill` が実装へ正しく bind されたことを
  **再検査**する。bind 検査に失敗した場合、スクリーニング段階へ進まない。

いずれかが許容差を外れた場合 `INSTRUMENT_CHECK_FAIL` とし、次段階へ進まない。

### 11.4 射影 Hessian の方法（本契約で確定・数値は契約 B）

1. 勾配の**中心差分**により `H` を構成する（差分幅 `ε_H` は契約 B）
2. **対称化**: `H ← (H + Hᵀ)/2`
3. **剛体モードの構成**: 3 並進と、重心まわりの 3 個の微小回転を解析的に構成する
4. 得た 6 本を **Gram–Schmidt / QR で正規直交化**し `Q ∈ R^{3n×6}` を作る。期待 rank は **6**
   （rank が 6 でなければ `INSTRUMENT_CHECK_FAIL`）
5. **射影**: `P = I − QQᵀ`、`H_proj = P H P`
6. `H_proj` の固有値を求め、剛体由来の数値的零固有値 6 個を**個数で**除く
7. 残る `3n−6` 個のうち最小固有値を報告する。負固有値・零固有値の**閾値**は契約 B

他の射影法（拘束付き固有値問題、質量加重、内部座標への変換等）は用いない。

---

## 12. 契約 B へ送る未決定項目（較正由来の**数値のみ**。方式選択を含まない）

1. `g_tol`, `g_tol_hill`, `diag_grad_budget`, `p_min`, `k_max`, `t_max_screen`, `f_sham_max`
2. `q_r` の桁数 `r`、hill 幅 `σ`、penalty 高さ `h`、`T_BH`
3. PT 梯子（`Tlo`, `Thi`, `M`）× 2 種（`A6` 用・`A7` 用）
4. `B1`/`B2` の `rms_desc_disp` 等価点（`E_kin` / `step`）、MD の `dt`、`n_md`
5. yoked sham の操作チェック等価性帯の数値
6. 昇格規則の入力閾値（`p_min`, `k_max`, `t_max_screen`, `δ_tie`）と同点時の tie-break 数値
7. `δ_tie`
8. `p_μ` の `n_traj`, `n_per_traj`, `τ_bh`, `τ_pt`
9. 射影 Hessian の差分幅 `ε_H`、負固有値・零固有値の閾値
10. `PYTHONHASHSEED`

**本契約で確定済みのため契約 B へ送らないもの:** `α`、bootstrap seed（anchor-1 から導出）、
`δ_equiv`、自己相関推定法、Hessian の対称化・射影法、burn-in／時間シフトの有無（不使用）、
`source × routing`（範囲外）、壁時計の扱い（独立予算にしない）、`g_test`、numpy/scipy 版。
